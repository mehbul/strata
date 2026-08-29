//! Context compaction.
//!
//! The compute process is started with a fixed context window, and a
//! conversation that outgrows it does not fail gracefully: llama.cpp either
//! rejects the prompt or silently drops the front of it, taking the system
//! prompt and the statement of the task with it. Compaction replaces the
//! middle of a conversation with a summary the model writes itself, so the
//! oldest turns cost a few hundred tokens instead of several thousand and the
//! space they free is reused by new ones.
//!
//! Two properties matter more here than the prose of any one summary:
//!
//! * **It happens rarely.** Summarising costs a full forward pass. A scheme
//!   that trimmed a little every turn would add that cost to every turn, so
//!   compaction instead swallows `DROP_FRACTION` of the budget at once and
//!   leaves the rest as headroom for the turns that follow.
//! * **The result is stable.** llama.cpp reuses its KV cache for the longest
//!   common prefix of consecutive prompts, so rewriting history costs a full
//!   reprocess of the prompt. Summaries are cached by a digest of the messages
//!   they replace: a conversation is summarised once, and every later turn
//!   re-sends a byte-identical prefix that the KV cache still holds.

use serde::Serialize;
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::server::Msg;

/// Compact once the prompt passes this fraction of the usable budget.
const TRIGGER_FRACTION: f32 = 0.80;
/// How much of the budget one compaction swallows. Boundaries sit at multiples
/// of this many tokens counted from the *start* of the conversation, so the
/// remainder left in front of the model after a compaction is roughly
/// `TRIGGER_FRACTION - DROP_FRACTION` of the budget plus the summary.
const DROP_FRACTION: f32 = 0.50;
/// How much the summary is allowed to compress what it replaces. The summary
/// budget scales with the conversation being dropped rather than being a fixed
/// number of tokens: one compaction at `--ctx 8192` drops about 4,000 tokens,
/// the same compaction at `--ctx 65536` drops about 32,000, and a summary
/// sized for the first loses almost everything of the second.
const COMPRESSION_RATIO: usize = 8;
/// Floor, so a small compaction still gets room for the headings.
const MIN_SUMMARY_TOKENS: usize = 512;
/// Ceiling, because the summary is generated a token at a time and the whole
/// turn waits on it. At ~42 tok/s this is about a hundred seconds.
const MAX_SUMMARY_TOKENS: usize = 4096;
/// Chat-template overhead per message: role tags and turn delimiters, which
/// `/tokenize` never sees because it is handed message content alone.
const PER_MESSAGE_OVERHEAD: usize = 8;
/// Slack for the template preamble and for the tokenizer disagreeing with
/// itself about a boundary.
const SAFETY_TOKENS: usize = 128;
/// A request may reserve no more than this fraction of the context for its own
/// reply. `max_tokens` is an upper bound clients set generously - 4096 of an
/// 8192 context is a common default - and honouring it literally would leave
/// almost nothing for the conversation.
const MAX_RESERVE_FRACTION: f32 = 0.34;
/// Bounds on how long a summary may take, the allowance itself derived from
/// the work. A fixed 180s was set when a summary was 768 tokens; now that the
/// budget scales with the context, one measured here already took 121s — inside
/// the margin of error of the thing it was meant to bound. Timing out does not
/// merely cost time, it discards the history being summarised.
const SUMMARY_TIMEOUT_FLOOR_S: u64 = 120;
const SUMMARY_TIMEOUT_CEILING_S: u64 = 600;

/// Separates the compacted history from whatever else is in the system message.
/// A second compaction finds the previous summary by this marker, folds it into
/// the new one and replaces it, so summaries never nest.
const MARKER: &str = "\n\n--- Earlier in this conversation (compacted) ---\n";

const SUMMARY_PROMPT: &str = concat!(
    "You are compacting the earlier part of a working session so it can be ",
    "dropped from the context window. Write a dense factual record for whoever ",
    "picks the work up with only your notes and the most recent messages.\n\n",
    "Cover these headings, omitting any you have nothing for:\n",
    "GOAL - what the user is trying to build or find out, in their own terms.\n",
    "DECISIONS - choices made, and the reason each was made.\n",
    "STATE - files, functions, commands, APIs and identifiers touched, with ",
    "their exact names, and what was done to each.\n",
    "FACTS - specific values established: numbers, paths, versions, errors ",
    "seen, results measured. Copy them exactly; they cannot be looked up again.\n",
    "OPEN - what was unresolved or explicitly deferred.\n",
    "NEXT - the step that was about to be taken.\n\n",
    "Write only the record: no preamble, no closing remark, no offer to help. ",
    "Prefer exact names and values over readable prose. This replaces the ",
    "original text, so anything you leave out is lost."
);

// ------------------------------------------------------------------ reporting

/// What one compaction did, for `/compact` and the console.
#[derive(Clone, Serialize)]
pub struct Event {
    /// Prompt size that tripped the trigger.
    pub before_tokens: usize,
    /// Prompt size after the summary replaced the middle.
    pub after_tokens: usize,
    pub summarised_messages: usize,
    pub kept_messages: usize,
    /// True when this conversation was already summarised and the cached
    /// summary was reused - the common case, and the cheap one.
    pub reused: bool,
    pub took_ms: u64,
}

#[derive(Default)]
pub struct Stats {
    /// Summaries actually generated.
    pub summarised: AtomicU64,
    /// Turns served from a cached summary.
    pub reused: AtomicU64,
    /// Prompt tokens compaction has removed over the life of the process.
    pub reclaimed: AtomicU64,
    /// Size of the most recent prompt, and the budget it was measured against,
    /// so the console can draw how full the window currently is.
    pub last_prompt: AtomicU64,
    pub last_budget: AtomicU64,
    pub last: Mutex<Option<Event>>,
}

/// The outcome of preparing one request.
pub enum Prepared {
    /// Fits as it stands. Carries the measured prompt size.
    AsIs { tokens: usize },
    /// Rewritten to fit.
    Compacted { messages: Vec<Msg>, event: Event },
}

// ------------------------------------------------------------------ compactor

pub struct Compactor {
    /// The context the compute process was started with - the hard ceiling.
    ctx: usize,
    upstream: String,
    client: reqwest::Client,
    enabled: bool,
    /// Summaries by digest of the messages they replace.
    summaries: Mutex<HashMap<u64, String>>,
    /// Token counts by digest of message content. A conversation re-sends its
    /// whole history every turn, so after the first turn only the new messages
    /// reach the tokenizer.
    counts: Mutex<HashMap<u64, usize>>,
    pub stats: Stats,
}

impl Compactor {
    pub fn new(ctx: usize, upstream: String, client: reqwest::Client, enabled: bool) -> Self {
        Self {
            ctx,
            upstream,
            client,
            enabled,
            summaries: Mutex::new(HashMap::new()),
            counts: Mutex::new(HashMap::new()),
            stats: Stats::default(),
        }
    }

    /// Tokens available to the conversation once the reply is accounted for.
    pub fn budget(&self, reserve: usize) -> usize {
        let cap = (self.ctx as f32 * MAX_RESERVE_FRACTION) as usize;
        self.ctx.saturating_sub(reserve.min(cap) + SAFETY_TOKENS)
    }

    /// The prompt size that trips compaction, and the size it cuts back to,
    /// for a request reserving `reserve` tokens for its reply.
    pub fn thresholds(&self, reserve: usize) -> (usize, usize) {
        let budget = self.budget(reserve);
        let after =
            budget as f32 * (1.0 - DROP_FRACTION) + summary_budget(budget) as f32;
        ((budget as f32 * TRIGGER_FRACTION) as usize, after as usize)
    }

    /// Measure a request and, if it will not fit, rewrite it so it does.
    pub async fn prepare(&self, messages: &[Msg], reserve: usize) -> Prepared {
        let counts = self.measure(messages).await;
        let total: usize = counts.iter().sum();
        let budget = self.budget(reserve);
        let trigger = (budget as f32 * TRIGGER_FRACTION) as usize;
        self.stats.last_budget.store(budget as u64, Ordering::Relaxed);
        self.stats.last_prompt.store(total as u64, Ordering::Relaxed);
        if !self.enabled || budget == 0 || total <= trigger {
            return Prepared::AsIs { tokens: total };
        }

        let started = Instant::now();
        let head_len = messages.iter().take_while(|m| m.role == "system").count();
        let (system_base, prior) = if head_len > 0 {
            split_marker(&messages[head_len - 1].content)
        } else {
            (String::new(), None)
        };

        let summary_tokens = summary_budget(budget);
        let start = boundary(&counts, messages, head_len, budget, trigger, summary_tokens);
        let middle = &messages[head_len..start];
        if middle.is_empty() {
            // Nothing between the system prompt and the tail to reclaim: the
            // last exchange alone is over budget. Rewriting cannot help, so
            // pass it through and let the compute process report the overflow
            // rather than invent a truncation of our own.
            return Prepared::AsIs { tokens: total };
        }

        let digest = digest_of(prior.as_deref(), middle);
        let cached = self.summaries.lock().unwrap_or_else(|e| e.into_inner()).get(&digest).cloned();
        let reused = cached.is_some();
        let summary = match cached {
            Some(s) => s,
            None => match self.summarise(prior.as_deref(), middle, summary_tokens).await {
                // Only a real summary is cached. Caching the failure note would
                // make one timeout permanent for that conversation: every later
                // turn would find it, reuse it, and never try again - the whole
                // history lost to a transient slow response.
                Some(s) => {
                    self.summaries
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .insert(digest, s.clone());
                    s
                }
                None => lost_note(middle.len()),
            },
        };

        // Earlier system messages unchanged, then one carrying the client's
        // system prompt plus the summary. Folding the summary into the system
        // message rather than inserting a free-standing one keeps user and
        // assistant turns alternating, which some chat templates require.
        let mut system = system_base;
        if !system.is_empty() {
            system.push_str(MARKER);
        }
        system.push_str(&summary);

        let mut out: Vec<Msg> = Vec::with_capacity(head_len + 1 + messages.len() - start);
        out.extend(messages[..head_len.saturating_sub(1)].iter().cloned());
        out.push(Msg { role: "system".into(), content: system });
        out.extend(messages[start..].iter().cloned());

        let after: usize = self.measure(&out).await.iter().sum();
        let event = Event {
            before_tokens: total,
            after_tokens: after,
            summarised_messages: middle.len(),
            kept_messages: messages.len() - start,
            reused,
            took_ms: started.elapsed().as_millis() as u64,
        };

        if reused {
            self.stats.reused.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.summarised.fetch_add(1, Ordering::Relaxed);
        }
        self.stats.reclaimed.fetch_add(total.saturating_sub(after) as u64, Ordering::Relaxed);
        self.stats.last_prompt.store(after as u64, Ordering::Relaxed);
        *self.stats.last.lock().unwrap_or_else(|e| e.into_inner()) = Some(event.clone());

        Prepared::Compacted { messages: out, event }
    }

    /// Current settings and what compaction has done so far.
    pub fn report(&self) -> Value {
        let last = self.stats.last.lock().unwrap_or_else(|e| e.into_inner()).clone();
        json!({
            "enabled": self.enabled,
            "ctx": self.ctx,
            "trigger_fraction": TRIGGER_FRACTION,
            "drop_fraction": DROP_FRACTION,
            "summary_tokens": summary_budget(
                self.stats.last_budget.load(Ordering::Relaxed).max(1) as usize,
            ),
            "prompt_tokens": self.stats.last_prompt.load(Ordering::Relaxed),
            "budget": self.stats.last_budget.load(Ordering::Relaxed),
            "summarised": self.stats.summarised.load(Ordering::Relaxed),
            "reused": self.stats.reused.load(Ordering::Relaxed),
            "tokens_reclaimed": self.stats.reclaimed.load(Ordering::Relaxed),
            "last": last,
        })
    }

    // ------------------------------------------------------------- internals

    /// Ask the model to summarise the messages being dropped. A failure here is
    /// not fatal: the caller still needs a prompt that fits, so fall back to a
    /// truthful note that the text is gone rather than to silence.
    /// `None` when the model could not be reached or produced nothing, so the
    /// caller knows not to cache the result.
    async fn summarise(
        &self,
        prior: Option<&str>,
        middle: &[Msg],
        max_tokens: usize,
    ) -> Option<String> {
        let mut transcript = String::new();
        if let Some(p) = prior {
            transcript.push_str("[record of still earlier messages]\n");
            transcript.push_str(p);
            transcript.push_str("\n\n");
        }
        for m in middle {
            transcript.push_str(&m.role.to_uppercase());
            transcript.push_str(":\n");
            transcript.push_str(&m.content);
            transcript.push_str("\n\n");
        }

        let body = json!({
            "messages": [
                { "role": "system", "content": SUMMARY_PROMPT },
                { "role": "user", "content": transcript },
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": false,
        });
        let sent = self
            .client
            .post(format!("{}/v1/chat/completions", self.upstream))
            .json(&body)
            .timeout(summary_timeout(transcript.len() / 4, max_tokens))
            .send()
            .await;
        let text = match sent {
            Ok(r) if r.status().is_success() => r
                .json::<Value>()
                .await
                .ok()
                .and_then(|v| {
                    v.pointer("/choices/0/message/content")
                        .and_then(|c| c.as_str())
                        .map(str::to_owned)
                })
                .unwrap_or_default(),
            _ => String::new(),
        };
        let text = text.trim();
        (!text.is_empty()).then(|| text.to_string())
    }

    /// Token count per message, including template overhead.
    async fn measure(&self, messages: &[Msg]) -> Vec<usize> {
        let digests: Vec<u64> = messages.iter().map(digest_msg).collect();
        let mut out = vec![0usize; messages.len()];
        let mut missing: Vec<usize> = Vec::new();
        {
            let cache = self.counts.lock().unwrap_or_else(|e| e.into_inner());
            for (i, d) in digests.iter().enumerate() {
                match cache.get(d) {
                    Some(n) => out[i] = *n,
                    None => missing.push(i),
                }
            }
        }

        if !missing.is_empty() {
            // One tokenize call per unseen message, all in flight together.
            // After the first turn of a conversation only the newest messages
            // are unseen.
            let fetched =
                futures::future::join_all(missing.iter().map(|&i| self.tokenize(&messages[i].content)))
                    .await;
            let mut cache = self.counts.lock().unwrap_or_else(|e| e.into_inner());
            for (&i, n) in missing.iter().zip(fetched) {
                out[i] = n;
                cache.insert(digests[i], n);
            }
            // Keyed by content, so entries cannot go stale, only accumulate.
            // Dropping the lot costs one round of tokenize calls to rebuild.
            if cache.len() > 4096 {
                cache.clear();
            }
        }

        for n in out.iter_mut() {
            *n += PER_MESSAGE_OVERHEAD;
        }
        out
    }

    /// Count tokens with the model's own tokenizer, estimating if it cannot be
    /// reached. The estimate is deliberately high: undercounting overflows the
    /// context, overcounting only compacts a little sooner than needed.
    async fn tokenize(&self, text: &str) -> usize {
        let sent = self
            .client
            .post(format!("{}/tokenize", self.upstream))
            .json(&json!({ "content": text }))
            .timeout(Duration::from_secs(20))
            .send()
            .await;
        if let Ok(r) = sent {
            if r.status().is_success() {
                if let Ok(v) = r.json::<Value>().await {
                    if let Some(t) = v.get("tokens").and_then(|t| t.as_array()) {
                        return t.len();
                    }
                }
            }
        }
        text.chars().count().div_ceil(3)
    }
}

// --------------------------------------------------------------------- helpers

/// Where to cut: everything from `head_len` up to the returned index is
/// summarised, everything from it onward is kept verbatim.
///
/// The obvious rule - keep the newest N tokens, summarise the rest - puts the
/// cut a fixed distance from the *end*, so every new turn shifts it and every
/// turn needs a fresh summary. This walks forward from the start instead and
/// cuts at a multiple of `DROP_FRACTION` of the budget, which depends only on
/// messages that are already settled. Appending a turn does not move the cut,
/// so the same summary is reused and the prompt keeps the byte-identical prefix
/// that llama.cpp's KV cache is holding. Only when the tail has grown back past
/// the trigger does the cut advance to the next multiple, costing one summary.
fn boundary(
    counts: &[usize],
    messages: &[Msg],
    head_len: usize,
    budget: usize,
    trigger: usize,
    summary_tokens: usize,
) -> usize {
    let step = ((budget as f32 * DROP_FRACTION) as usize).max(1);
    // What the summary itself will cost once it replaces the messages.
    let summary_cost = summary_tokens + PER_MESSAGE_OVERHEAD;
    let total: usize = counts.iter().sum();

    let mut cut = head_len;
    let mut covered = 0usize;
    let mut rung = step;
    loop {
        while cut < messages.len() && covered < rung {
            covered += counts[cut];
            cut += 1;
        }
        // Land on a user message so the kept tail opens on a question rather
        // than on the back half of an exchange.
        while cut < messages.len() && messages[cut].role != "user" {
            covered += counts[cut];
            cut += 1;
        }
        // Never summarise the message being answered.
        if cut >= messages.len() {
            return messages.len().saturating_sub(1).max(head_len);
        }
        if total - covered + summary_cost <= trigger {
            return cut;
        }
        rung += step;
    }
}

/// Stands in for history that could not be summarised. It says the content is
/// gone rather than inventing a gist of it, and it is never cached.
fn lost_note(dropped: usize) -> String {
    format!(
        "{dropped} earlier messages were dropped to fit the context window, and \
         summarising them did not complete. Their content is not available: ask \
         the user rather than assuming what they said."
    )
}

/// How long to allow for reading `transcript_tokens` and writing `max_tokens`.
///
/// Pessimistic rates on purpose — a timeout is a backstop against an upstream
/// that has stopped answering, not a schedule — but bounded, so a wedged
/// process cannot hold a turn open indefinitely.
fn summary_timeout(transcript_tokens: usize, max_tokens: usize) -> Duration {
    let seconds = 60 + transcript_tokens as u64 / 40 + max_tokens as u64 / 4;
    Duration::from_secs(seconds.clamp(SUMMARY_TIMEOUT_FLOOR_S, SUMMARY_TIMEOUT_CEILING_S))
}

/// Tokens the summary may use, for a conversation with this much budget.
///
/// One compaction replaces roughly `DROP_FRACTION` of the budget, so the
/// summary gets that divided by `COMPRESSION_RATIO` - a fixed size would mean
/// a 5:1 squeeze at a small context and a 43:1 squeeze at a large one.
fn summary_budget(budget: usize) -> usize {
    let dropped = (budget as f32 * DROP_FRACTION) as usize;
    (dropped / COMPRESSION_RATIO).clamp(MIN_SUMMARY_TOKENS, MAX_SUMMARY_TOKENS)
}

/// Split a system message into the part the client wrote and any summary a
/// previous compaction appended to it.
fn split_marker(content: &str) -> (String, Option<String>) {
    match content.split_once(MARKER) {
        Some((base, summary)) => (base.to_string(), Some(summary.to_string())),
        None => (content.to_string(), None),
    }
}

fn digest_msg(m: &Msg) -> u64 {
    let mut h = DefaultHasher::new();
    m.role.hash(&mut h);
    m.content.hash(&mut h);
    h.finish()
}

fn digest_of(prior: Option<&str>, middle: &[Msg]) -> u64 {
    let mut h = DefaultHasher::new();
    prior.hash(&mut h);
    for m in middle {
        m.role.hash(&mut h);
        m.content.hash(&mut h);
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> Msg {
        Msg { role: role.into(), content: content.into() }
    }

    #[test]
    fn marker_round_trips() {
        let (base, prior) = split_marker("be terse");
        assert_eq!(base, "be terse");
        assert!(prior.is_none());

        let joined = format!("be terse{MARKER}GOAL - ship it");
        let (base, prior) = split_marker(&joined);
        assert_eq!(base, "be terse");
        assert_eq!(prior.as_deref(), Some("GOAL - ship it"));
    }

    #[test]
    fn digest_tracks_exactly_what_it_replaces() {
        let a = [msg("user", "one"), msg("assistant", "two")];
        let b = [msg("user", "one"), msg("assistant", "three")];
        assert_eq!(digest_of(None, &a), digest_of(None, &a));
        assert_ne!(digest_of(None, &a), digest_of(None, &b));
        assert_ne!(digest_of(None, &a), digest_of(Some("earlier"), &a));
    }

    /// A conversation of `pairs` exchanges, every message the same size.
    fn conversation(pairs: usize) -> (Vec<Msg>, Vec<usize>) {
        let mut m = vec![msg("system", "be terse")];
        for i in 0..pairs {
            m.push(msg("user", &format!("question {i}")));
            m.push(msg("assistant", &format!("answer {i}")));
        }
        let counts = vec![100usize; m.len()];
        (m, counts)
    }

    #[test]
    fn boundary_does_not_move_when_the_conversation_grows() {
        let (mut m, mut counts) = conversation(30);
        let (budget, trigger) = (5000, 4000);
        let first = boundary(&counts, &m, 1, budget, trigger, 768);
        assert!(first > 1 && first < m.len(), "cut lands inside the conversation");
        assert_eq!(m[first].role, "user", "the kept tail opens on a user message");

        // Two more turns arrive. The cut is measured from the start, so it must
        // stay put - that is what keeps the summary cached and the KV prefix
        // intact. It may only advance once the tail has grown past the trigger.
        for _ in 0..2 {
            m.push(msg("user", "another question"));
            m.push(msg("assistant", "another answer"));
            counts.push(100);
            counts.push(100);
        }
        assert_eq!(boundary(&counts, &m, 1, budget, trigger, 768), first);
    }

    #[test]
    fn boundary_advances_once_the_tail_outgrows_the_trigger() {
        let (short, short_counts) = conversation(30);
        let (long, long_counts) = conversation(120);
        let (budget, trigger) = (5000, 4000);
        assert!(
            boundary(&long_counts, &long, 1, budget, trigger, 768)
                > boundary(&short_counts, &short, 1, budget, trigger, 768)
        );
    }

    #[test]
    fn boundary_never_swallows_the_message_being_answered() {
        let (m, counts) = conversation(2);
        // A trigger nothing can satisfy: the cut must still stop short of the end.
        let cut = boundary(&counts, &m, 1, 5000, 0, 768);
        assert!(cut < m.len());
    }

    #[test]
    fn summary_budget_tracks_what_is_being_dropped() {
        // The squeeze should stay near COMPRESSION_RATIO across contexts rather
        // than worsening as the window grows, which is what a fixed size did:
        // 5:1 at ctx 8192 and 43:1 at ctx 65536.
        for budget in [8_000usize, 61_312, 65_344] {
            let dropped = (budget as f32 * DROP_FRACTION) as usize;
            let squeeze = dropped / summary_budget(budget);
            assert!(
                squeeze <= COMPRESSION_RATIO + 1,
                "budget {budget}: {dropped} tokens into {} is {squeeze}:1",
                summary_budget(budget)
            );
        }
        // Floor and ceiling still hold at the extremes.
        assert_eq!(summary_budget(100), MIN_SUMMARY_TOKENS);
        assert_eq!(summary_budget(10_000_000), MAX_SUMMARY_TOKENS);
    }

    #[test]
    fn the_summary_allowance_covers_the_work_it_was_measured_doing() {
        // The compaction measured on this machine: ~32,700 tokens read, a
        // 4,075-token summary written, 121 seconds. A fixed 180s left almost
        // no margin over that, and exceeding it discards the history.
        let allowed = summary_timeout(32_700, 4_075);
        assert!(allowed >= Duration::from_secs(240), "got {allowed:?}");
        assert!(allowed <= Duration::from_secs(SUMMARY_TIMEOUT_CEILING_S));
    }

    #[test]
    fn a_small_summary_still_gets_a_usable_floor() {
        assert_eq!(summary_timeout(0, 0), Duration::from_secs(SUMMARY_TIMEOUT_FLOOR_S));
    }

    #[test]
    fn reserve_is_capped_at_a_third_of_context() {
        let c = Compactor::new(8192, String::new(), reqwest::Client::new(), true);
        // A client asking to reserve half the window gets a third.
        assert_eq!(c.budget(4096), 8192 - (8192.0 * MAX_RESERVE_FRACTION) as usize - SAFETY_TOKENS);
        // A modest request is honoured as asked.
        assert_eq!(c.budget(512), 8192 - 512 - SAFETY_TOKENS);
    }
}
