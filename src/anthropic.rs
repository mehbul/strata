//! Anthropic Messages API compatibility.
//!
//! Claude Code speaks `POST /v1/messages`, not OpenAI's chat completions, and
//! it does not fall back to the cloud when `ANTHROPIC_BASE_URL` points at
//! something that answers in the wrong shape - it errors out. So serving an
//! OpenAI endpoint is not enough to make Strata usable from Claude Code; the
//! request has to be translated on the way in and the response on the way out.
//!
//! The translation is deliberately lossy in one direction only: fields Strata
//! does not model are dropped rather than guessed at, because a wrong guess
//! inside a tool call is worse than an omission the model can work around.

use serde_json::{json, Map, Value};

/// Translate an Anthropic Messages request into an OpenAI chat completion.
///
/// The two APIs disagree about where the system prompt lives, how tool results
/// are carried, and what a tool schema is called. Everything else - sampling,
/// stop sequences - maps across with a rename.
pub fn request_to_openai(req: &Value) -> anyhow::Result<Value> {
    let mut out = Map::new();

    // Model is passed through; the server substitutes its own when empty.
    if let Some(model) = req.get("model").and_then(|v| v.as_str()) {
        out.insert("model".into(), json!(model));
    }

    let mut messages: Vec<Value> = Vec::new();

    // Anthropic carries the system prompt as a top-level field, either a string
    // or an array of content blocks. OpenAI wants it as the first message.
    if let Some(system) = req.get("system") {
        let text = blocks_to_text(system);
        if !text.is_empty() {
            messages.push(json!({ "role": "system", "content": text }));
        }
    }

    let Some(input) = req.get("messages").and_then(|v| v.as_array()) else {
        anyhow::bail!("messages must be an array");
    };

    for msg in input {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let content = msg.get("content");

        match role {
            "assistant" => {
                // An assistant turn can hold text and `tool_use` blocks. OpenAI
                // wants the text in `content` and the calls in `tool_calls`.
                let mut text = String::new();
                let mut calls: Vec<Value> = Vec::new();
                if let Some(blocks) = content.and_then(|v| v.as_array()) {
                    for b in blocks {
                        match b.get("type").and_then(|v| v.as_str()) {
                            Some("text") => {
                                text.push_str(b.get("text").and_then(|v| v.as_str()).unwrap_or(""))
                            }
                            Some("tool_use") => {
                                let id = b.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                let name = b.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                let input = b.get("input").cloned().unwrap_or_else(|| json!({}));
                                calls.push(json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        // OpenAI carries arguments as a JSON
                                        // *string*, not an object.
                                        "arguments": serde_json::to_string(&input)
                                            .unwrap_or_else(|_| "{}".into()),
                                    }
                                }));
                            }
                            _ => {}
                        }
                    }
                } else if let Some(s) = content.and_then(|v| v.as_str()) {
                    text.push_str(s);
                }

                let mut m = Map::new();
                m.insert("role".into(), json!("assistant"));
                m.insert(
                    "content".into(),
                    if text.is_empty() {
                        Value::Null
                    } else {
                        json!(text)
                    },
                );
                if !calls.is_empty() {
                    m.insert("tool_calls".into(), json!(calls));
                }
                messages.push(Value::Object(m));
            }
            _ => {
                // A user turn can hold text, images, and `tool_result` blocks.
                // Each tool result becomes its own OpenAI `tool` message, and any
                // text alongside it stays a user message.
                let mut text = String::new();
                let mut results: Vec<Value> = Vec::new();
                if let Some(blocks) = content.and_then(|v| v.as_array()) {
                    for b in blocks {
                        match b.get("type").and_then(|v| v.as_str()) {
                            Some("text") => {
                                text.push_str(b.get("text").and_then(|v| v.as_str()).unwrap_or(""))
                            }
                            Some("tool_result") => {
                                let id =
                                    b.get("tool_use_id").and_then(|v| v.as_str()).unwrap_or("");
                                results.push(json!({
                                    "role": "tool",
                                    "tool_call_id": id,
                                    "content": blocks_to_text(
                                        b.get("content").unwrap_or(&Value::Null)
                                    ),
                                }));
                            }
                            _ => {}
                        }
                    }
                } else if let Some(s) = content.and_then(|v| v.as_str()) {
                    text.push_str(s);
                }

                // Tool results go first: OpenAI expects the tool messages to
                // follow the assistant turn that called them, before any new
                // user text.
                messages.extend(results);
                if !text.is_empty() {
                    messages.push(json!({ "role": "user", "content": text }));
                }
            }
        }
    }

    out.insert("messages".into(), json!(messages));

    // Anthropic names the schema `input_schema`; OpenAI calls it `parameters`.
    if let Some(tools) = req.get("tools").and_then(|v| v.as_array()) {
        let converted: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                        "description": t.get("description").and_then(|v| v.as_str()).unwrap_or(""),
                        "parameters": t.get("input_schema").cloned().unwrap_or_else(|| json!({})),
                    }
                })
            })
            .collect();
        if !converted.is_empty() {
            out.insert("tools".into(), json!(converted));
        }
    }

    // `tool_choice` shapes differ: Anthropic uses {"type":"auto"|"any"|"tool"},
    // OpenAI uses "auto"|"required" or {"type":"function","function":{...}}.
    if let Some(choice) = req.get("tool_choice") {
        let kind = choice
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("auto");
        let mapped = match kind {
            "any" => json!("required"),
            "tool" => {
                let name = choice.get("name").and_then(|v| v.as_str()).unwrap_or("");
                json!({ "type": "function", "function": { "name": name } })
            }
            "none" => json!("none"),
            _ => json!("auto"),
        };
        out.insert("tool_choice".into(), mapped);
    }

    for (from, to) in [
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
    ] {
        if let Some(v) = req.get(from) {
            out.insert(to.into(), v.clone());
        }
    }
    if let Some(stop) = req.get("stop_sequences") {
        out.insert("stop".into(), stop.clone());
    }
    if let Some(stream) = req.get("stream") {
        out.insert("stream".into(), stream.clone());
    }

    // A reasoning model's chat template routes its output to `reasoning_content`
    // unless thinking is switched off explicitly, which leaves `content` empty
    // and makes a correct answer look like a blank one. The OpenAI handler sets
    // this too; a translator that skips it silently returns nothing.
    out.insert(
        "chat_template_kwargs".into(),
        json!({ "enable_thinking": false }),
    );

    Ok(Value::Object(out))
}

/// Flatten an Anthropic content value into plain text.
///
/// Content arrives as a string, or as an array of blocks where only `text`
/// blocks carry anything a chat message can hold. Images are dropped here: an
/// OpenAI text message has nowhere to put them, and inventing a placeholder
/// would be worse than leaving them out.
fn blocks_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| {
                if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                    b.get("text")
                        .and_then(|t| t.as_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Translate a completed OpenAI chat response into an Anthropic message.
pub fn response_to_anthropic(resp: &Value, model: &str) -> Value {
    let choice = resp
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first());

    let mut content: Vec<Value> = Vec::new();

    if let Some(message) = choice.and_then(|c| c.get("message")) {
        // Prefer `content`, but fall back to `reasoning_content`: a reasoning
        // model that was not told to suppress thinking puts its answer there,
        // and returning an empty content array makes a working turn look broken.
        let text = message
            .get("content")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                message
                    .get("reasoning_content")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
            });
        if let Some(text) = text {
            content.push(json!({ "type": "text", "text": text }));
        }
        if let Some(calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
            for call in calls {
                let f = call.get("function").cloned().unwrap_or_else(|| json!({}));
                // Arguments come back as a JSON string; Anthropic wants an
                // object. A model that emits malformed JSON gets an empty
                // object rather than a parse failure that kills the turn.
                let args = f
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .and_then(|s| serde_json::from_str::<Value>(s).ok())
                    .unwrap_or_else(|| json!({}));
                content.push(json!({
                    "type": "tool_use",
                    "id": call.get("id").and_then(|v| v.as_str()).unwrap_or("toolu_0"),
                    "name": f.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                    "input": args,
                }));
            }
        }
    }

    let finish = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|v| v.as_str())
        .unwrap_or("stop");
    let stop_reason = match finish {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        "content_filter" => "refusal",
        _ => "end_turn",
    };

    let usage = resp.get("usage").cloned().unwrap_or_else(|| json!({}));
    let input_tokens = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    json!({
        "id": resp.get("id").and_then(|v| v.as_str()).unwrap_or("msg_strata"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": { "input_tokens": input_tokens, "output_tokens": output_tokens },
    })
}

/// Turns a stream of OpenAI deltas into Anthropic SSE events.
///
/// The two streaming formats do not line up: OpenAI sends a delta per chunk
/// with no block structure, Anthropic brackets every block with a start and a
/// stop event and reports the stop reason separately at the end. This keeps the
/// state needed to emit that framing.
#[derive(Default)]
pub struct StreamTranslator {
    started: bool,
    block_index: usize,
    /// Which kind of block is currently open: `Some("text")`, `Some("tool_use")`,
    /// or `None` when nothing is open.
    open: Option<&'static str>,
    model: String,
    input_tokens: u64,
    output_tokens: u64,
    stop_reason: Option<&'static str>,
    /// Tool calls arrive spread across chunks, keyed by their index.
    tool_seen: Vec<bool>,
}

impl StreamTranslator {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            ..Default::default()
        }
    }

    /// Feed one OpenAI SSE payload. Returns the Anthropic events to emit.
    pub fn push(&mut self, chunk: &Value) -> Vec<(String, Value)> {
        let mut events = Vec::new();

        if !self.started {
            self.started = true;
            if let Some(u) = chunk.get("usage") {
                self.input_tokens = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
            }
            events.push((
                "message_start".into(),
                json!({
                    "type": "message_start",
                    "message": {
                        "id": chunk.get("id").and_then(|v| v.as_str()).unwrap_or("msg_strata"),
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": { "input_tokens": self.input_tokens, "output_tokens": 0 },
                    }
                }),
            ));
        }

        if let Some(usage) = chunk.get("usage") {
            if let Some(n) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                self.output_tokens = n;
            }
            if let Some(n) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                if n > 0 {
                    self.input_tokens = n;
                }
            }
        }

        let Some(choice) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
        else {
            return events;
        };

        if let Some(delta) = choice.get("delta") {
            // --- text
            if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    if self.open != Some("text") {
                        self.close_open(&mut events);
                        events.push((
                            "content_block_start".into(),
                            json!({
                                "type": "content_block_start",
                                "index": self.block_index,
                                "content_block": { "type": "text", "text": "" },
                            }),
                        ));
                        self.open = Some("text");
                    }
                    events.push((
                        "content_block_delta".into(),
                        json!({
                            "type": "content_block_delta",
                            "index": self.block_index,
                            "delta": { "type": "text_delta", "text": text },
                        }),
                    ));
                }
            }

            // --- tool calls, which arrive as fragments
            if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                for call in calls {
                    let idx = call.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    while self.tool_seen.len() <= idx {
                        self.tool_seen.push(false);
                    }
                    if !self.tool_seen[idx] {
                        self.tool_seen[idx] = true;
                        self.close_open(&mut events);
                        let f = call.get("function").cloned().unwrap_or_else(|| json!({}));
                        events.push((
                            "content_block_start".into(),
                            json!({
                                "type": "content_block_start",
                                "index": self.block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": call.get("id").and_then(|v| v.as_str())
                                        .unwrap_or("toolu_0"),
                                    "name": f.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                                    "input": {},
                                },
                            }),
                        ));
                        self.open = Some("tool_use");
                    }
                    if let Some(args) = call
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(|v| v.as_str())
                    {
                        if !args.is_empty() {
                            events.push((
                                "content_block_delta".into(),
                                json!({
                                    "type": "content_block_delta",
                                    "index": self.block_index,
                                    "delta": { "type": "input_json_delta", "partial_json": args },
                                }),
                            ));
                        }
                    }
                }
            }
        }

        if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            self.stop_reason = Some(match reason {
                "tool_calls" => "tool_use",
                "length" => "max_tokens",
                "content_filter" => "refusal",
                _ => "end_turn",
            });
        }

        events
    }

    /// Emit the closing events. Called once the upstream stream ends.
    pub fn finish(&mut self) -> Vec<(String, Value)> {
        let mut events = Vec::new();
        self.close_open(&mut events);
        events.push((
            "message_delta".into(),
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": self.stop_reason.unwrap_or("end_turn"),
                    "stop_sequence": Value::Null,
                },
                "usage": { "output_tokens": self.output_tokens },
            }),
        ));
        events.push(("message_stop".into(), json!({ "type": "message_stop" })));
        events
    }

    fn close_open(&mut self, events: &mut Vec<(String, Value)>) {
        if self.open.is_some() {
            events.push((
                "content_block_stop".into(),
                json!({ "type": "content_block_stop", "index": self.block_index }),
            ));
            self.block_index += 1;
            self.open = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_becomes_the_first_message() {
        let req = json!({
            "model": "m",
            "system": "You are helpful.",
            "messages": [{ "role": "user", "content": "hi" }],
        });
        let out = request_to_openai(&req).unwrap();
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are helpful.");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn a_tool_result_becomes_a_tool_message_before_the_user_text() {
        let req = json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "tool_result", "tool_use_id": "t1", "content": "42" },
                    { "type": "text", "text": "now what?" }
                ]
            }],
        });
        let out = request_to_openai(&req).unwrap();
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "t1");
        assert_eq!(msgs[0]["content"], "42");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn tool_schema_is_renamed_to_parameters() {
        let req = json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "x" }],
            "tools": [{
                "name": "read_file",
                "description": "Read a file",
                "input_schema": { "type": "object", "properties": { "path": { "type": "string" } } }
            }],
        });
        let out = request_to_openai(&req).unwrap();
        let t = &out["tools"][0]["function"];
        assert_eq!(t["name"], "read_file");
        assert_eq!(t["parameters"]["properties"]["path"]["type"], "string");
    }

    #[test]
    fn an_assistant_tool_use_becomes_openai_tool_calls_with_string_arguments() {
        let req = json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "read it" },
                { "role": "assistant", "content": [
                    { "type": "text", "text": "Reading." },
                    { "type": "tool_use", "id": "t1", "name": "read_file", "input": { "path": "a.rs" } }
                ]}
            ],
        });
        let out = request_to_openai(&req).unwrap();
        let m = &out["messages"][1];
        assert_eq!(m["content"], "Reading.");
        assert_eq!(m["tool_calls"][0]["id"], "t1");
        // OpenAI wants a JSON *string* here, not an object.
        let args = m["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert_eq!(serde_json::from_str::<Value>(args).unwrap()["path"], "a.rs");
    }

    #[test]
    fn a_tool_call_response_maps_to_a_tool_use_block() {
        let resp = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "t1",
                        "function": { "name": "read_file", "arguments": "{\"path\":\"a.rs\"}" }
                    }]
                }
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5 },
        });
        let out = response_to_anthropic(&resp, "m");
        assert_eq!(out["stop_reason"], "tool_use");
        assert_eq!(out["content"][0]["type"], "tool_use");
        assert_eq!(out["content"][0]["input"]["path"], "a.rs");
        assert_eq!(out["usage"]["input_tokens"], 10);
        assert_eq!(out["usage"]["output_tokens"], 5);
    }

    #[test]
    fn malformed_tool_arguments_do_not_kill_the_turn() {
        let resp = json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": { "tool_calls": [{
                    "id": "t1",
                    "function": { "name": "f", "arguments": "{not json" }
                }]}
            }],
        });
        let out = response_to_anthropic(&resp, "m");
        // An empty object, not a panic and not a dropped block.
        assert_eq!(out["content"][0]["input"], json!({}));
    }

    #[test]
    fn streaming_brackets_text_in_start_and_stop_events() {
        let mut t = StreamTranslator::new("m");
        let first = t.push(&json!({
            "id": "c1",
            "choices": [{ "delta": { "content": "Hello" } }]
        }));
        let names: Vec<&str> = first.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );

        let last = t.finish();
        let names: Vec<&str> = last.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec!["content_block_stop", "message_delta", "message_stop"]
        );
    }

    #[test]
    fn streaming_tool_calls_open_a_tool_use_block_once() {
        let mut t = StreamTranslator::new("m");
        t.push(&json!({"choices": [{"delta": {"content": "ok"}}]}));
        // First fragment opens the block.
        let a = t.push(&json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "t1", "function": {"name": "f", "arguments": "{\"a\""}}
        ]}}]}));
        let starts = a.iter().filter(|(n, _)| n == "content_block_start").count();
        assert_eq!(starts, 1, "the block opens once");

        // Second fragment only appends arguments.
        let b = t.push(&json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ":1}"}}
        ]}}]}));
        let starts = b.iter().filter(|(n, _)| n == "content_block_start").count();
        assert_eq!(starts, 0, "no second block for the same call");
        let deltas = b.iter().filter(|(n, _)| n == "content_block_delta").count();
        assert_eq!(deltas, 1);
    }
}
