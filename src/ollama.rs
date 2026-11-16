//! Ollama-native API compatibility.
//!
//! A lot of harnesses are written against Ollama's own API rather than OpenAI's:
//! `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/ps`. They will
//! not talk to an OpenAI endpoint, so speaking only OpenAI quietly excludes
//! them. This translates the Ollama shapes to and from the OpenAI shapes Strata
//! already forwards, so one compute process serves both dialects.
//!
//! Two differences are easy to get wrong and are handled explicitly here:
//! Ollama carries tool-call `arguments` as a JSON *object* where OpenAI uses a
//! JSON *string*, and Ollama streams newline-delimited JSON rather than SSE.

use serde_json::{json, Map, Value};

/// Translate an Ollama `/api/chat` request into an OpenAI chat completion.
pub fn chat_to_openai(req: &Value) -> anyhow::Result<Value> {
    let mut out = Map::new();
    out.insert(
        "model".into(),
        req.get("model").cloned().unwrap_or_else(|| json!("")),
    );

    let Some(msgs) = req.get("messages").and_then(|v| v.as_array()) else {
        anyhow::bail!("messages must be an array");
    };

    let mut messages: Vec<Value> = Vec::new();
    for m in msgs {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let text = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let images = m.get("images").and_then(|v| v.as_array());

        // Ollama carries images as bare base64 strings beside the text. OpenAI
        // wants them as content parts with a data URI, so an image turn has to
        // become an array rather than a string.
        let content = match images {
            Some(imgs) if !imgs.is_empty() => {
                let mut parts: Vec<Value> = Vec::new();
                if !text.is_empty() {
                    parts.push(json!({ "type": "text", "text": text }));
                }
                for img in imgs {
                    if let Some(b64) = img.as_str() {
                        parts.push(json!({
                            "type": "image_url",
                            "image_url": { "url": format!("data:image/png;base64,{b64}") }
                        }));
                    }
                }
                json!(parts)
            }
            _ => json!(text),
        };

        let mut msg = Map::new();
        msg.insert("role".into(), json!(role));
        msg.insert("content".into(), content);

        // Ollama's tool_calls carry `arguments` as an object; OpenAI wants a
        // string. A harness round-tripping through here would otherwise send
        // arguments that the compute process cannot parse.
        if let Some(calls) = m.get("tool_calls").and_then(|v| v.as_array()) {
            let converted: Vec<Value> = calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let f = c.get("function").cloned().unwrap_or_else(|| json!({}));
                    let args = f.get("arguments").cloned().unwrap_or_else(|| json!({}));
                    let args_str = match &args {
                        Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_else(|_| "{}".into()),
                    };
                    json!({
                        "id": format!("call_{i}"),
                        "type": "function",
                        "function": {
                            "name": f.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                            "arguments": args_str,
                        }
                    })
                })
                .collect();
            msg.insert("tool_calls".into(), json!(converted));
        }

        messages.push(Value::Object(msg));
    }
    out.insert("messages".into(), json!(messages));

    if let Some(tools) = req.get("tools").and_then(|v| v.as_array()) {
        let converted: Vec<Value> = tools
            .iter()
            .map(|t| {
                // Ollama accepts the OpenAI tool shape directly, so this is a
                // pass-through that only fills in a missing wrapper.
                if t.get("type").is_some() {
                    t.clone()
                } else {
                    json!({ "type": "function", "function": t })
                }
            })
            .collect();
        out.insert("tools".into(), json!(converted));
    }

    if let Some(opts) = req.get("options") {
        for (from, to) in [
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("num_predict", "max_tokens"),
            ("seed", "seed"),
        ] {
            if let Some(v) = opts.get(from) {
                out.insert(to.into(), v.clone());
            }
        }
        if let Some(stop) = opts.get("stop") {
            out.insert("stop".into(), stop.clone());
        }
    }

    out.insert(
        "stream".into(),
        req.get("stream").cloned().unwrap_or(json!(true)),
    );
    // See the note in anthropic.rs: without this a reasoning model answers into
    // `reasoning_content` and the visible text is empty.
    out.insert(
        "chat_template_kwargs".into(),
        json!({ "enable_thinking": false }),
    );
    Ok(Value::Object(out))
}

/// Translate an Ollama `/api/generate` request into an OpenAI chat completion.
///
/// Ollama's generate is a raw completion, but the compute process is a chat
/// server, so the prompt becomes a single user turn. A `system` field, if
/// present, is kept as a system message rather than concatenated, so the
/// model's chat template still sees the boundary.
pub fn generate_to_openai(req: &Value) -> anyhow::Result<Value> {
    let mut out = Map::new();
    out.insert(
        "model".into(),
        req.get("model").cloned().unwrap_or_else(|| json!("")),
    );

    let mut messages: Vec<Value> = Vec::new();
    if let Some(system) = req.get("system").and_then(|v| v.as_str()) {
        if !system.is_empty() {
            messages.push(json!({ "role": "system", "content": system }));
        }
    }
    let prompt = req.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    messages.push(json!({ "role": "user", "content": prompt }));
    out.insert("messages".into(), json!(messages));

    if let Some(opts) = req.get("options") {
        for (from, to) in [
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("num_predict", "max_tokens"),
            ("seed", "seed"),
        ] {
            if let Some(v) = opts.get(from) {
                out.insert(to.into(), v.clone());
            }
        }
        if let Some(stop) = opts.get("stop") {
            out.insert("stop".into(), stop.clone());
        }
    }

    out.insert(
        "stream".into(),
        req.get("stream").cloned().unwrap_or(json!(true)),
    );
    out.insert(
        "chat_template_kwargs".into(),
        json!({ "enable_thinking": false }),
    );
    Ok(Value::Object(out))
}

/// One Ollama `/api/chat` streaming chunk from an OpenAI delta.
pub fn chat_chunk(delta: &Value, model: &str, done: bool) -> Value {
    let mut message = Map::new();
    message.insert("role".into(), json!("assistant"));
    // Fall back to `reasoning_content` so a reasoning model's stream is not a
    // run of empty chunks.
    let text = delta
        .get("content")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| delta.get("reasoning_content").and_then(|v| v.as_str()))
        .unwrap_or("");
    message.insert("content".into(), json!(text));

    // OpenAI streams tool calls as fragments with string arguments; Ollama
    // wants whole calls with object arguments. Fragments are passed through as
    // they arrive, with arguments parsed when they already form valid JSON.
    if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
        let converted: Vec<Value> = calls
            .iter()
            .map(|c| {
                let f = c.get("function").cloned().unwrap_or_else(|| json!({}));
                let args = f
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .and_then(|s| serde_json::from_str::<Value>(s).ok())
                    .unwrap_or_else(|| json!({}));
                json!({
                    "function": {
                        "name": f.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                        "arguments": args,
                    }
                })
            })
            .collect();
        message.insert("tool_calls".into(), json!(converted));
    }

    json!({
        "model": model,
        "created_at": now_iso(),
        "message": Value::Object(message),
        "done": done,
    })
}

/// One Ollama `/api/generate` streaming chunk from an OpenAI delta.
pub fn generate_chunk(delta: &Value, model: &str, done: bool) -> Value {
    json!({
        "model": model,
        "created_at": now_iso(),
        "response": delta
            .get("content")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .or_else(|| delta.get("reasoning_content").and_then(|v| v.as_str()))
            .unwrap_or(""),
        "done": done,
    })
}

/// The final chunk of a stream, carrying the accounting Ollama clients read.
pub fn final_chunk(model: &str, chat: bool, stop_reason: &str, usage: &Value) -> Value {
    let eval_count = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let prompt_count = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let mut out = Map::new();
    out.insert("model".into(), json!(model));
    out.insert("created_at".into(), json!(now_iso()));
    if chat {
        out.insert(
            "message".into(),
            json!({ "role": "assistant", "content": "" }),
        );
    } else {
        out.insert("response".into(), json!(""));
    }
    out.insert("done".into(), json!(true));
    out.insert("done_reason".into(), json!(stop_reason));
    out.insert("prompt_eval_count".into(), json!(prompt_count));
    out.insert("eval_count".into(), json!(eval_count));
    Value::Object(out)
}

/// A non-streaming `/api/chat` or `/api/generate` response body.
pub fn complete(
    model: &str,
    chat: bool,
    message: &Value,
    stop_reason: &str,
    usage: &Value,
) -> Value {
    let eval_count = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let prompt_count = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let mut out = Map::new();
    out.insert("model".into(), json!(model));
    out.insert("created_at".into(), json!(now_iso()));
    if chat {
        out.insert("message".into(), message.clone());
    } else {
        out.insert(
            "response".into(),
            json!(message
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")),
        );
    }
    out.insert("done".into(), json!(true));
    out.insert("done_reason".into(), json!(stop_reason));
    out.insert("prompt_eval_count".into(), json!(prompt_count));
    out.insert("eval_count".into(), json!(eval_count));
    Value::Object(out)
}

/// An entry in `/api/tags`, built from the registry's view of a model file.
pub fn tag_entry(id: &str, arch: &str, bytes: u64, context: usize, moe: bool) -> Value {
    json!({
        "name": id,
        "model": id,
        "modified_at": now_iso(),
        "size": bytes,
        "digest": format!("strata-{id}"),
        "details": {
            "format": "gguf",
            "family": arch,
            "parameter_size": if moe { "MoE" } else { "dense" },
            "quantization_level": "see `strata inspect`",
            "context_length": context,
        }
    })
}

/// RFC 3339 UTC, the shape Ollama stamps on every response.
fn now_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Days-to-civil, so this needs no date crate.
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_chat_messages_become_openai_messages() {
        let req = json!({
            "model": "m",
            "messages": [
                { "role": "system", "content": "be brief" },
                { "role": "user", "content": "hi" }
            ]
        });
        let out = chat_to_openai(&req).unwrap();
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["content"], "hi");
    }

    #[test]
    fn ollama_tool_arguments_are_objects_and_must_become_strings() {
        let req = json!({
            "model": "m",
            "messages": [{
                "role": "assistant",
                "content": "",
                "tool_calls": [{ "function": { "name": "read", "arguments": { "path": "a.rs" } } }]
            }]
        });
        let out = chat_to_openai(&req).unwrap();
        let args = out["messages"][0]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .expect("OpenAI wants a JSON string");
        assert_eq!(serde_json::from_str::<Value>(args).unwrap()["path"], "a.rs");
    }

    #[test]
    fn ollama_options_map_to_openai_sampling_fields() {
        let req = json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "x" }],
            "options": { "temperature": 0.2, "num_predict": 64, "stop": ["\n\n"] }
        });
        let out = chat_to_openai(&req).unwrap();
        assert_eq!(out["temperature"], 0.2);
        assert_eq!(out["max_tokens"], 64);
        assert_eq!(out["stop"][0], "\n\n");
    }

    #[test]
    fn generate_becomes_a_single_user_turn_with_system_kept_separate() {
        let req = json!({ "model": "m", "system": "be terse", "prompt": "why?" });
        let out = generate_to_openai(&req).unwrap();
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "why?");
    }

    #[test]
    fn images_become_openai_content_parts() {
        let req = json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "what is this?", "images": ["AAAA"] }]
        });
        let out = chat_to_openai(&req).unwrap();
        let parts = out["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert!(parts[1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
    }

    #[test]
    fn a_streaming_chunk_has_the_ollama_envelope() {
        let chunk = chat_chunk(&json!({ "content": "Hello" }), "m", false);
        assert_eq!(chunk["message"]["role"], "assistant");
        assert_eq!(chunk["message"]["content"], "Hello");
        assert_eq!(chunk["done"], false);
        assert!(chunk["created_at"].as_str().unwrap().ends_with('Z'));
    }

    #[test]
    fn the_final_chunk_reports_usage_in_ollamas_names() {
        let usage = json!({ "prompt_tokens": 12, "completion_tokens": 7 });
        let f = final_chunk("m", true, "stop", &usage);
        assert_eq!(f["done"], true);
        assert_eq!(f["prompt_eval_count"], 12);
        assert_eq!(f["eval_count"], 7);
    }

    #[test]
    fn timestamps_are_well_formed() {
        let s = now_iso();
        assert_eq!(s.len(), 20, "{s}");
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[10..11], "T");
        assert!(s.ends_with('Z'));
    }
}
