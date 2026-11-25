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
        // A reasoning turn carries its visible text in `content` and its
        // private trace in `thinking`. OpenAI calls the second field
        // `reasoning_content`; dropping it leaves the model answering as if it
        // had never thought.
        if let Some(thinking) = m.get("thinking").and_then(|v| v.as_str()) {
            if !thinking.is_empty() {
                msg.insert("reasoning_content".into(), json!(thinking));
            }
        }
        if let Some(name) = m.get("tool_name").and_then(|v| v.as_str()) {
            msg.insert("name".into(), json!(name));
        }

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
            ("top_k", "top_k"),
            ("num_predict", "max_tokens"),
            ("seed", "seed"),
            ("repeat_penalty", "repeat_penalty"),
            ("presence_penalty", "presence_penalty"),
            ("frequency_penalty", "frequency_penalty"),
            ("min_p", "min_p"),
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
    apply_thinking(req, &mut out);
    if let Some(format) = req.get("format") {
        out.insert(
            "response_format".into(),
            crate::dialects::response_format(format),
        );
    }
    Ok(Value::Object(out))
}

/// `think` and `format` are Ollama request fields, not options. Thinking was
/// previously forced off: that stopped reasoning models writing their trace
/// into `reasoning_content` and leaving `content` empty, but it also ignored a
/// client that explicitly asked for thinking. Levels are mapped to on/off
/// because the compute process has no per-level switch.
fn apply_thinking(req: &Value, out: &mut Map<String, Value>) {
    let enabled = match req.get("think") {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !matches!(s.as_str(), "off" | "none" | "false"),
        _ => false,
    };
    out.insert(
        "chat_template_kwargs".into(),
        json!({ "enable_thinking": enabled }),
    );
    if enabled {
        if let Some(level) = req.get("think").and_then(|v| v.as_str()) {
            // Ollama's public levels are low/medium/high/max. The chat
            // templates on this machine's models accept low/medium/xhigh and
            // raise on anything else, so map the upper end rather than passing
            // a string through that aborts the template.
            let accepted = match level {
                "low" => "low",
                "medium" => "medium",
                "high" | "max" => "xhigh",
                _ => "",
            };
            if !accepted.is_empty() {
                out.insert("reasoning_effort".into(), json!(accepted));
            }
        }
    }
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
    let raw = req.get("raw").and_then(Value::as_bool).unwrap_or(false);
    // `raw` means the caller already wrote the templated prompt, so Strata must
    // not add a system turn of its own. llama.cpp still applies its chat
    // template; disabling that would require a raw /completion endpoint.
    if !raw {
        if let Some(system) = req.get("system").and_then(|v| v.as_str()) {
            if !system.is_empty() {
                messages.push(json!({ "role": "system", "content": system }));
            }
        }
    }
    let mut prompt = req
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    // Ollama's `suffix` is the text after the generated span. A chat compute
    // process has no insertion point, so append it as a prompt hint rather than
    // dropping the field.
    if let Some(suffix) = req.get("suffix").and_then(Value::as_str) {
        if !suffix.is_empty() {
            prompt.push('\n');
            prompt.push_str(suffix);
        }
    }
    let content = match req.get("images").and_then(Value::as_array) {
        Some(images) if !images.is_empty() => {
            let mut parts = vec![json!({ "type": "text", "text": prompt })];
            for image in images.iter().filter_map(Value::as_str) {
                parts.push(json!({
                    "type": "image_url",
                    "image_url": { "url": format!("data:image/png;base64,{image}") }
                }));
            }
            json!(parts)
        }
        _ => json!(prompt),
    };
    messages.push(json!({ "role": "user", "content": content }));
    out.insert("messages".into(), json!(messages));

    if let Some(opts) = req.get("options") {
        for (from, to) in [
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("num_predict", "max_tokens"),
            ("seed", "seed"),
            ("repeat_penalty", "repeat_penalty"),
            ("presence_penalty", "presence_penalty"),
            ("frequency_penalty", "frequency_penalty"),
            ("min_p", "min_p"),
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
    apply_thinking(req, &mut out);
    if let Some(format) = req.get("format") {
        out.insert(
            "response_format".into(),
            crate::dialects::response_format(format),
        );
    }
    Ok(Value::Object(out))
}

/// One Ollama `/api/chat` streaming chunk from an OpenAI delta.
pub fn chat_chunk(delta: &Value, model: &str, done: bool) -> Value {
    let mut message = Map::new();
    message.insert("role".into(), json!("assistant"));
    let text = delta.get("content").and_then(Value::as_str).unwrap_or("");
    let thinking = delta
        .get("reasoning_content")
        .and_then(Value::as_str)
        .or_else(|| delta.get("reasoning").and_then(Value::as_str))
        .unwrap_or("");
    message.insert("content".into(), json!(text));
    // Ollama clients read a reasoning model's trace from `message.thinking`.
    if !thinking.is_empty() {
        message.insert("thinking".into(), json!(thinking));
    }

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
            .or_else(|| delta.get("reasoning").and_then(|v| v.as_str()))
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
    let cached = usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    out.insert("prompt_eval_cached_count".into(), json!(cached));
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
    let cached = usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    out.insert("prompt_eval_cached_count".into(), json!(cached));
    out.insert("eval_count".into(), json!(eval_count));
    Value::Object(out)
}

/// The `details` object shared by `/api/tags`, `/api/ps` and `/api/show`.
///
/// Every field here comes from the GGUF header or the file itself. The one
/// derived value is `parameter_size`, which is a sum of tensor element counts
/// and is labelled `derived` in the docs.
pub fn details(e: &crate::registry::ModelEntry) -> Value {
    json!({
        "parent_model": "model.gguf",
        "format": "gguf",
        "family": e.arch,
        "families": e.families,
        "parameter_size": parameter_size(e.parameter_count),
        "quantization_level": e.quantization_level,
        "context_length": e.context,
        "embedding_length": e.embedding,
        "vocab_size": e.vocab,
        "runner": "llamacpp",
    })
}

/// What this Strata instance can do with the model, derived from the model
/// directory rather than guessed from a model card.
pub fn capabilities(e: &crate::registry::ModelEntry) -> Vec<&'static str> {
    let mut out = vec!["completion"];
    // The model's own chat template is the closest thing to a declaration of
    // what it can do: a template that renders `tools` can call them, and one
    // that emits <think> or reads enable_thinking can reason.
    let template = e
        .metadata_info
        .iter()
        .find(|(k, _)| k == "tokenizer.chat_template")
        .map(|(_, v)| v.as_str())
        .unwrap_or("");
    if template.contains("tools") || template.contains("tool_calls") {
        out.push("tools");
    }
    if template.contains("enable_thinking")
        || template.contains("<think>")
        || template.contains("reasoning_effort")
    {
        out.push("thinking");
    }
    if e.has_projector {
        out.push("vision");
    }
    out
}

/// An entry in `/api/tags`, built from the registry's view of a model file.
pub fn tag_entry(e: &crate::registry::ModelEntry) -> Value {
    json!({
        "name": e.id,
        "model": e.id,
        "modified_at": iso_utc(e.modified_secs),
        "size": e.file_bytes,
        "digest": e.digest,
        "details": details(e),
    })
}

/// The `/api/show` body, in Ollama's shape.
pub fn show_body(e: &crate::registry::ModelEntry) -> Value {
    show_body_verbose(e, false)
}

/// `verbose: true` adds the large tokenizer arrays Ollama only returns when
/// asked. Reading the header again is cheap; the weights are never touched.
pub fn show_body_verbose(e: &crate::registry::ModelEntry, verbose: bool) -> Value {
    let mut info: serde_json::Map<String, Value> = e
        .metadata_info
        .iter()
        .map(|(k, v)| {
            // Ollama keeps numeric metadata numeric. The GGUF reader keeps it
            // lossless as text, so restore the obvious integer/float cases.
            let value = if let Ok(n) = v.parse::<u64>() {
                json!(n)
            } else if let Ok(n) = v.parse::<i64>() {
                json!(n)
            } else if let Ok(n) = v.parse::<f64>() {
                json!(n)
            } else {
                Value::String(v.clone())
            };
            (k.clone(), value)
        })
        .collect();
    info.insert("general.parameter_count".into(), json!(e.parameter_count));
    if verbose {
        if let Ok(g) = crate::gguf::Gguf::open(&e.path) {
            for key in [
                "tokenizer.ggml.tokens",
                "tokenizer.ggml.merges",
                "tokenizer.ggml.token_type",
            ] {
                if let Some(crate::gguf::Value::Array(items)) = g.metadata.get(key) {
                    let values: Vec<Value> = items
                        .iter()
                        .map(|v| match v {
                            crate::gguf::Value::Str(s) => json!(s),
                            crate::gguf::Value::U32(n) => json!(n),
                            crate::gguf::Value::I32(n) => json!(n),
                            crate::gguf::Value::U64(n) => json!(n),
                            _ => json!(null),
                        })
                        .collect();
                    info.insert(key.into(), json!(values));
                }
            }
        }
    }
    // A derived model keeps its Modelfile overrides in strata.json beside the
    // weights. They belong in `show` — a client rebuilding the model from the
    // printed Modelfile must get the same SYSTEM and PARAMETER lines back.
    let local = std::path::Path::new(&e.path)
        .parent()
        .map(|p| p.join("strata.json"))
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .unwrap_or_else(|| json!({}));
    let mut modelfile = format!(
        "# Modelfile generated by \"strata show\"\n\
         # To build a new Modelfile based on this, replace FROM with:\n\
         # FROM {}\n\n\
         FROM {}\n",
        e.id, e.path
    );
    let template = local
        .get("template")
        .and_then(Value::as_str)
        .unwrap_or("{{ .Prompt }}")
        .to_string();
    modelfile.push_str(&format!("TEMPLATE \"\"\"{template}\"\"\"\n"));
    if let Some(system) = local.get("system").and_then(Value::as_str) {
        modelfile.push_str(&format!("SYSTEM \"\"\"{system}\"\"\"\n"));
    }
    let mut parameters = format!("num_ctx                       {}\n", e.context);
    if let Some(params) = local.get("parameters").and_then(Value::as_object) {
        for (k, v) in params {
            parameters.push_str(&format!("{k:<30} {}\n", v.as_str().unwrap_or("")));
        }
    }
    let mut caps: Vec<String> = capabilities(e).iter().map(|s| s.to_string()).collect();
    if let Some(extra) = local.get("capabilities").and_then(Value::as_array) {
        for cap in extra.iter().filter_map(Value::as_str) {
            let owned = cap.to_string();
            if !caps.contains(&owned) {
                caps.push(owned);
            }
        }
    }
    let projector_info = e
        .has_projector
        .then(|| {
            let dir = std::path::Path::new(&e.path).parent()?;
            let path = std::fs::read_dir(dir).ok()?.flatten().find(|f| {
                f.file_name()
                    .to_string_lossy()
                    .to_ascii_lowercase()
                    .starts_with("mmproj")
            })?;
            let g = crate::gguf::Gguf::open(path.path()).ok()?;
            Some(json!({
                "name": g.name(),
                "architecture": g.arch(),
                "embedding_length": g.key_u64("embedding_length").unwrap_or(0),
                "file_bytes": g.file_size,
            }))
        })
        .flatten();
    json!({
        "modelfile": modelfile,
        "parameters": parameters,
        "template": template,
        "system": local.get("system").cloned().unwrap_or(Value::Null),
        "details": details(e),
        "model_info": info,
        "projector_info": projector_info.unwrap_or(Value::Null),
        "capabilities": caps,
        "modified_at": iso_utc(e.modified_secs),
    })
}

/// "9.2B", "730M" — Ollama's compact parameter-size spelling.
pub fn parameter_size(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else {
        count.to_string()
    }
}

fn now_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    iso_utc(secs)
}

/// RFC 3339 UTC, the shape Ollama stamps on every response.
pub fn iso_utc(secs: u64) -> String {
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

    #[test]
    fn a_fixed_timestamp_is_not_the_current_time() {
        // The epoch in a model's mtime must not be replaced by "now" when the
        // server answers /api/tags.
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso_utc(1_700_000_000).len(), 20);
    }

    #[test]
    fn generate_suffix_and_raw_change_the_prompt_shape() {
        let out = generate_to_openai(&json!({
            "model": "m", "system": "be terse", "prompt": "def f():", "suffix": "return 1"
        }))
        .unwrap();
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert!(msgs[1]["content"].as_str().unwrap().contains("return 1"));

        let raw = generate_to_openai(&json!({
            "model": "m", "system": "be terse", "prompt": "full templated prompt", "raw": true
        }))
        .unwrap();
        let msgs = raw["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn think_is_forwarded_instead_of_forced_off() {
        let on = chat_to_openai(&json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "x" }],
            "think": true
        }))
        .unwrap();
        assert_eq!(on["chat_template_kwargs"]["enable_thinking"], true);

        let off = chat_to_openai(&json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "x" }],
            "think": "low"
        }))
        .unwrap();
        assert_eq!(off["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(off["reasoning_effort"], "low");

        let high = chat_to_openai(&json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "x" }],
            "think": "high"
        }))
        .unwrap();
        assert_eq!(high["reasoning_effort"], "xhigh");

        let none = generate_to_openai(&json!({ "model": "m", "think": false })).unwrap();
        assert_eq!(none["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn a_reasoning_trace_and_tool_name_survive_translation() {
        let req = json!({
            "model": "m",
            "messages": [
                { "role": "assistant", "content": "", "thinking": "step 1", "tool_calls": [] },
                { "role": "tool", "content": "11C", "tool_name": "get_weather" }
            ]
        });
        let out = chat_to_openai(&req).unwrap();
        assert_eq!(out["messages"][0]["reasoning_content"], "step 1");
        assert_eq!(out["messages"][1]["name"], "get_weather");
    }

    #[test]
    fn ollama_json_format_becomes_a_response_format() {
        let out = chat_to_openai(&json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "x" }],
            "format": "json"
        }))
        .unwrap();
        assert_eq!(out["response_format"]["type"], "json_object");
    }

    #[test]
    fn parameter_sizes_use_ollama_spelling() {
        assert_eq!(parameter_size(9_200_000_000), "9.2B");
        assert_eq!(parameter_size(730_000_000), "730.0M");
        assert_eq!(parameter_size(17), "17");
    }

    #[test]
    fn details_come_from_the_file_not_from_placeholders() {
        let e = crate::registry::ModelEntry {
            id: "tiny".into(),
            path: "rocm/models/tiny/model.gguf".into(),
            arch: "llama".into(),
            name: "Tiny".into(),
            experts: 0,
            experts_used: 0,
            context: 4096,
            file_bytes: 100,
            is_moe: false,
            parameter_count: 1_100_000_000,
            quantization_level: "Q4_K_M".into(),
            embedding: 2048,
            vocab: 32_000,
            families: vec!["llama".into()],
            modified_secs: 0,
            digest: "strata-abc".into(),
            has_projector: false,
            metadata_info: vec![("general.architecture".into(), "llama".into())],
        };
        let d = details(&e);
        assert_eq!(d["parameter_size"], "1.1B");
        assert_eq!(d["quantization_level"], "Q4_K_M");
        assert_eq!(d["families"][0], "llama");
        assert_eq!(d["embedding_length"], 2048);
        let s = show_body(&e);
        assert_eq!(s["model_info"]["general.architecture"], "llama");
        assert_eq!(s["capabilities"][0], "completion");
        assert_eq!(tag_entry(&e)["digest"], "strata-abc");
        assert_eq!(tag_entry(&e)["modified_at"], "1970-01-01T00:00:00Z");
    }
}
