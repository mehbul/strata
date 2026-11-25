//! API dialects that are not quite chat completions.
//!
//! Ollama 0.40.16 answers `/v1/completions` and `/v1/responses` as well as
//! `/v1/chat/completions`. Clients written against either of the first two do
//! not fall back, so Strata translates them into the one shape the compute
//! process understands and translates the answer back. No new model behaviour
//! is invented here: everything is a mapping over the chat endpoint.

use anyhow::{bail, Result};
use serde_json::{json, Map, Value};

fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|i| i.get("text").and_then(Value::as_str).or_else(|| i.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// OpenAI's legacy `/v1/completions` → chat completions.
pub fn completions_to_chat(req: &Value) -> Result<Value> {
    let prompt = match req.get("prompt") {
        Some(Value::Array(items)) => {
            // A token-id array is a prompt in OpenAI's own definition. The
            // compute process takes text, so refuse rather than silently turn
            // token ids into the string "[1,2,3]" and answer a different ask.
            if items.iter().all(Value::is_u64) {
                bail!("token-id prompts are not supported; pass a text prompt");
            }
            items.iter().map(text_of).collect::<Vec<_>>().join("")
        }
        Some(v) => text_of(v),
        None => String::new(),
    };
    let mut messages = Vec::new();
    if let Some(system) = req.get("system").and_then(Value::as_str) {
        if !system.is_empty() {
            messages.push(json!({ "role": "system", "content": system }));
        }
    }
    messages.push(json!({ "role": "user", "content": prompt }));

    let mut out = Map::new();
    out.insert(
        "model".into(),
        req.get("model").cloned().unwrap_or(json!("")),
    );
    out.insert("messages".into(), json!(messages));
    out.insert(
        "stream".into(),
        req.get("stream").cloned().unwrap_or(json!(false)),
    );
    for (from, to) in [
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("n", "n"),
        ("seed", "seed"),
        ("max_tokens", "max_tokens"),
    ] {
        if let Some(v) = req.get(from) {
            out.insert(to.into(), v.clone());
        }
    }
    if let Some(stop) = req.get("stop") {
        out.insert("stop".into(), stop.clone());
    }
    if let Some(format) = req.get("format") {
        out.insert("response_format".into(), response_format(format));
    }
    Ok(Value::Object(out))
}

/// Ollama's `format` (JSON mode or a schema) → OpenAI's `response_format`.
pub fn response_format(format: &Value) -> Value {
    match format {
        Value::String(s) if s == "json" => json!({ "type": "json_object" }),
        Value::Object(schema) => json!({
            "type": "json_schema",
            "json_schema": { "name": "response", "schema": schema }
        }),
        _ => format.clone(),
    }
}

/// OpenAI Responses API `/v1/responses` → chat completions.
pub fn responses_to_chat(req: &Value) -> Result<Value> {
    let mut messages: Vec<Value> = Vec::new();
    if let Some(instructions) = req.get("instructions").and_then(Value::as_str) {
        if !instructions.is_empty() {
            messages.push(json!({ "role": "system", "content": instructions }));
        }
    }
    match req.get("input") {
        Some(Value::String(s)) => messages.push(json!({ "role": "user", "content": s })),
        Some(Value::Array(items)) => {
            for item in items {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let content = item
                    .get("content")
                    .map(text_of)
                    .unwrap_or_else(|| text_of(item));
                messages.push(json!({ "role": role, "content": content }));
            }
        }
        _ => bail!("input must be a string or an array"),
    }
    let mut out = Map::new();
    out.insert(
        "model".into(),
        req.get("model").cloned().unwrap_or(json!("")),
    );
    out.insert("messages".into(), json!(messages));
    out.insert(
        "stream".into(),
        req.get("stream").cloned().unwrap_or(json!(false)),
    );
    if let Some(n) = req.get("max_output_tokens") {
        out.insert("max_tokens".into(), n.clone());
    }
    if let Some(t) = req.get("temperature") {
        out.insert("temperature".into(), t.clone());
    }
    if let Some(tools) = req.get("tools") {
        out.insert("tools".into(), tools.clone());
    }
    if let Some(format) = req.get("text").and_then(|t| t.get("format")) {
        out.insert("response_format".into(), response_format(format));
    }
    Ok(Value::Object(out))
}

fn choice(body: &Value) -> (String, String, Value) {
    let first = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first());
    let text = first
        .and_then(|c| c.get("message").or_else(|| c.get("delta")))
        .and_then(|m| {
            ["content", "reasoning_content", "reasoning"]
                .iter()
                .filter_map(|key| m.get(*key))
                .find(|v| v.as_str().is_some_and(|s| !s.is_empty()))
        })
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let finish = first
        .and_then(|c| c.get("finish_reason"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let usage = body.get("usage").cloned().unwrap_or_else(|| json!({}));
    (text, finish, usage)
}

/// A chat completion answer → `/v1/completions` answer.
pub fn chat_to_completions(body: &Value, model: &str, id: &str, created: u64) -> Value {
    let (text, finish, usage) = choice(body);
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "text": text,
            "logprobs": null,
            "finish_reason": if finish.is_empty() { Value::Null } else { Value::String(finish) },
        }],
        "usage": usage,
    })
}

/// A chat completion answer → a Responses API object.
pub fn chat_to_response(body: &Value, model: &str, id: &str, created: u64) -> Value {
    let (text, _finish, usage) = choice(body);
    let input = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    json!({
        "id": id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model,
        "output": [{
            "type": "message",
            "id": format!("msg_{id}"),
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": text, "annotations": [] }],
        }],
        "usage": {
            "input_tokens": input,
            "output_tokens": output,
            "total_tokens": input + output,
        },
    })
}

/// Validate a structured answer. `json` must parse; a schema must at least
/// contain the keys it declares as `required`. This is deliberately shallow —
/// it catches a model that ignored the format, not every schema violation.
pub fn valid_structured(text: &str, format: &Value) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(text) else {
        return false;
    };
    let schema = match format {
        Value::String(s) if s == "json" => return value.is_object() || value.is_array(),
        Value::Object(o) => o,
        _ => return true,
    };
    match jsonschema::validator_for(&Value::Object(schema.clone())) {
        Ok(validator) => validator.is_valid(&value),
        // A schema we cannot compile is not evidence that the answer is bad.
        Err(_) => {
            if let Some(Value::Array(required)) = schema.get("required") {
                required
                    .iter()
                    .all(|k| k.as_str().is_some_and(|key| value.get(key).is_some()))
            } else {
                true
            }
        }
    }
}

/// One SSE event for a streaming `/v1/responses` answer.
pub fn response_event(name: &str, data: &Value) -> axum::response::sse::Event {
    axum::response::sse::Event::default()
        .event(name)
        .data(data.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_output_validation_checks_json_and_required_keys() {
        assert!(valid_structured("{\"a\":1}", &json!("json")));
        assert!(!valid_structured("not json", &json!("json")));
        let schema = json!({"type":"object","required":["a","b"]});
        assert!(!valid_structured("{\"a\":1}", &schema));
        assert!(valid_structured("{\"a\":1,\"b\":2}", &schema));
    }

    #[test]
    fn a_legacy_prompt_becomes_a_user_turn() {
        let out = completions_to_chat(&json!({ "model": "m", "prompt": "hello" })).unwrap();
        assert_eq!(out["messages"][0]["content"], "hello");
        assert_eq!(out["stream"], false);
    }

    #[test]
    fn token_id_prompts_are_refused_not_mangled() {
        assert!(completions_to_chat(&json!({ "prompt": [1, 2, 3] })).is_err());
    }

    #[test]
    fn ollama_json_format_becomes_openai_response_format() {
        let out = completions_to_chat(&json!({ "prompt": "x", "format": "json" })).unwrap();
        assert_eq!(out["response_format"]["type"], "json_object");
    }

    #[test]
    fn responses_input_items_become_messages() {
        let out = responses_to_chat(&json!({
            "model": "m",
            "instructions": "be brief",
            "input": [
                { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hi" }] }
            ]
        }))
        .unwrap();
        assert_eq!(out["messages"][0]["role"], "system");
        assert_eq!(out["messages"][1]["content"], "hi");
    }

    #[test]
    fn reasoning_only_answers_are_not_reported_as_empty() {
        // Observed from the local Ollama 0.40.16 /v1/chat/completions: a
        // thinking model can put all its text in `reasoning` and leave
        // `content` empty.
        let chat = json!({
            "choices": [{ "message": { "content": "", "reasoning": "hello" } }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1 }
        });
        assert_eq!(
            chat_to_completions(&chat, "m", "cmpl-1", 0)["choices"][0]["text"],
            "hello"
        );
    }

    #[test]
    fn answers_come_back_in_the_dialect_the_client_spoke() {
        let chat = json!({
            "choices": [{ "message": { "content": "42" }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 3, "completion_tokens": 1 }
        });
        let c = chat_to_completions(&chat, "m", "cmpl-1", 10);
        assert_eq!(c["object"], "text_completion");
        assert_eq!(c["choices"][0]["text"], "42");
        let r = chat_to_response(&chat, "m", "resp-1", 10);
        assert_eq!(r["object"], "response");
        assert_eq!(r["output"][0]["content"][0]["text"], "42");
        assert_eq!(r["usage"]["total_tokens"], 4);
    }
}
