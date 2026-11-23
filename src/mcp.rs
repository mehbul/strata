//! Minimal MCP (Model Context Protocol) server over stdio.
//!
//! Exposes a small tool surface so an AI agent can inspect and drive a
//! Strata installation: list installed models, check server status, and run
//! a benchmark against a running instance.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, Write};

#[derive(Deserialize)]
struct RpcRequest {
    #[serde(default)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

pub async fn run() -> Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let req: RpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = RpcResponse {
                    jsonrpc: "2.0",
                    id: Value::Null,
                    result: None,
                    error: Some(json!({"code": -32700, "message": format!("parse error: {e}")})),
                };
                writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap()).ok();
                stdout.flush().ok();
                continue;
            }
        };
        let id = req.id.clone().unwrap_or(Value::Null);
        let result = handle(&req.method, &req.params).await;
        let resp = match result {
            Ok(v) => RpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(v),
                error: None,
            },
            Err(e) => RpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code": -32000, "message": e.to_string()})),
            },
        };
        writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap()).ok();
        stdout.flush().ok();
    }
    Ok(())
}

async fn handle(method: &str, params: &Value) -> Result<Value> {
    match method {
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "strata", "version": "0.1.0"}
        })),
        "tools/list" => Ok(json!({"tools": [
            {"name": "list_models", "description": "List installed models", "inputSchema": {"type":"object","properties":{}}},
            {"name": "show_model", "description": "Show one model's GGUF facts in Ollama's /api/show shape", "inputSchema": {"type":"object","properties":{"model":{"type":"string"}},"required":["model"]}},
            {"name": "copy_model", "description": "Copy a model under a second name", "inputSchema": {"type":"object","properties":{"source":{"type":"string"},"destination":{"type":"string"}},"required":["source","destination"]}},
            {"name": "remove_model", "description": "Remove a model from the local store", "inputSchema": {"type":"object","properties":{"model":{"type":"string"}},"required":["model"]}},
            {"name": "pull_model", "description": "Pull a model from Ollama or Hugging Face", "inputSchema": {"type":"object","properties":{"model":{"type":"string"}},"required":["model"]}},
            {"name": "create_model", "description": "Create a model from Modelfile text", "inputSchema": {"type":"object","properties":{"name":{"type":"string"},"modelfile":{"type":"string"}},"required":["name","modelfile"]}},
            {"name": "run_prompt", "description": "Ask a model one question and return its answer", "inputSchema": {"type":"object","properties":{"model":{"type":"string"},"prompt":{"type":"string"},"host":{"type":"string"}},"required":["model","prompt"]}},
            {"name": "running_models", "description": "List models currently loaded by a server", "inputSchema": {"type":"object","properties":{"host":{"type":"string"}}}},
            {"name": "stop_model", "description": "Unload a model from a running server", "inputSchema": {"type":"object","properties":{"model":{"type":"string"},"host":{"type":"string"}},"required":["model"]}},
            {"name": "push_model", "description": "Push a model to a registry", "inputSchema": {"type":"object","properties":{"model":{"type":"string"},"registry":{"type":"string"},"token":{"type":"string"}},"required":["model"]}},
            {"name": "embed", "description": "Generate embeddings for a list of strings", "inputSchema": {"type":"object","properties":{"model":{"type":"string"},"input":{"type":"array","items":{"type":"string"}},"host":{"type":"string"}},"required":["model","input"]}},
            {"name": "version", "description": "Report the running server's version", "inputSchema": {"type":"object","properties":{"host":{"type":"string"}}}},
            {"name": "model_details", "description": "Show one model's declared facts from a running server", "inputSchema": {"type":"object","properties":{"model":{"type":"string"},"host":{"type":"string"}},"required":["model"]}},
            {"name": "server_status", "description": "Check a running Strata server", "inputSchema": {"type":"object","properties":{"url":{"type":"string"}}}},
        ]})),
        "tools/call" => {
            let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            match name {
                "list_models" => {
                    let entries = crate::registry::scan(&crate::registry::default_root());
                    Ok(
                        json!({"content": [{"type":"text","text": serde_json::to_string_pretty(&entries).unwrap_or_default()}]}),
                    )
                }
                "show_model" => {
                    let model = args.get("model").and_then(|v| v.as_str()).unwrap_or("");
                    let entries = crate::registry::scan(&crate::registry::default_root());
                    match entries.iter().find(|e| e.id == model) {
                        Some(e) => Ok(json!({"content": [{"type":"text","text":
                            serde_json::to_string_pretty(&crate::ollama::show_body(e)).unwrap_or_default()}]})),
                        None => Ok(json!({"content": [{"type":"text","text": format!("model '{model}' not found")}], "isError": true})),
                    }
                }
                "copy_model" => {
                    let source = args.get("source").and_then(|v| v.as_str()).unwrap_or("");
                    let destination = args.get("destination").and_then(|v| v.as_str()).unwrap_or("");
                    match crate::registry::copy(&crate::registry::default_root(), source, destination) {
                        Ok(()) => Ok(json!({"content": [{"type":"text","text": format!("copied {source} -> {destination}")}]})),
                        Err(e) => Ok(json!({"content": [{"type":"text","text": e.to_string()}], "isError": true})),
                    }
                }
                "remove_model" => {
                    let model = args.get("model").and_then(|v| v.as_str()).unwrap_or("");
                    match crate::registry::remove(&crate::registry::default_root(), model) {
                        Ok(()) => Ok(json!({"content": [{"type":"text","text": format!("removed {model}")}]})),
                        Err(e) => Ok(json!({"content": [{"type":"text","text": e.to_string()}], "isError": true})),
                    }
                }
                "pull_model" => {
                    let model = args.get("model").and_then(|v| v.as_str()).unwrap_or("");
                    match crate::hub::Hub::new().pull(model).await {
                        Ok(p) => Ok(json!({"content": [{"type":"text","text": format!("pulled {model} to {}", p.display())}]})),
                        Err(e) => Ok(json!({"content": [{"type":"text","text": e.to_string()}], "isError": true})),
                    }
                }
                "create_model" => {
                    let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let text = args.get("modelfile").and_then(|v| v.as_str()).unwrap_or("");
                    match crate::modelfile::Modelfile::parse(text)
                        .and_then(|m| crate::registry::create_from_modelfile(&crate::registry::default_root(), name, &m))
                    {
                        Ok(()) => Ok(json!({"content": [{"type":"text","text": format!("created {name}")}]})),
                        Err(e) => Ok(json!({"content": [{"type":"text","text": e.to_string()}], "isError": true})),
                    }
                }
                "run_prompt" => {
                    let model = args.get("model").and_then(|v| v.as_str()).unwrap_or("");
                    let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
                    let host = args.get("host").and_then(|v| v.as_str()).unwrap_or("http://127.0.0.1:8080");
                    let client = reqwest::Client::new();
                    let resp = client
                        .post(format!("{}/v1/chat/completions", host.trim_end_matches('/')))
                        .json(&json!({"model": model, "messages": [{"role":"user","content":prompt}], "stream": false}))
                        .send().await;
                    match resp {
                        Ok(r) if r.status().is_success() => {
                            let body: Value = r.json().await.unwrap_or(json!({}));
                            let text = body.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first())
                                .and_then(|c| c.get("message"))
                                .and_then(|m| m.get("content").or_else(|| m.get("reasoning_content")).or_else(|| m.get("reasoning")))
                                .and_then(|v| v.as_str()).unwrap_or("");
                            Ok(json!({"content": [{"type":"text","text": text}]}))
                        }
                        Ok(r) => Ok(json!({"content": [{"type":"text","text": format!("HTTP {}", r.status())}], "isError": true})),
                        Err(e) => Ok(json!({"content": [{"type":"text","text": e.to_string()}], "isError": true})),
                    }
                }
                "running_models" => {
                    let host = args.get("host").and_then(|v| v.as_str()).unwrap_or("http://127.0.0.1:8080");
                    let client = reqwest::Client::new();
                    let body: Value = client.get(format!("{}/api/ps", host.trim_end_matches('/'))).send().await?
                        .json().await.unwrap_or(json!({}));
                    Ok(json!({"content": [{"type":"text","text": serde_json::to_string_pretty(&body).unwrap_or_default()}]}))
                }
                "stop_model" => {
                    let model = args.get("model").and_then(|v| v.as_str()).unwrap_or("");
                    let host = args.get("host").and_then(|v| v.as_str()).unwrap_or("http://127.0.0.1:8080");
                    let client = reqwest::Client::new();
                    let resp = client.post(format!("{}/api/models/unload", host.trim_end_matches('/')))
                        .json(&json!({"model": model})).send().await?;
                    Ok(json!({"content": [{"type":"text","text": format!("unload requested for {model}: {}", resp.status())}]}))
                }
                "push_model" => {
                    let model = args.get("model").and_then(|v| v.as_str()).unwrap_or("");
                    match crate::push::push(model, args.get("registry").and_then(|v| v.as_str()), args.get("token").and_then(|v| v.as_str()), false).await {
                        Ok(()) => Ok(json!({"content": [{"type":"text","text": format!("pushed {model}")}]})),
                        Err(e) => Ok(json!({"content": [{"type":"text","text": e.to_string()}], "isError": true})),
                    }
                }
                "embed" => {
                    let model = args.get("model").and_then(|v| v.as_str()).unwrap_or("");
                    let input = args.get("input").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                    let host = args.get("host").and_then(|v| v.as_str()).unwrap_or("http://127.0.0.1:8080");
                    let client = reqwest::Client::new();
                    match client
                        .post(format!("{}/api/embed", host.trim_end_matches('/')))
                        .json(&json!({ "model": model, "input": input }))
                        .send().await
                    {
                        Ok(r) if r.status().is_success() => {
                            let body: Value = r.json().await.unwrap_or(json!({}));
                            let count = body.get("embeddings").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
                            Ok(json!({"content": [{"type":"text","text": format!("{count} embedding vector(s)")}]}))
                        }
                        Ok(r) => Ok(json!({"content": [{"type":"text","text": format!("HTTP {}", r.status())}], "isError": true})),
                        Err(e) => Ok(json!({"content": [{"type":"text","text": e.to_string()}], "isError": true})),
                    }
                }
                "version" => {
                    let host = args.get("host").and_then(|v| v.as_str()).unwrap_or("http://127.0.0.1:8080");
                    let client = reqwest::Client::new();
                    match client.get(format!("{}/api/version", host.trim_end_matches('/'))).send().await {
                        Ok(r) if r.status().is_success() => {
                            let body: Value = r.json().await.unwrap_or(json!({}));
                            Ok(json!({"content": [{"type":"text","text": body.to_string()}]}))
                        }
                        Ok(r) => Ok(json!({"content": [{"type":"text","text": format!("HTTP {}", r.status())}], "isError": true})),
                        Err(e) => Ok(json!({"content": [{"type":"text","text": e.to_string()}], "isError": true})),
                    }
                }
                "model_details" => {
                    let model = args.get("model").and_then(|v| v.as_str()).unwrap_or("");
                    let host = args.get("host").and_then(|v| v.as_str()).unwrap_or("http://127.0.0.1:8080");
                    let client = reqwest::Client::new();
                    match client
                        .post(format!("{}/api/show", host.trim_end_matches('/')))
                        .json(&json!({ "model": model }))
                        .send().await
                    {
                        Ok(r) if r.status().is_success() => {
                            let body: Value = r.json().await.unwrap_or(json!({}));
                            Ok(json!({"content": [{"type":"text","text": serde_json::to_string_pretty(&body).unwrap_or_default()}]}))
                        }
                        Ok(r) => Ok(json!({"content": [{"type":"text","text": format!("HTTP {}", r.status())}], "isError": true})),
                        Err(e) => Ok(json!({"content": [{"type":"text","text": e.to_string()}], "isError": true})),
                    }
                }
                "server_status" => {
                    let url = args
                        .get("url")
                        .and_then(|v| v.as_str())
                        .unwrap_or("http://127.0.0.1:8080");
                    let client = reqwest::Client::new();
                    let r = client.get(format!("{}/health", url)).send().await;
                    match r {
                        Ok(resp) => Ok(
                            json!({"content": [{"type":"text","text": format!("HTTP {}", resp.status())}]}),
                        ),
                        Err(e) => Ok(
                            json!({"content": [{"type":"text","text": format!("unreachable: {e}")}], "isError": true}),
                        ),
                    }
                }
                _ => anyhow::bail!("unknown tool: {name}"),
            }
        }
        _ => anyhow::bail!("unknown method: {method}"),
    }
}
