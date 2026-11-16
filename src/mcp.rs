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
            Ok(v) => RpcResponse { jsonrpc: "2.0", id, result: Some(v), error: None },
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
            {"name": "server_status", "description": "Check a running Strata server", "inputSchema": {"type":"object","properties":{"url":{"type":"string"}}}},
        ]})),
        "tools/call" => {
            let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            match name {
                "list_models" => {
                    let entries = crate::registry::scan(&crate::registry::default_root());
                    Ok(json!({"content": [{"type":"text","text": serde_json::to_string_pretty(&entries).unwrap_or_default()}]}))
                }
                "server_status" => {
                    let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("http://127.0.0.1:8080");
                    let client = reqwest::Client::new();
                    let r = client.get(format!("{}/health", url)).send().await;
                    match r {
                        Ok(resp) => Ok(json!({"content": [{"type":"text","text": format!("HTTP {}", resp.status())}]})),
                        Err(e) => Ok(json!({"content": [{"type":"text","text": format!("unreachable: {e}")}], "isError": true})),
                    }
                }
                _ => anyhow::bail!("unknown tool: {name}"),
            }
        }
        _ => anyhow::bail!("unknown method: {method}"),
    }
}
