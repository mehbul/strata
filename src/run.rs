//! `strata run` — the Ollama-style interactive shell and one-shot prompt.
//!
//! It talks to the Strata HTTP API rather than importing the engine in-process,
//! so the same code works against a server started here, one already running,
//! or a remote host. A started server is left up, matching Ollama's daemon
//! behaviour and the `pi.ps1` workflow.

use anyhow::{bail, Context, Result};
use futures::StreamExt;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

fn base64(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

pub struct RunArgs {
    pub host: String,
    pub model: String,
    pub prompt: Option<Vec<String>>,
    pub format: Option<String>,
    pub keepalive: Option<String>,
    pub image: Option<String>,
    pub think: Option<bool>,
    pub verbose: bool,
}

struct Session {
    client: reqwest::Client,
    host: String,
    key: Option<String>,
    model: String,
    history: Vec<Value>,
    think: Option<bool>,
    format: Option<String>,
    keepalive: Option<String>,
    image: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    system: Option<String>,
    verbose: bool,
}

impl Session {
    fn post(&self, path: &str, body: Value) -> reqwest::RequestBuilder {
        let mut req = self
            .client
            .post(format!("{}{path}", self.host.trim_end_matches('/')))
            .json(&body);
        if let Some(key) = &self.key {
            req = req.bearer_auth(key);
        }
        req
    }

    async fn complete(&mut self, text: &str) -> Result<()> {
        self.history
            .push(json!({ "role": "user", "content": text }));
        if let Some(path) = &self.image {
            match std::fs::read(path) {
                Ok(bytes) => {
                    if let Some(last) = self.history.last_mut() {
                        let prompt = last["content"].as_str().unwrap_or("").to_string();
                        let mime = if path.to_ascii_lowercase().ends_with(".png") {
                            "image/png"
                        } else {
                            "image/jpeg"
                        };
                        last["content"] = json!([
                            { "type": "text", "text": prompt },
                            { "type": "image_url", "image_url": {
                                "url": format!("data:{mime};base64,{}", base64(&bytes))
                            }}
                        ]);
                    }
                }
                Err(e) => println!("could not read image {path}: {e}"),
            }
        }
        let mut body = json!({
            "model": self.model,
            "messages": self.history,
            "stream": true,
        });
        if let Some(k) = self.think {
            body["chat_template_kwargs"] = json!({ "enable_thinking": k });
        }
        if let Some(f) = &self.format {
            body["response_format"] = crate::dialects::response_format(&Value::String(f.clone()));
        }
        if let Some(k) = &self.keepalive {
            body["keep_alive"] = json!(k);
        }
        if let Some(t) = self.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(n) = self.max_tokens {
            body["max_tokens"] = json!(n);
        }
        if let Some(p) = self.top_p {
            body["top_p"] = json!(p);
        }
        let start = std::time::Instant::now();
        let resp = self
            .post("/v1/chat/completions", body)
            .send()
            .await
            .context("cannot reach Strata; is `strata serve` running?")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("server returned {status}: {text}");
        }

        let mut inner = resp.bytes_stream();
        let mut buf = String::new();
        let mut answer = String::new();
        let mut reasoning = String::new();
        let mut first_token: Option<std::time::Duration> = None;
        let mut usage = json!({});
        while let Some(chunk) = inner.next().await {
            let chunk = chunk?;
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buf.find("\n\n") {
                let frame: String = buf.drain(..pos + 2).collect();
                for line in frame.lines() {
                    let Some(payload) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let payload = payload.trim_start();
                    if payload == "[DONE]" {
                        continue;
                    }
                    let Ok(v) = serde_json::from_str::<Value>(payload) else {
                        continue;
                    };
                    if let Some(u) = v.get("usage") {
                        if u.get("completion_tokens").is_some() {
                            usage = u.clone();
                        }
                    }
                    let Some(delta) = v
                        .get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|c| c.first())
                        .and_then(|c| c.get("delta"))
                    else {
                        continue;
                    };
                    if let Some(s) = delta.get("content").and_then(Value::as_str) {
                        if !s.is_empty() {
                            if first_token.is_none() {
                                first_token = Some(start.elapsed());
                            }
                            answer.push_str(s);
                            print!("{s}");
                            io::stdout().flush().ok();
                        }
                    }
                    if let Some(s) = delta
                        .get("reasoning_content")
                        .and_then(Value::as_str)
                        .or_else(|| delta.get("reasoning").and_then(Value::as_str))
                    {
                        reasoning.push_str(s);
                    }
                }
            }
        }
        println!();
        let mut msg = json!({ "role": "assistant", "content": answer });
        if !reasoning.is_empty() {
            msg["reasoning_content"] = json!(reasoning);
        }
        self.history.push(msg);

        if self.verbose {
            let total = start.elapsed();
            let first = first_token.unwrap_or(total);
            let eval = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let prompt = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let rate = if total.as_secs_f64() > 0.0 {
                eval as f64 / total.as_secs_f64()
            } else {
                0.0
            };
            println!(
                "\ntotal duration: {:.2}s  first token: {:.2}s\n\
                 prompt eval count: {prompt}  eval count: {eval}  eval rate: {rate:.1} tok/s",
                total.as_secs_f64(),
                first.as_secs_f64()
            );
        }
        Ok(())
    }
}

/// Wait until the server answers /health, or 120 seconds pass.
async fn wait_ready(client: &reqwest::Client, host: &str) -> Result<()> {
    let url = format!("{}/health", host.trim_end_matches('/'));
    for _ in 0..120 {
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                return Ok(());
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
    bail!("server at {host} did not become ready in 120s")
}

async fn ensure_server(host: &str, model: &str) -> Result<()> {
    let client = reqwest::Client::new();
    if client
        .get(format!("{}/health", host.trim_end_matches('/')))
        .send()
        .await
        .is_ok()
    {
        return Ok(());
    }
    let root = crate::registry::default_root();
    let chosen = if model.is_empty() {
        crate::registry::scan(&root)
            .first()
            .map(|e| e.id.clone())
            .context("no models installed; run `strata pull --model <name>` first")?
    } else {
        model.to_string()
    };
    println!("Starting the Strata engine for {chosen}…");
    let exe = std::env::current_exe()?;
    std::process::Command::new(exe)
        .arg("serve")
        .arg("--model")
        .arg(&chosen)
        .spawn()
        .context("could not start `strata serve`")?;
    wait_ready(&client, host).await?;
    Ok(())
}

fn apply_system(session: &mut Session) {
    let overrides = crate::registry::overrides(&crate::registry::default_root(), &session.model);
    if let Some(system) = overrides.get("system").and_then(Value::as_str) {
        if !session
            .history
            .iter()
            .any(|m| m.get("role").and_then(Value::as_str) == Some("system"))
        {
            session
                .history
                .insert(0, json!({ "role": "system", "content": system }));
        }
    }
    if let Some(messages) = overrides.get("messages").and_then(Value::as_array) {
        for m in messages {
            if !session.history.contains(m) {
                session.history.push(m.clone());
            }
        }
    }
}

async fn slash(session: &mut Session, line: &str) -> Result<bool> {
    match line.split_whitespace().next().unwrap_or("") {
        "/bye" | "/exit" | "/quit" => return Ok(true),
        "/clear" => session.history.clear(),
        "/help" => println!(
            "/bye  /clear  /help  /model <name>  /show  /pull <name>  \
             /load-model <name>  /save-model <name>  /set think|verbose <on|off>  \
             /set system <text>  /save <file>  /load <file>"
        ),
        "/model" | "/load-model" => {
            let name = line.split_whitespace().nth(1).unwrap_or("");
            let body = json!({ "model": name });
            let resp = session.post("/api/models/load", body).send().await?;
            if resp.status().is_success() {
                session.model = name.to_string();
                session.history.clear();
                println!("switched to {name}");
            } else {
                println!("could not load {name}: {}", resp.text().await?);
            }
        }
        "/show" => {
            let resp = session
                .post("/api/show", json!({ "model": session.model.clone() }))
                .send()
                .await?;
            let body: Value = resp.json().await.unwrap_or(json!({}));
            println!(
                "{}",
                serde_json::to_string_pretty(&body).unwrap_or_default()
            );
        }
        "/pull" => {
            let name = line.split_whitespace().nth(1).unwrap_or("");
            if name.is_empty() {
                println!("usage: /pull <model>");
            } else {
                crate::hub::Hub::new().pull(name).await?;
            }
        }
        "/save-model" => {
            let dest = line.split_whitespace().nth(1).unwrap_or("");
            crate::registry::copy(&crate::registry::default_root(), &session.model, dest)?;
            println!("saved as {dest}");
        }
        "/save" => {
            let path = line
                .split_whitespace()
                .nth(1)
                .unwrap_or("conversation.json");
            let body = json!({
                "model": session.model,
                "history": session.history,
                "think": session.think,
                "format": session.format,
            });
            std::fs::write(path, serde_json::to_string_pretty(&body)?)?;
            println!("saved conversation to {path}");
        }
        "/load" => {
            let path = line
                .split_whitespace()
                .nth(1)
                .unwrap_or("conversation.json");
            let body: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            if let Some(model) = body
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string)
            {
                session.model = model;
            }
            session.history = body
                .get("history")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            session.think = body.get("think").and_then(Value::as_bool);
            session.format = body
                .get("format")
                .and_then(Value::as_str)
                .map(str::to_string);
            println!("loaded {} messages from {path}", session.history.len());
        }
        "/set" => {
            let mut parts = line.split_whitespace().skip(1);
            match (parts.next(), parts.next()) {
                (Some("think"), Some(v)) => session.think = Some(matches!(v, "on" | "true" | "1")),
                (Some("verbose"), Some(v)) => session.verbose = matches!(v, "on" | "true" | "1"),
                (Some("format"), Some(v)) => session.format = Some(v.to_string()),
                (Some("temperature"), Some(v)) => session.temperature = v.parse().ok(),
                (Some("max-tokens"), Some(v)) => session.max_tokens = v.parse().ok(),
                (Some("top-p"), Some(v)) => session.top_p = v.parse().ok(),
                (Some("system"), _) => {
                    // Keep the whole prompt, not just the first word.
                    let text = line
                        .strip_prefix("/set")
                        .and_then(|rest| rest.trim_start().strip_prefix("system"))
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    session
                        .history
                        .retain(|m| m.get("role").and_then(Value::as_str) != Some("system"));
                    if !text.is_empty() {
                        session
                            .history
                            .insert(0, json!({ "role": "system", "content": text }));
                        session.system = Some(text);
                    } else {
                        session.system = None;
                    }
                }
                _ => println!(
                    "usage: /set think|verbose|format|system|temperature|max-tokens|top-p <value>"
                ),
            }
        }
        _ => println!("unknown command: {line}"),
    }
    Ok(false)
}

pub async fn run(args: RunArgs) -> Result<()> {
    ensure_server(&args.host, &args.model).await?;
    let key = std::env::var("STRATA_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty());
    let mut session = Session {
        client: reqwest::Client::new(),
        host: args.host.clone(),
        key,
        model: args.model.clone(),
        history: Vec::new(),
        think: args.think,
        format: args.format,
        keepalive: args.keepalive,
        image: args.image,
        temperature: None,
        max_tokens: None,
        top_p: None,
        system: None,
        verbose: args.verbose,
    };
    apply_system(&mut session);

    if let Some(parts) = args.prompt {
        session.complete(&parts.join(" ")).await?;
        return Ok(());
    }

    println!("Strata interactive shell. Type /help for commands, /bye to exit.");
    let stdin = io::stdin();
    let mut locked = stdin.lock();
    let mut buffer = String::new();
    loop {
        print!(">>> ");
        io::stdout().flush().ok();
        buffer.clear();
        let n = locked.read_line(&mut buffer)?;
        if n == 0 {
            break;
        }
        let line = buffer.trim_end().to_string();
        if line.trim() == "\"\"\"" {
            let mut multi = String::new();
            loop {
                buffer.clear();
                if locked.read_line(&mut buffer)? == 0 {
                    break;
                }
                if buffer.trim() == "\"\"\"" {
                    break;
                }
                multi.push_str(&buffer);
            }
            session.complete(&multi).await?;
            continue;
        }
        if line.trim_start().starts_with('/') {
            if slash(&mut session, line.trim()).await? {
                break;
            }
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        session.complete(&line).await?;
    }
    Ok(())
}
