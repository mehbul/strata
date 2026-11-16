//! Installed-model registry: scan rocm/models for GGUF files and read their
//! headers so the server can list and switch between them.

use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub path: String,
    pub arch: String,
    pub name: String,
    pub experts: usize,
    pub experts_used: usize,
    pub context: usize,
    pub file_bytes: u64,
    pub is_moe: bool,
}

/// Scan `root` (typically `rocm/models`) for `<name>/model.gguf` and read facts.
pub fn scan(root: &Path) -> Vec<ModelEntry> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        return out;
    };
    for e in rd.flatten() {
        let dir = e.path();
        if !dir.is_dir() {
            continue;
        }
        let gguf = dir.join("model.gguf");
        if !gguf.exists() {
            continue;
        }
        let id = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if id.is_empty() {
            continue;
        }
        let facts = crate::gguf::Gguf::open(&gguf)
            .ok()
            .and_then(|g| crate::gguf::ModelFacts::read(&g).ok());
        let (arch, name, experts, experts_used, context, file_bytes, is_moe) = match facts {
            Some(f) => (
                f.arch.clone(),
                f.name.clone(),
                f.experts,
                f.experts_used,
                f.context,
                f.file_bytes,
                f.is_moe(),
            ),
            None => ("unknown".into(), id.clone(), 0, 0, 0, 0, false),
        };
        out.push(ModelEntry {
            id,
            path: gguf.to_string_lossy().to_string(),
            arch,
            name,
            experts,
            experts_used,
            context,
            file_bytes,
            is_moe,
        });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// Resolve a model name (or path) to a GGUF path.
pub fn resolve(root: &Path, name: &str) -> Option<PathBuf> {
    // Direct path
    let p = PathBuf::from(name);
    if p.exists() {
        return Some(p);
    }
    // rocm/models/<name>/model.gguf
    let candidate = root.join(name).join("model.gguf");
    if candidate.exists() {
        return Some(candidate);
    }
    // partial match
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.flatten() {
            let dir = e.path();
            if !dir.is_dir() {
                continue;
            }
            let id = dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if id.contains(name) {
                let gguf = dir.join("model.gguf");
                if gguf.exists() {
                    return Some(gguf);
                }
            }
        }
    }
    None
}

/// Default models root.
pub fn default_root() -> PathBuf {
    PathBuf::from("rocm").join("models")
}

/// Ask a running server to load `model` in place of whatever it holds.
///
/// The server owns the compute process, so this is a request rather than an
/// action: it posts to `/api/models/load` and reports what came back. A failed
/// load is reported as a failure, not swallowed, because the server rolls back
/// to the previous model and the caller needs to know which one is resident.
pub async fn switch(url: &str, model: &str, api_key: Option<&str>) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let endpoint = format!("{}/api/models/load", url.trim_end_matches('/'));
    let mut req = client
        .post(&endpoint)
        .json(&serde_json::json!({ "model": model }));
    if let Some(key) = api_key.filter(|k| !k.trim().is_empty()) {
        req = req.bearer_auth(key);
    }

    println!("Switching {url} to {model}");
    let resp = req
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("could not reach {url}: {e} (is `strata serve` running?)"))?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);

    if status.is_success() {
        println!("  loaded {model}");
        return Ok(());
    }

    let message = body
        .get("error")
        .and_then(|v| v.as_str())
        .unwrap_or("switch failed");
    match body.get("rolled_back_to").and_then(|v| v.as_str()) {
        Some(previous) => anyhow::bail!("{message}\n  still serving {previous}"),
        None => anyhow::bail!("{message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_directory_resolves_to_its_gguf() {
        let root = std::env::temp_dir().join("strata-registry-resolve");
        let dir = root.join("tiny");
        std::fs::create_dir_all(&dir).unwrap();
        let gguf = dir.join("model.gguf");
        std::fs::write(&gguf, b"not really a gguf").unwrap();

        assert_eq!(resolve(&root, "tiny"), Some(gguf.clone()));
        // A partial name still finds it, because typing the whole directory
        // name is the thing people get wrong.
        assert_eq!(resolve(&root, "tin"), Some(gguf));
        assert_eq!(resolve(&root, "absent"), None);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_directory_without_a_gguf_is_not_a_model() {
        let root = std::env::temp_dir().join("strata-registry-empty");
        let dir = root.join("hollow");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();

        assert_eq!(resolve(&root, "hollow"), None);
        assert!(scan(&root).is_empty());

        std::fs::remove_dir_all(&root).ok();
    }
}
