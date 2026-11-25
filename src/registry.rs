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
    /// Derived from the tensor index; see gguf::ModelFacts.
    pub parameter_count: u64,
    pub quantization_level: String,
    pub embedding: usize,
    pub vocab: usize,
    pub families: Vec<String>,
    pub modified_secs: u64,
    /// Stable opaque identifier. Not a content digest: hashing a 20 GB file on
    /// every `/api/tags` would be absurd. Clients treat it as an id.
    pub digest: String,
    /// A projector GGUF beside the model. Drives the `vision` capability.
    pub has_projector: bool,
    /// Scalar GGUF header declarations, exposed as Ollama's `model_info`.
    pub metadata_info: Vec<(String, String)>,
}

impl ModelEntry {
    /// A short, stable id for a model. It changes when the facts that identify
    /// a model change, and stays the same across restarts.
    fn digest(arch: &str, name: &str, file_bytes: u64, context: usize, params: u64) -> String {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (arch, name, file_bytes, context, params).hash(&mut h);
        format!("strata-{:016x}", h.finish())
    }
}

/// Logical Ollama names (`team/model:tag`) mapped to the safe directory names
/// used on disk. Keep it beside the models so moving the store moves its tags.
fn tags_path(root: &Path) -> PathBuf {
    root.join("tags.json")
}

fn load_tags(root: &Path) -> serde_json::Map<String, serde_json::Value> {
    std::fs::read_to_string(tags_path(root))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

fn save_tags(root: &Path, tags: &serde_json::Map<String, serde_json::Value>) -> anyhow::Result<()> {
    std::fs::write(tags_path(root), serde_json::to_string_pretty(tags)?)?;
    Ok(())
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
        let opened = crate::gguf::Gguf::open(&gguf).ok();
        let facts = opened
            .as_ref()
            .and_then(|g| crate::gguf::ModelFacts::read(g).ok());
        let (arch, name, experts, experts_used, context, file_bytes, is_moe) = match &facts {
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
        let parameter_count = facts.as_ref().map(|f| f.parameter_count).unwrap_or(0);
        let quantization_level = facts
            .as_ref()
            .map(|f| f.quantization_level.clone())
            .unwrap_or_else(|| "unknown".into());
        let embedding = facts.as_ref().map(|f| f.embedding).unwrap_or(0);
        let vocab = facts.as_ref().map(|f| f.vocab).unwrap_or(0);
        let metadata_info = opened
            .as_ref()
            .map(|g| g.metadata_info())
            .unwrap_or_default();
        let modified_secs = std::fs::metadata(&gguf)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let has_projector = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.flatten().any(|f| {
                    f.file_name()
                        .to_string_lossy()
                        .to_ascii_lowercase()
                        .starts_with("mmproj")
                })
            })
            .unwrap_or(false);
        let digest = ModelEntry::digest(&arch, &name, file_bytes, context, parameter_count);
        let families = if arch == "unknown" {
            Vec::new()
        } else {
            vec![arch.clone()]
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
            parameter_count,
            quantization_level,
            embedding,
            vocab,
            families,
            modified_secs,
            digest,
            has_projector,
            metadata_info,
        });
    }
    let tags = load_tags(root);
    for entry in &mut out {
        if let Some(logical) = tags.iter().find_map(|(name, dir)| {
            (dir.as_str() == Some(entry.id.as_str())).then_some(name.clone())
        }) {
            entry.id = logical;
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// Resolve a model name (or path) to a GGUF path.
pub fn resolve(root: &Path, name: &str) -> Option<PathBuf> {
    // A logical tag ("team/model:tag") points at a safe directory name.
    if let Some(dir) = load_tags(root).get(name).and_then(|v| v.as_str()) {
        let candidate = root.join(dir).join("model.gguf");
        if candidate.exists() {
            return Some(candidate);
        }
    }
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

/// Default models root. `OLLAMA_MODELS` is honoured so an existing Ollama
/// installation's store can be reused by pointing Strata at it.
pub fn default_root() -> PathBuf {
    std::env::var("OLLAMA_MODELS")
        .or_else(|_| std::env::var("STRATA_MODELS"))
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("rocm").join("models"))
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

/// A model name that cannot escape the models root.
///
/// Ollama names may contain `namespace/model:tag`. Strata stores one directory
/// per model, so those characters are folded rather than turned into a path.
pub fn safe_name(name: &str) -> anyhow::Result<String> {
    let name = name.trim();
    anyhow::ensure!(!name.is_empty(), "empty model name");
    let kept: String = name
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '/'))
        .map(|c| if c == ':' || c == '/' { '_' } else { c })
        .collect();
    anyhow::ensure!(
        !kept.is_empty() && !kept.starts_with('.') && !kept.contains(".."),
        "unsafe model name: {name}"
    );
    Ok(kept)
}

/// `strata cp` / `POST /api/copy`: duplicate a model under a new name.
///
/// Files are hard-linked where the filesystem allows it, so two tags of one
/// model do not each consume 20 GB. A cross-device link falls back to a copy.
pub fn copy(root: &Path, from: &str, to: &str) -> anyhow::Result<()> {
    let src = resolve(root, from).ok_or_else(|| anyhow::anyhow!("model '{from}' not found"))?;
    let src_dir = src
        .parent()
        .ok_or_else(|| anyhow::anyhow!("model '{from}' has no directory"))?;
    let dst_dir = root.join(safe_name(to)?);
    anyhow::ensure!(!dst_dir.exists(), "model '{to}' already exists");
    std::fs::create_dir_all(&dst_dir)?;
    for entry in std::fs::read_dir(src_dir)?.flatten() {
        if !entry.path().is_file() {
            continue;
        }
        let dst = dst_dir.join(entry.file_name());
        println!(
            "copying {} ({:.1} MB)",
            dst.file_name().unwrap_or_default().to_string_lossy(),
            entry
                .metadata()
                .map(|m| m.len() as f64 / 1e6)
                .unwrap_or(0.0)
        );
        if std::fs::hard_link(entry.path(), &dst).is_err() {
            std::fs::copy(entry.path(), &dst)?;
        }
    }
    std::fs::write(
        dst_dir.join("strata-origin.json"),
        serde_json::json!({ "copied_from": from }).to_string(),
    )?;
    if to != safe_name(to)? {
        let mut tags = load_tags(root);
        tags.insert(to.to_string(), serde_json::json!(safe_name(to)?));
        save_tags(root, &tags)?;
    }
    Ok(())
}

/// `strata rm` / `DELETE /api/delete`: remove a model directory.
pub fn remove(root: &Path, name: &str) -> anyhow::Result<()> {
    let dir = root.join(safe_name(name)?);
    anyhow::ensure!(dir.exists(), "model '{name}' not found");
    let root_canonical = std::fs::canonicalize(root)?;
    let dir_canonical = std::fs::canonicalize(&dir)?;
    anyhow::ensure!(
        dir_canonical.starts_with(&root_canonical),
        "model '{name}' is outside the models root"
    );
    let stored = safe_name(name)?;
    std::fs::remove_dir_all(dir_canonical)?;
    let mut tags = load_tags(root);
    tags.retain(|_, dir| dir.as_str() != Some(stored.as_str()));
    save_tags(root, &tags)?;
    Ok(())
}

/// `POST /api/create` from an existing model: copy it, then record the
/// Modelfile overrides beside the weights so `/api/show` can report them.
pub fn create_from(
    root: &Path,
    name: &str,
    from: &str,
    overrides: &serde_json::Value,
) -> anyhow::Result<()> {
    copy(root, from, name)?;
    let dir = root.join(safe_name(name)?);
    std::fs::write(
        dir.join("strata.json"),
        serde_json::to_string_pretty(overrides)?,
    )?;
    tag(root, name, &safe_name(name)?)?;
    Ok(())
}

/// `POST /api/create` with `files`: build a model from blobs already uploaded
/// through `/api/blobs/:digest`.
pub fn create_from_blobs(
    root: &Path,
    name: &str,
    files: &serde_json::Map<String, serde_json::Value>,
    overrides: &serde_json::Value,
) -> anyhow::Result<()> {
    let dir = root.join(safe_name(name)?);
    anyhow::ensure!(!dir.exists(), "model '{name}' already exists");
    anyhow::ensure!(!files.is_empty(), "files must name at least one blob");
    std::fs::create_dir_all(&dir)?;
    let mut has_gguf = false;
    for (filename, digest) in files {
        let clean = filename
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(filename)
            .to_string();
        anyhow::ensure!(
            !clean.is_empty() && !clean.contains("..") && !clean.starts_with('.'),
            "unsafe filename in files: {filename}"
        );
        let digest = digest
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("digest for {filename} must be a string"))?;
        let blob = crate::blob::blob_path(root, digest)?;
        anyhow::ensure!(
            blob.exists(),
            "blob {digest} is not on this server; upload it with POST /api/blobs/:digest first"
        );
        let dest = dir.join(&clean);
        if std::fs::hard_link(&blob, &dest).is_err() {
            std::fs::copy(&blob, &dest)?;
        }
        has_gguf |= clean.to_ascii_lowercase().ends_with(".gguf");
    }
    anyhow::ensure!(has_gguf, "at least one file must be a .gguf");
    std::fs::write(
        dir.join("strata.json"),
        serde_json::to_string_pretty(overrides)?,
    )?;
    tag(root, name, &safe_name(name)?)?;
    Ok(())
}

/// Record a logical Ollama name for a stored directory.
pub fn tag(root: &Path, logical: &str, stored: &str) -> anyhow::Result<()> {
    if logical == stored {
        return Ok(());
    }
    let mut tags = load_tags(root);
    tags.insert(logical.to_string(), serde_json::json!(stored));
    save_tags(root, &tags)
}

/// Read the Modelfile overrides stored beside a derived model.
pub fn overrides(root: &Path, name: &str) -> serde_json::Value {
    std::fs::read_to_string(
        root.join(safe_name(name).unwrap_or_default())
            .join("strata.json"),
    )
    .ok()
    .and_then(|s| serde_json::from_str(&s).ok())
    .unwrap_or(serde_json::Value::Null)
}

/// `strata create <name> -f Modelfile`.
///
/// `FROM` may name an installed model or a path to a GGUF. A GGUF path is
/// imported into the new model directory; an installed model is hard-linked.
pub fn create_from_modelfile(
    root: &Path,
    name: &str,
    m: &crate::modelfile::Modelfile,
) -> anyhow::Result<()> {
    let target = root.join(safe_name(name)?);
    anyhow::ensure!(!target.exists(), "model '{name}' already exists");
    let from = m.from.trim();
    let from_path = std::path::Path::new(from);
    if from_path.is_file() {
        std::fs::create_dir_all(&target)?;
        let dest = target.join("model.gguf");
        if std::fs::hard_link(from_path, &dest).is_err() {
            std::fs::copy(from_path, &dest)?;
        }
    } else {
        create_from(root, name, from, &m.to_overrides())?;
    }
    std::fs::write(
        target.join("strata.json"),
        serde_json::to_string_pretty(&m.to_overrides())?,
    )?;
    tag(root, name, &safe_name(name)?)?;
    Ok(())
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
    fn model_names_cannot_escape_the_root() {
        assert!(safe_name("llama3.2:latest").is_ok());
        assert_eq!(safe_name("team/model:tag").unwrap(), "team_model_tag");
        assert!(safe_name("../etc/passwd").is_err());
        assert!(safe_name("..").is_err());
        assert!(safe_name("").is_err());
    }

    #[test]
    fn copying_a_model_keeps_both_names_and_removal_removes_one() {
        let root = std::env::temp_dir().join("strata-registry-copy");
        std::fs::remove_dir_all(&root).ok();
        let src = root.join("one");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("model.gguf"), b"not really a gguf").unwrap();

        copy(&root, "one", "two").unwrap();
        assert!(root.join("two").join("model.gguf").exists());
        assert!(root.join("one").join("model.gguf").exists());
        assert!(copy(&root, "one", "two").is_err(), "overwrite must fail");

        remove(&root, "two").unwrap();
        assert!(!root.join("two").exists());
        assert!(root.join("one").join("model.gguf").exists());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn logical_ollama_names_resolve_to_stored_directories() {
        let root = std::env::temp_dir().join("strata-registry-tags");
        std::fs::remove_dir_all(&root).ok();
        let dir = root.join("team_model_latest");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.gguf"), b"not really a gguf").unwrap();

        tag(&root, "team/model:latest", "team_model_latest").unwrap();
        assert_eq!(
            resolve(&root, "team/model:latest"),
            Some(dir.join("model.gguf"))
        );
        assert!(scan(&root).iter().any(|e| e.id == "team/model:latest"));
        remove(&root, "team/model:latest").unwrap();
        assert!(load_tags(&root).is_empty());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_model_can_be_built_from_uploaded_blobs() {
        let root = std::env::temp_dir().join("strata-registry-blobs");
        std::fs::remove_dir_all(&root).ok();
        let blobs = crate::blob::blob_dir(&root);
        std::fs::create_dir_all(&blobs).unwrap();
        let digest = "a".repeat(64);
        std::fs::write(blobs.join(format!("sha256-{digest}")), b"gguf bytes").unwrap();

        let mut files = serde_json::Map::new();
        files.insert(
            "model.gguf".into(),
            serde_json::json!(format!("sha256:{digest}")),
        );
        create_from_blobs(&root, "from-blob", &files, &serde_json::json!({})).unwrap();
        assert_eq!(
            std::fs::read(root.join("from-blob").join("model.gguf")).unwrap(),
            b"gguf bytes"
        );
        assert!(create_from_blobs(&root, "from-blob", &files, &serde_json::json!({})).is_err());
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
