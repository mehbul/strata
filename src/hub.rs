//! Getting a model onto this machine.
//!
//! Three sources, cheapest first: a copy already under `rocm/models`, a blob
//! Ollama has downloaded, or a GGUF from a named Hugging Face repository.
//!
//! Everything here ends at `rocm/models/<name>/model.gguf`, because that is
//! where `plan`, `tune` and `serve` look for it.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub struct Hub {
    /// Where models are kept. `rocm/models`, matching `placement::find`.
    pub cache_dir: PathBuf,
    /// Optional sink for progress lines, used by `/api/pull` to stream NDJSON
    /// while the CLI keeps printing to stdout.
    progress: Option<std::sync::Arc<dyn Fn(String) + Send + Sync>>,
}

impl Hub {
    pub fn new() -> Self {
        Self {
            cache_dir: PathBuf::from("rocm/models"),
            progress: None,
        }
    }

    pub fn with_progress(sink: impl Fn(String) + Send + Sync + 'static) -> Self {
        Self {
            cache_dir: PathBuf::from("rocm/models"),
            progress: Some(std::sync::Arc::new(sink)),
        }
    }

    fn say(&self, message: String) {
        println!("{message}");
        if let Some(sink) = &self.progress {
            sink(message);
        }
    }

    /// `strata pull --model <name>`.
    ///
    /// `<name>` is either an Ollama model (`ornith-1.5:35b`) or a Hugging Face
    /// repository (`owner/repo`, or `owner/repo:file.gguf` when the repository
    /// holds more than one). A bare name Ollama does not have is an error
    /// rather than a guess: inventing an owner for it only turns a typo into a
    /// 404 from a repository the caller never named.
    pub async fn pull(&self, model: &str) -> Result<PathBuf> {
        let force_registry = model.starts_with("ollama:");
        let model = model.strip_prefix("ollama:").unwrap_or(model);
        let out = self.cache_dir.join(Self::dir_name(model));
        let dest = out.join("model.gguf");

        // 1. A usable local copy means there is nothing to fetch. "Usable" is
        //    decided by reading the header, not by file size: that is true of a
        //    500 MB model and false of a truncated 30 GB download.
        if is_gguf(&dest) {
            self.say(format!(
                "hub: {model} is already at {} ({:.1} GB)",
                dest.display(),
                file_gb(&dest)
            ));
            return Ok(out);
        }

        // 2. A blob Ollama has already downloaded, found through its manifests.
        if let Some(blob) = ollama_blob(model) {
            std::fs::create_dir_all(&out)?;
            self.say(format!(
                "hub: Ollama has {model} ({:.1} GB)",
                file_gb(&blob)
            ));
            self.say(format!("     {} -> {}", blob.display(), dest.display()));
            std::fs::copy(&blob, &dest)
                .with_context(|| format!("copying {} to {}", blob.display(), dest.display()))?;
            // Vision models ship the projector as a second Ollama layer. Strata
            // discovers it by name in the model directory, so give it one.
            if let Some(projector) = ollama_projector(model) {
                let proj_dest = out.join("model.mmproj.gguf");
                self.say(format!(
                    "     {} -> {}",
                    projector.display(),
                    proj_dest.display()
                ));
                std::fs::copy(&projector, &proj_dest)
                    .with_context(|| format!("copying projector to {}", proj_dest.display()))?;
            }
            return Ok(out);
        }

        // 3. Ollama's own registry for a plain model name (or an explicit
        //    `ollama:` prefix for a namespaced one).
        if force_registry || hf_ref(model).is_none() {
            return self.pull_from_registry(model, &out).await;
        }

        // 4. Hugging Face, for a repository named in full.
        let Some((repo, wanted)) = hf_ref(model) else {
            bail!("nothing called {model} to pull")
        };

        let client = reqwest::Client::new();
        let files = gguf_files(&client, repo).await?;
        let file = match (wanted, files.len()) {
            (Some(w), _) => files
                .iter()
                .find(|f| f.as_str() == w || f.ends_with(&format!("/{w}")))
                .cloned()
                .with_context(|| {
                    format!("{repo} has no {w}. It holds:\n  {}", files.join("\n  "))
                })?,
            (None, 1) => files[0].clone(),
            (None, 0) => bail!(
                "{repo} holds no .gguf files. Strata loads GGUF, so a repository \
                 of safetensors has to be converted first."
            ),
            (None, _) => bail!(
                "{repo} holds {} GGUF files. Name one:\n  {}\n\n  \
                 strata pull --model {repo}:<file>",
                files.len(),
                files.join("\n  ")
            ),
        };

        std::fs::create_dir_all(&out)?;
        self.say(format!(
            "hub: pulling https://huggingface.co/{repo} -> {}",
            dest.display()
        ));
        self.download(&client, repo, &file, &dest).await?;
        if !is_gguf(&dest) {
            bail!(
                "{} downloaded but does not start with the GGUF magic",
                dest.display()
            );
        }
        self.say(format!(
            "hub: {model} ready at {} ({:.1} GB)",
            dest.display(),
            file_gb(&dest)
        ));
        Ok(out)
    }

    /// The directory this model gets, which is also the name it is served by.
    ///
    /// It has to be short enough to type after `--model`, and distinct per
    /// quantisation: naming the directory after the whole reference gives
    /// `TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF_tinyllama-1.1b-chat-v1.0.Q2_K.gguf`,
    /// which nobody is going to type. A named file supplies the name, since two
    /// quants of one repository must not land in the same directory; otherwise
    /// the repository's own last segment does.
    pub fn dir_name(model: &str) -> String {
        let name = match hf_ref(model) {
            Some((_, Some(file))) => {
                let base = file.rsplit('/').next().unwrap_or(file);
                base.strip_suffix(".gguf")
                    .or_else(|| base.strip_suffix(".GGUF"))
                    .unwrap_or(base)
            }
            Some((repo, None)) => repo.rsplit('/').next().unwrap_or(repo),
            None => model,
        };
        Self::sanitize_model(name)
    }

    /// Only what can appear in a directory name, and nothing that climbs out of one.
    fn sanitize_model(model: &str) -> String {
        let kept: String = model
            .chars()
            .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | ':' | '/' | '.'))
            .collect();
        kept.replace([':', '/'], "_").replace("..", "_")
    }

    /// Pull a model straight from Ollama's public registry. No Ollama install
    /// is needed: the manifest names content-addressed layers, and those blobs
    /// are downloadable anonymously for public models.
    async fn pull_from_registry(&self, model: &str, out: &std::path::Path) -> Result<PathBuf> {
        let (name, tag) = match model.split_once(':') {
            Some((n, t)) => (n, t),
            None => (model, "latest"),
        };
        let repo = if name.contains('/') {
            name.to_string()
        } else {
            format!("library/{name}")
        };
        let base = std::env::var("OLLAMA_REGISTRY")
            .unwrap_or_else(|_| "https://registry.ollama.ai".into());
        let base = base.trim_end_matches('/').to_string();
        let client = reqwest::Client::new();
        self.say(format!("hub: pulling manifest for {repo}:{tag}"));

        let manifest_url = format!("{base}/v2/{repo}/manifests/{tag}");
        let mut request = client.get(&manifest_url).header(
            "Accept",
            "application/vnd.docker.distribution.manifest.v2+json",
        );
        if let Some(token) = crate::auth::token() {
            request = request.bearer_auth(token);
        }
        let resp = request
            .send()
            .await
            .with_context(|| format!("GET {manifest_url}"))?;
        if !resp.status().is_success() {
            bail!("registry returned {} for {repo}:{tag}", resp.status());
        }
        let manifest: Value = resp.json().await?;
        let layers = manifest
            .get("layers")
            .and_then(Value::as_array)
            .context("registry manifest has no layers")?;

        let blob_dir = crate::blob::blob_dir(&self.cache_dir);
        std::fs::create_dir_all(&blob_dir)?;
        std::fs::create_dir_all(out)?;
        let mut weights: Option<PathBuf> = None;
        let mut projector: Option<PathBuf> = None;
        let mut system = String::new();
        let mut template = String::new();
        let mut renderer = String::new();
        let mut parser = String::new();
        let mut license = String::new();
        let mut parameters = serde_json::Map::new();
        let mut total: u64 = 0;
        let wanted: u64 = layers
            .iter()
            .filter_map(|l| l.get("size").and_then(Value::as_u64))
            .sum();

        for layer in layers {
            let media = layer.get("mediaType").and_then(Value::as_str).unwrap_or("");
            let digest = layer
                .get("digest")
                .and_then(Value::as_str)
                .context("registry layer has no digest")?;
            let size = layer.get("size").and_then(Value::as_u64).unwrap_or(0);
            let blob = crate::blob::blob_path(&self.cache_dir, digest)?;
            if !blob.exists() {
                self.download_registry_blob(&client, &base, &repo, digest, size, &blob)
                    .await?;
            }
            total += size;
            self.say(format!("hub: layer {digest} ({:.1} GB)", size as f32 / 1e9));
            match media {
                "application/vnd.ollama.image.model" => weights = Some(blob),
                "application/vnd.ollama.image.projector" => projector = Some(blob),
                "application/vnd.ollama.image.system" => {
                    system = std::fs::read_to_string(&blob).unwrap_or_default()
                }
                "application/vnd.ollama.image.template" => {
                    template = std::fs::read_to_string(&blob).unwrap_or_default()
                }
                "application/vnd.ollama.image.renderer" => {
                    renderer = std::fs::read_to_string(&blob).unwrap_or_default()
                }
                "application/vnd.ollama.image.parser" => {
                    parser = std::fs::read_to_string(&blob).unwrap_or_default()
                }
                "application/vnd.ollama.image.license" => {
                    license = std::fs::read_to_string(&blob).unwrap_or_default()
                }
                "application/vnd.ollama.image.parameters" => {
                    let text = std::fs::read_to_string(&blob).unwrap_or_default();
                    for line in text.lines() {
                        let mut parts = line.splitn(2, char::is_whitespace);
                        if let (Some(k), Some(v)) = (parts.next(), parts.next()) {
                            parameters.insert(k.into(), Value::String(v.trim().into()));
                        }
                    }
                }
                _ => {}
            }
        }

        let src = weights.context("registry manifest has no model layer")?;
        let dest = out.join("model.gguf");
        if std::fs::hard_link(&src, &dest).is_err() {
            std::fs::copy(&src, &dest)?;
        }
        if let Some(p) = projector {
            let to = out.join("mmproj.gguf");
            let fallback = p.clone();
            if std::fs::hard_link(p, &to).is_err() {
                std::fs::copy(fallback, &to)?;
            }
        }
        let mut overrides = serde_json::Map::new();
        if !system.is_empty() {
            overrides.insert("system".into(), json!(system));
        }
        if !template.is_empty() {
            overrides.insert("template".into(), json!(template));
        }
        if !renderer.is_empty() {
            overrides.insert("renderer".into(), json!(renderer));
        }
        if !parser.is_empty() {
            overrides.insert("parser".into(), json!(parser));
        }
        if !license.is_empty() {
            overrides.insert("license".into(), json!(license));
        }
        if !parameters.is_empty() {
            overrides.insert("parameters".into(), Value::Object(parameters));
        }
        if !overrides.is_empty() {
            std::fs::write(
                out.join("strata.json"),
                serde_json::to_string_pretty(&Value::Object(overrides))?,
            )?;
        }
        crate::registry::tag(&self.cache_dir, model, &Self::dir_name(model))?;
        let _ = (total, wanted);
        self.say(format!(
            "hub: {model} ready at {} ({:.1} GB)",
            dest.display(),
            file_gb(&dest)
        ));
        Ok(out.to_path_buf())
    }

    async fn download_registry_blob(
        &self,
        client: &reqwest::Client,
        base: &str,
        repo: &str,
        digest: &str,
        size: u64,
        dest: &std::path::Path,
    ) -> Result<()> {
        use futures::StreamExt;
        use tokio::io::AsyncWriteExt;
        let url = format!("{base}/v2/{repo}/blobs/{digest}");
        let partial = dest.with_extension("partial");
        let mut request = client.get(&url);
        if let Some(token) = crate::auth::token() {
            request = request.bearer_auth(token);
        }
        let resp = request.send().await.with_context(|| format!("GET {url}"))?;
        if !resp.status().is_success() {
            bail!("blob download returned {}", resp.status());
        }
        let mut file = tokio::fs::File::create(&partial).await?;
        let mut stream = resp.bytes_stream();
        let mut done = 0u64;
        let mut next_report = 500_000_000u64;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            done += chunk.len() as u64;
            if done >= next_report {
                self.say(format!(
                    "  {:.1} / {:.1} GB",
                    done as f32 / 1e9,
                    size as f32 / 1e9
                ));
                next_report = done + 500_000_000;
            }
        }
        file.flush().await?;
        drop(file);
        let got = crate::blob::sha256_hex(std::fs::File::open(&partial)?)?;
        let expected = digest.strip_prefix("sha256:").unwrap_or(digest);
        anyhow::ensure!(
            got == expected,
            "digest mismatch for {digest}: got sha256:{got}"
        );
        tokio::fs::rename(&partial, dest).await?;
        Ok(())
    }

    /// Stream one file down, resuming a partial one where the server allows it.
    ///
    /// A failure here returns an error. It used to write a few bytes of apology
    /// to the destination and report a verified download, which left a text
    /// file named `model.gguf` for `serve` to fail on much later.
    async fn download(
        &self,
        client: &reqwest::Client,
        repo: &str,
        fname: &str,
        dest: &Path,
    ) -> Result<()> {
        let url = format!("https://huggingface.co/{repo}/resolve/main/{fname}");
        let partial = dest.with_extension("gguf.partial");
        let resume_from = std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);

        let mut req = client.get(&url);
        if resume_from > 0 {
            self.say(format!("  resuming at {:.1} GB", resume_from as f32 / 1e9));
            req = req.header("Range", format!("bytes={resume_from}-"));
        }
        let res = req.send().await.with_context(|| format!("GET {url}"))?;
        let status = res.status();
        if !(status.is_success() || status.as_u16() == 206) {
            bail!("{url} -> {status}. A gated repository needs HF_TOKEN.");
        }
        // A server that ignores Range answers 200 with the whole file, so the
        // partial has to be replaced rather than appended to.
        let append = status.as_u16() == 206 && resume_from > 0;

        let total = res.content_length().unwrap_or(0) + if append { resume_from } else { 0 };
        let mut file = if append {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(&partial)
                .await?
        } else {
            tokio::fs::File::create(&partial).await?
        };

        use futures::StreamExt;
        use tokio::io::AsyncWriteExt;
        let mut stream = res.bytes_stream();
        let mut done = if append { resume_from } else { 0 };
        let mut next_report = done + 500_000_000;
        while let Some(chunk) = stream.next().await {
            let c = chunk.context("download interrupted")?;
            file.write_all(&c).await?;
            done += c.len() as u64;
            if done >= next_report {
                self.say(match total {
                    0 => format!("  {:.1} GB", done as f32 / 1e9),
                    t => format!("  {:.1} / {:.1} GB", done as f32 / 1e9, t as f32 / 1e9),
                });
                next_report = done + 500_000_000;
            }
        }
        file.flush().await?;
        drop(file);
        tokio::fs::rename(&partial, dest).await?;
        Ok(())
    }
}

/// The `.gguf` files a Hugging Face repository lists.
async fn gguf_files(client: &reqwest::Client, repo: &str) -> Result<Vec<String>> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    let res = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    if !res.status().is_success() {
        bail!("huggingface.co/{repo} -> {}", res.status());
    }
    let json: Value = res.json().await.context("Hugging Face returned no JSON")?;
    let mut files: Vec<String> = json
        .get("siblings")
        .and_then(Value::as_array)
        .map(|s| {
            s.iter()
                .filter_map(|f| f.get("rfilename").and_then(Value::as_str))
                .filter(|f| f.to_lowercase().ends_with(".gguf"))
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    Ok(files)
}

/// A Hugging Face reference, if this name is one.
///
/// `owner/repo` or `owner/repo:file.gguf`. The slash is what distinguishes it
/// from an Ollama `name:tag`, so a name without one is never sent to the API.
fn hf_ref(model: &str) -> Option<(&str, Option<&str>)> {
    match model.split_once(':') {
        Some((repo, file)) if repo.contains('/') => Some((repo, Some(file))),
        None if model.contains('/') => Some((model, None)),
        _ => None,
    }
}

fn file_gb(path: &Path) -> f32 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) as f32 / 1e9
}

/// Whether a file starts with the GGUF magic.
///
/// Reading four bytes is a better test of "is this a model" than any size
/// threshold: it is true for a 500 MB model and false for a truncated 30 GB
/// download.
fn is_gguf(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).is_ok() && &magic == b"GGUF"
}

/// Where Ollama keeps its store on this machine.
fn ollama_root() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("OLLAMA_MODELS") {
        return Some(PathBuf::from(dir));
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()?;
    Some(PathBuf::from(home).join(".ollama").join("models"))
}

/// The blob Ollama holds for `name[:tag]`.
fn ollama_blob(model: &str) -> Option<PathBuf> {
    ollama_blob_in(&ollama_root()?, model)
}

/// The projector shipped beside `name[:tag]`, when the model has one.
fn ollama_projector(model: &str) -> Option<PathBuf> {
    ollama_layer_in(
        &ollama_root()?,
        model,
        "application/vnd.ollama.image.projector",
    )
}

/// The blob a store at `root` holds for `name[:tag]`, found through its manifests.
fn ollama_blob_in(root: &Path, model: &str) -> Option<PathBuf> {
    ollama_layer_in(root, model, "application/vnd.ollama.image.model")
}

/// A named layer of `name[:tag]`, found through Ollama's manifests.
///
/// Ollama names blobs by digest, so the manifest tree is the only thing that
/// says which blob is which model. Taking the largest GGUF instead - as this
/// used to - answers every pull with whatever the biggest model on the machine
/// happens to be. Newer Ollama releases write `manifests-v2` rather than
/// `manifests`; both layouts are checked so a locally cached model is reused
/// instead of being downloaded from the registry again.
fn ollama_layer_in(root: &Path, model: &str, media_type: &str) -> Option<PathBuf> {
    let (name, tag) = match model.split_once(':') {
        Some((n, t)) => (n.to_lowercase(), Some(t.to_lowercase())),
        None => (model.to_lowercase(), None),
    };
    let manifest = find_manifest(&root.join("manifests"), &name, tag.as_deref())
        .or_else(|| find_manifest(&root.join("manifests-v2"), &name, tag.as_deref()))?;
    let json: Value = serde_json::from_slice(&std::fs::read(manifest).ok()?).ok()?;
    let digest = json
        .get("layers")?
        .as_array()?
        .iter()
        .find(|l| l.get("mediaType").and_then(Value::as_str) == Some(media_type))?
        .get("digest")?
        .as_str()?
        .replace(':', "-");
    let blob = root.join("blobs").join(digest);
    blob.exists().then_some(blob)
}

/// The manifest file for `name`, at `tag` when one was asked for.
///
/// Manifests live at `manifests/<registry>/<namespace>/<name>/<tag>`, so the
/// model name is a directory and the tag is a file inside it. Without a tag,
/// `latest` wins if it exists and otherwise the only tag present: a model with
/// several tags and none asked for is ambiguous, and answering it by picking
/// one is how the wrong weights arrive under the right name.
fn find_manifest(dir: &Path, name: &str, tag: Option<&str>) -> Option<PathBuf> {
    let entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .collect();
    let here = dir.file_name()?.to_string_lossy().to_lowercase();

    if here == name || name.ends_with(&format!("/{here}")) {
        let tags: Vec<&PathBuf> = entries.iter().filter(|p| p.is_file()).collect();
        let named = |t: &str| {
            tags.iter()
                .find(|p| {
                    p.file_name()
                        .is_some_and(|f| f.to_string_lossy().to_lowercase() == t)
                })
                .map(|p| (*p).clone())
        };
        if let Some(t) = tag {
            return named(t);
        }
        if let Some(latest) = named("latest") {
            return Some(latest);
        }
        if tags.len() == 1 {
            return Some(tags[0].clone());
        }
    }

    entries
        .iter()
        .filter(|p| p.is_dir())
        .find_map(|sub| find_manifest(sub, name, tag))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An Ollama store holding the named models, each with a projector beside it.
    fn store(tag: &str, models: &[(&str, &str, &str)]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("strata-hub-{tag}"));
        let _ = std::fs::remove_dir_all(&root);
        let lib = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library");
        std::fs::create_dir_all(root.join("blobs")).unwrap();
        for (name, tag, digest) in models {
            std::fs::create_dir_all(lib.join(name)).unwrap();
            let manifest = format!(
                "{{\"layers\":[\
                   {{\"mediaType\":\"application/vnd.ollama.image.projector\",\"digest\":\"sha256:{digest}pro\"}},\
                   {{\"mediaType\":\"application/vnd.ollama.image.model\",\"digest\":\"sha256:{digest}\"}}\
                 ]}}"
            );
            std::fs::write(lib.join(name).join(tag), manifest).unwrap();
            std::fs::write(root.join("blobs").join(format!("sha256-{digest}")), b"GGUF").unwrap();
            std::fs::write(
                root.join("blobs").join(format!("sha256-{digest}pro")),
                b"GGUF",
            )
            .unwrap();
        }
        root
    }

    #[test]
    fn a_model_smaller_than_ten_gigabytes_is_still_a_model() {
        // The old check required 10 GB, so every small model looked absent.
        let dir = std::env::temp_dir().join("strata-hub-test");
        let _ = std::fs::create_dir_all(&dir);
        let small = dir.join("small.gguf");
        std::fs::write(&small, b"GGUF\x03\x00\x00\x00rest of a small model").unwrap();
        assert!(is_gguf(&small));

        let not_a_model = dir.join("notes.txt");
        std::fs::write(&not_a_model, b"this is not a model").unwrap();
        assert!(!is_gguf(&not_a_model));

        assert!(!is_gguf(&dir.join("does-not-exist.gguf")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The bug this replaced: whichever blob was biggest answered every pull.
    #[test]
    fn each_model_resolves_to_its_own_blob() {
        let root = store(
            "two",
            &[("small-one", "3b", "aaa"), ("big-one", "70b", "bbb")],
        );
        let small = ollama_blob_in(&root, "small-one:3b").unwrap();
        assert!(small.ends_with("sha256-aaa"), "{}", small.display());
        assert!(ollama_blob_in(&root, "big-one:70b")
            .unwrap()
            .ends_with("sha256-bbb"));
        assert!(ollama_blob_in(&root, "not-here:1b").is_none());
        assert!(ollama_blob_in(&root, "small-one:70b").is_none());
        assert!(ollama_layer_in(
            &root,
            "small-one:3b",
            "application/vnd.ollama.image.projector"
        )
        .unwrap()
        .ends_with("sha256-aaapro"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Ollama 0.40 stores its manifests under `manifests-v2`; that is where
    /// this machine's Inkling 2 Mini lives.
    #[test]
    fn a_v2_manifest_tree_resolves_the_model_and_its_projector() {
        let root = std::env::temp_dir().join("strata-hub-v2-test");
        let _ = std::fs::remove_dir_all(&root);
        let lib = root
            .join("manifests-v2")
            .join("ollama.com")
            .join("library")
            .join("inkling-2-mini");
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::create_dir_all(root.join("blobs")).unwrap();
        std::fs::write(
            lib.join("chat"),
            r#"{"layers":[
                {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:ddd"},
                {"mediaType":"application/vnd.ollama.image.projector","digest":"sha256:dddpro"}
            ]}"#,
        )
        .unwrap();
        std::fs::write(root.join("blobs").join("sha256-ddd"), b"GGUF").unwrap();
        std::fs::write(root.join("blobs").join("sha256-dddpro"), b"GGUF").unwrap();

        assert!(ollama_blob_in(&root, "inkling-2-mini:chat")
            .unwrap()
            .ends_with("sha256-ddd"));
        assert!(ollama_layer_in(
            &root,
            "inkling-2-mini:chat",
            "application/vnd.ollama.image.projector"
        )
        .unwrap()
        .ends_with("sha256-dddpro"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_single_tag_answers_a_name_with_no_tag() {
        let root = store("one", &[("only-one", "35b", "ccc")]);
        assert!(ollama_blob_in(&root, "only-one")
            .unwrap()
            .ends_with("sha256-ccc"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_pulled_model_gets_a_name_worth_typing() {
        // The whole reference makes a directory no one would type after --model.
        assert_eq!(
            Hub::dir_name(
                "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
            ),
            "tinyllama-1.1b-chat-v1.0.Q2_K"
        );
        // Two quants of one repository are two models, not one.
        assert_ne!(
            Hub::dir_name("owner/repo:model.Q4_K_M.gguf"),
            Hub::dir_name("owner/repo:model.Q6_K.gguf")
        );
        assert_eq!(Hub::dir_name("owner/repo"), "repo");
        assert_eq!(Hub::dir_name("ornith-1.5:35b"), "ornith-1.5_35b");
    }

    #[test]
    fn a_hugging_face_reference_needs_a_slash() {
        assert_eq!(hf_ref("owner/repo"), Some(("owner/repo", None)));
        assert_eq!(
            hf_ref("owner/repo:file.gguf"),
            Some(("owner/repo", Some("file.gguf")))
        );
        // An Ollama name:tag is not a repository, and must never be sent as one.
        assert_eq!(hf_ref("ornith-1.5:35b"), None);
        assert_eq!(hf_ref("ornith-1.5_35b"), None);
    }
}
