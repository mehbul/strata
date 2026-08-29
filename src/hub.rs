//! Getting a model onto this machine.
//!
//! Three sources, cheapest first: a copy already under `rocm/models`, a blob
//! Ollama has downloaded, or a GGUF from a named Hugging Face repository.
//!
//! Everything here ends at `rocm/models/<name>/model.gguf`, because that is
//! where `plan`, `tune` and `serve` look for it.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};

pub struct Hub {
    /// Where models are kept. `rocm/models`, matching `placement::find`.
    pub cache_dir: PathBuf,
}

impl Hub {
    pub fn new() -> Self {
        Self { cache_dir: PathBuf::from("rocm/models") }
    }

    /// `strata pull --model <name>`.
    ///
    /// `<name>` is either an Ollama model (`ornith-1.5:35b`) or a Hugging Face
    /// repository (`owner/repo`, or `owner/repo:file.gguf` when the repository
    /// holds more than one). A bare name Ollama does not have is an error
    /// rather than a guess: inventing an owner for it only turns a typo into a
    /// 404 from a repository the caller never named.
    pub async fn pull(&self, model: &str) -> Result<PathBuf> {
        let out = self.cache_dir.join(Self::dir_name(model));
        let dest = out.join("model.gguf");

        // 1. A usable local copy means there is nothing to fetch. "Usable" is
        //    decided by reading the header, not by file size: that is true of a
        //    500 MB model and false of a truncated 30 GB download.
        if is_gguf(&dest) {
            println!("hub: {model} is already at {} ({:.1} GB)", dest.display(), file_gb(&dest));
            return Ok(out);
        }

        // 2. A blob Ollama has already downloaded, found through its manifests.
        if let Some(blob) = ollama_blob(model) {
            std::fs::create_dir_all(&out)?;
            println!("hub: Ollama has {model} ({:.1} GB)", file_gb(&blob));
            println!("     {} -> {}", blob.display(), dest.display());
            std::fs::copy(&blob, &dest)
                .with_context(|| format!("copying {} to {}", blob.display(), dest.display()))?;
            return Ok(out);
        }

        // 3. Hugging Face, for a repository named in full.
        let Some((repo, wanted)) = hf_ref(model) else {
            bail!(
                "nothing called {model} to pull.\n  \
                 Ollama does not have it, and it does not name a Hugging Face\n  \
                 repository. Pass one as owner/repo, or put a .gguf at\n  {}",
                dest.display()
            )
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
        println!("hub: pulling https://huggingface.co/{repo} -> {}", dest.display());
        self.download(&client, repo, &file, &dest).await?;
        if !is_gguf(&dest) {
            bail!("{} downloaded but does not start with the GGUF magic", dest.display());
        }
        println!("hub: {model} ready at {} ({:.1} GB)", dest.display(), file_gb(&dest));
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
    fn dir_name(model: &str) -> String {
        let name = match hf_ref(model) {
            Some((_, Some(file))) => {
                let base = file.rsplit('/').next().unwrap_or(file);
                base.strip_suffix(".gguf").or_else(|| base.strip_suffix(".GGUF")).unwrap_or(base)
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
        kept.replace(':', "_").replace('/', "_").replace("..", "_")
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
            println!("  resuming at {:.1} GB", resume_from as f32 / 1e9);
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
            tokio::fs::OpenOptions::new().append(true).open(&partial).await?
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
                match total {
                    0 => println!("  {:.1} GB", done as f32 / 1e9),
                    t => println!("  {:.1} / {:.1} GB", done as f32 / 1e9, t as f32 / 1e9),
                }
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
    let res = client.get(&url).send().await.with_context(|| format!("GET {url}"))?;
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
    let Ok(mut f) = std::fs::File::open(path) else { return false };
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).is_ok() && &magic == b"GGUF"
}

/// Where Ollama keeps its store on this machine.
fn ollama_root() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("OLLAMA_MODELS") {
        return Some(PathBuf::from(dir));
    }
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok()?;
    Some(PathBuf::from(home).join(".ollama").join("models"))
}

/// The blob Ollama holds for `name[:tag]`.
fn ollama_blob(model: &str) -> Option<PathBuf> {
    ollama_blob_in(&ollama_root()?, model)
}

/// The blob a store at `root` holds for `name[:tag]`, found through its manifests.
///
/// Ollama names blobs by their digest, so the manifest tree is the only thing
/// that says which blob is which model. Taking the largest GGUF instead - as
/// this used to - answers every pull with whatever the biggest model on the
/// machine happens to be, filed under whatever name was asked for. It also
/// cannot tell a model from the projector shipped beside it.
fn ollama_blob_in(root: &Path, model: &str) -> Option<PathBuf> {
    let (name, tag) = match model.split_once(':') {
        Some((n, t)) => (n.to_lowercase(), Some(t.to_lowercase())),
        None => (model.to_lowercase(), None),
    };
    let manifest = find_manifest(&root.join("manifests"), &name, tag.as_deref())?;
    let json: Value = serde_json::from_slice(&std::fs::read(manifest).ok()?).ok()?;
    let digest = json
        .get("layers")?
        .as_array()?
        .iter()
        .find(|l| {
            l.get("mediaType").and_then(Value::as_str) == Some("application/vnd.ollama.image.model")
        })?
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
    let entries: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).collect();
    let here = dir.file_name()?.to_string_lossy().to_lowercase();

    if here == name || name.ends_with(&format!("/{here}")) {
        let tags: Vec<&PathBuf> = entries.iter().filter(|p| p.is_file()).collect();
        let named = |t: &str| {
            tags.iter()
                .find(|p| p.file_name().is_some_and(|f| f.to_string_lossy().to_lowercase() == t))
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

    entries.iter().filter(|p| p.is_dir()).find_map(|sub| find_manifest(sub, name, tag))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An Ollama store holding the named models, each with a projector beside it.
    fn store(tag: &str, models: &[(&str, &str, &str)]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("strata-hub-{tag}"));
        let _ = std::fs::remove_dir_all(&root);
        let lib = root.join("manifests").join("registry.ollama.ai").join("library");
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
            std::fs::write(root.join("blobs").join(format!("sha256-{digest}pro")), b"GGUF").unwrap();
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
        let root = store("two", &[("small-one", "3b", "aaa"), ("big-one", "70b", "bbb")]);
        let small = ollama_blob_in(&root, "small-one:3b").unwrap();
        assert!(small.ends_with("sha256-aaa"), "{}", small.display());
        assert!(ollama_blob_in(&root, "big-one:70b").unwrap().ends_with("sha256-bbb"));
        assert!(ollama_blob_in(&root, "not-here:1b").is_none());
        assert!(ollama_blob_in(&root, "small-one:70b").is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_single_tag_answers_a_name_with_no_tag() {
        let root = store("one", &[("only-one", "35b", "ccc")]);
        assert!(ollama_blob_in(&root, "only-one").unwrap().ends_with("sha256-ccc"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_pulled_model_gets_a_name_worth_typing() {
        // The whole reference makes a directory no one would type after --model.
        assert_eq!(
            Hub::dir_name("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q2_K.gguf"),
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
        assert_eq!(hf_ref("owner/repo:file.gguf"), Some(("owner/repo", Some("file.gguf"))));
        // An Ollama name:tag is not a repository, and must never be sent as one.
        assert_eq!(hf_ref("ornith-1.5:35b"), None);
        assert_eq!(hf_ref("ornith-1.5_35b"), None);
    }
}
