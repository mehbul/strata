//! Push a local model to a Docker Registry v2 endpoint.
//!
//! This is the protocol Ollama's library speaks. Strata does not bundle
//! credentials, so an explicit registry and token are required; the token can
//! come from `--token`, `STRATA_API_KEY`, or `strata signin`. Pushing without
//! one would be a network request the user did not authorize.

use anyhow::{bail, Context, Result};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};

async fn upload_blob(
    client: &reqwest::Client,
    registry: &str,
    repo: &str,
    token: &str,
    path: &std::path::Path,
) -> Result<String> {
    let start = format!(
        "{}/v2/{}/blobs/uploads/",
        registry.trim_end_matches('/'),
        repo
    );
    let resp = client
        .post(&start)
        .header(AUTHORIZATION, format!("Bearer {token}"))
        .send()
        .await
        .with_context(|| format!("POST {start}"))?;
    if !resp.status().is_success() {
        bail!("blob upload start returned {}", resp.status());
    }
    let location = resp
        .headers()
        .get("Location")
        .and_then(|v| v.to_str().ok())
        .context("registry did not return an upload Location")?
        .to_string();
    let location = if location.starts_with("http") {
        location
    } else {
        format!("{}{}", registry.trim_end_matches('/'), location)
    };

    let digest = crate::blob::sha256_hex(std::fs::File::open(path)?)?;
    let size = std::fs::metadata(path)?.len();
    let url = format!(
        "{}?digest=sha256:{}",
        location.split('?').next().unwrap_or(&location),
        digest
    );
    let file = tokio::fs::File::open(path).await?;
    let stream = async_stream::stream! {
        let mut file = file;
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = tokio::io::AsyncReadExt::read(&mut file, &mut buf).await.unwrap_or(0);
            if n == 0 { break }
            yield Ok::<_, std::io::Error>(bytes::Bytes::copy_from_slice(&buf[..n]));
        }
    };
    let resp = client
        .put(&url)
        .header(AUTHORIZATION, format!("Bearer {token}"))
        .header(CONTENT_TYPE, "application/octet-stream")
        .header("Content-Length", size)
        .body(reqwest::Body::wrap_stream(stream))
        .send()
        .await
        .with_context(|| format!("PUT {url}"))?;
    if !resp.status().is_success() {
        bail!("blob upload returned {}", resp.status());
    }
    Ok(format!("sha256:{digest}"))
}

/// `strata push <name>` / `POST /api/push`.
pub async fn push(
    name: &str,
    registry: Option<&str>,
    token: Option<&str>,
    insecure: bool,
) -> Result<()> {
    let registry = registry
        .map(str::to_string)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(crate::auth::registry_url);
    let token = token
        .map(str::to_string)
        .filter(|s| !s.trim().is_empty())
        .or_else(crate::auth::token)
        .context("no registry token: run `strata signin --token <token>` or set STRATA_API_KEY")?;
    let root = crate::registry::default_root();
    let dir = std::path::PathBuf::from(&root).join(crate::registry::safe_name(name)?);
    let gguf = dir.join("model.gguf");
    anyhow::ensure!(gguf.exists(), "model '{name}' has no model.gguf");

    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(insecure)
        .build()?;
    let (repo, tag) = match name.rsplit_once(':') {
        Some((r, t)) => (r.to_string(), t.to_string()),
        None => (name.to_string(), "latest".into()),
    };

    println!("Pushing {name} to {registry}/{repo}:{tag}");
    let config = json!({});
    let config_path =
        std::env::temp_dir().join(format!("strata-config-{}.json", std::process::id()));
    std::fs::write(&config_path, config.to_string())?;
    let config_digest = upload_blob(&client, &registry, &repo, &token, &config_path).await?;
    std::fs::remove_file(&config_path).ok();
    println!("  uploaded config layer {config_digest}");
    let model_digest = upload_blob(&client, &registry, &repo, &token, &gguf).await?;
    println!("  uploaded model layer {model_digest}");

    // Derived models keep their Modelfile overrides in strata.json. Push them
    // as ordinary layers so a pull on another machine gets the same system
    // prompt and parameters, not just the weights.
    let mut extra_layers = Vec::new();
    let config_path = dir.join("strata.json");
    if let Ok(text) = std::fs::read_to_string(&config_path) {
        if let Ok(overrides) = serde_json::from_str::<serde_json::Value>(&text) {
            for key in ["system", "template", "parameters", "license"] {
                let value = match &overrides[key] {
                    Value::String(s) => Some(s.clone()),
                    other @ (Value::Object(_) | Value::Array(_)) => Some(other.to_string()),
                    _ => None,
                };
                if let Some(content) = value {
                    if content.is_empty() {
                        continue;
                    }
                    let tmp = std::env::temp_dir()
                        .join(format!("strata-{}-{key}.json", std::process::id()));
                    std::fs::write(&tmp, content)?;
                    let digest = upload_blob(&client, &registry, &repo, &token, &tmp).await?;
                    println!("  uploaded {key} layer {digest}");
                    std::fs::remove_file(&tmp).ok();
                    extra_layers.push(json!({
                        "mediaType": format!("application/vnd.ollama.image.{key}"),
                        "size": std::fs::metadata(&config_path).map(|m| m.len()).unwrap_or(0),
                        "digest": digest,
                    }));
                }
            }
        }
    }

    let mut manifest: Value = json!({
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {
            "mediaType": "application/vnd.docker.container.image.v1+json",
            "size": config.to_string().len(),
            "digest": config_digest,
        },
        "layers": [{
            "mediaType": "application/vnd.ollama.image.model",
            "size": std::fs::metadata(&gguf)?.len(),
            "digest": model_digest,
        }],
    });
    if !extra_layers.is_empty() {
        if let Some(layers) = manifest.get_mut("layers").and_then(Value::as_array_mut) {
            layers.extend(extra_layers);
        }
    }
    let url = format!(
        "{}/v2/{}/manifests/{tag}",
        registry.trim_end_matches('/'),
        repo
    );
    let resp = client
        .put(&url)
        .header(AUTHORIZATION, format!("Bearer {token}"))
        .header(
            CONTENT_TYPE,
            "application/vnd.docker.distribution.manifest.v2+json",
        )
        .json(&manifest)
        .send()
        .await
        .with_context(|| format!("PUT {url}"))?;
    if !resp.status().is_success() {
        bail!("manifest upload returned {}", resp.status());
    }
    println!("Pushed {name}");
    Ok(())
}
