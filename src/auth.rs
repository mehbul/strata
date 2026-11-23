//! Registry credentials, kept in `~/.strata` and never in the model store.
//!
//! Strata does not phone home. `signin` stores a token a user already has; it
//! does not implement Ollama's device-code dance, because that requires an
//! account and a browser flow this local engine has no reason to own.

use anyhow::{Context, Result};
use std::io::{self, BufRead};
use std::path::PathBuf;

fn config_dir() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".strata")
}

pub fn token_path() -> PathBuf {
    config_dir().join("token")
}

pub fn registry_path() -> PathBuf {
    config_dir().join("registry")
}

pub fn token() -> Option<String> {
    std::env::var("STRATA_API_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            std::fs::read_to_string(token_path())
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
}

pub fn registry_url() -> String {
    std::env::var("STRATA_REGISTRY")
        .or_else(|_| std::env::var("OLLAMA_REGISTRY"))
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::fs::read_to_string(registry_path()).ok().map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://registry.ollama.ai".into())
}

pub fn keys_path() -> PathBuf {
    config_dir().join("keys.json")
}

pub fn load_keys() -> Vec<String> {
    std::fs::read_to_string(keys_path())
        .ok()
        .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
        .unwrap_or_default()
}

pub fn save_keys(keys: &[String]) -> Result<()> {
    std::fs::create_dir_all(config_dir())?;
    std::fs::write(keys_path(), serde_json::to_string_pretty(keys)?)?;
    Ok(())
}

pub fn valid_key(presented: &str) -> bool {
    load_keys().iter().any(|k| k == presented)
}

pub fn signin(token: Option<&str>, registry: Option<&str>) -> Result<()> {
    std::fs::create_dir_all(config_dir())?;
    let value = match token {
        Some(t) if !t.trim().is_empty() => t.trim().to_string(),
        _ => {
            println!("Paste a registry token (it is stored in {}):", token_path().display());
            let mut line = String::new();
            io::stdin()
                .lock()
                .read_line(&mut line)
                .context("reading token")?;
            line.trim().to_string()
        }
    };
    anyhow::ensure!(!value.is_empty(), "empty token is not a sign-in");
    std::fs::write(token_path(), &value)?;
    if let Some(r) = registry {
        std::fs::write(registry_path(), r.trim())?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(token_path(), std::fs::Permissions::from_mode(0o600)).ok();
    }
    println!("Signed in. Token stored at {}", token_path().display());
    Ok(())
}

pub fn signout() -> Result<()> {
    for path in [token_path(), registry_path()] {
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
    }
    println!("Signed out.");
    Ok(())
}
