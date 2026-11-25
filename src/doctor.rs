//! `strata doctor` — a local health check that does not need a running server.
//!
//! The recurring failures on this project are environmental: a missing backend
//! library makes the GPU disappear from `--list-devices`, a stale binary makes
//! a new feature appear absent, and a missing console bundle makes the page
//! blank. Those are cheap to check and expensive to diagnose by eye.

use anyhow::Result;

fn mark(ok: bool) -> &'static str {
    if ok {
        "OK  "
    } else {
        "FAIL"
    }
}

pub async fn run(host: &str) -> Result<()> {
    println!("Strata doctor\n");

    // Binary and build freshness.
    let exe = std::env::current_exe()?;
    let newest_source = ["src", "Cargo.toml", "build.rs"]
        .iter()
        .filter_map(|p| {
            let path = std::path::Path::new(p);
            let mut newest = std::fs::metadata(path).ok()?.modified().ok()?;
            if path.is_dir() {
                for entry in walk(path) {
                    if let Ok(t) = entry.metadata().and_then(|m| m.modified()) {
                        if t > newest {
                            newest = t;
                        }
                    }
                }
            }
            Some(newest)
        })
        .max();
    let built = std::fs::metadata(&exe).and_then(|m| m.modified()).ok();
    let stale = match (newest_source, built) {
        (Some(s), Some(b)) => s > b,
        _ => false,
    };
    println!(
        "{} binary   {}",
        mark(!stale),
        if stale {
            "source is newer than the binary — cargo build --release"
        } else {
            "up to date"
        }
    );

    // Runtime and GPU. A missing ROCm DLL produces an empty device list with
    // no error in the log, so this has to be checked explicitly.
    let runtime = std::path::Path::new("runtime").join("llama-server.exe");
    let has_runtime = runtime.exists();
    println!("{} runtime  {}", mark(has_runtime), runtime.display());
    if has_runtime {
        match std::process::Command::new(&runtime)
            .arg("--list-devices")
            .output()
        {
            Ok(out) => {
                let text = String::from_utf8_lossy(&out.stdout);
                let devices: Vec<&str> = text
                    .lines()
                    .filter(|l| l.contains("ROCm") || l.contains("Vulkan") || l.contains("CUDA"))
                    .collect();
                let shown = devices.join(" | ");
                println!(
                    "{} gpu      {}",
                    mark(!devices.is_empty()),
                    if devices.is_empty() {
                        "no GPU listed — check ROCm DLLs on PATH (run strata.ps1)"
                    } else {
                        &shown
                    }
                );
            }
            Err(e) => println!("{} gpu      could not run --list-devices: {e}", mark(false)),
        }
    }

    // Model store and console.
    let models = crate::registry::scan(&crate::registry::default_root());
    println!(
        "{} models   {} installed in {}",
        mark(!models.is_empty()),
        models.len(),
        crate::registry::default_root().display()
    );
    let console = std::path::Path::new("web").join("dist").join("index.html");
    println!(
        "{} console  {}",
        mark(console.exists()),
        if console.exists() {
            "built"
        } else {
            "missing — npm run build (in web/)"
        }
    );

    // Registry token, only relevant for push.
    let token = crate::auth::token();
    println!(
        "{} registry {}",
        if token.is_some() { "OK  " } else { "--  " },
        if token.is_some() {
            "token available"
        } else {
            "no token (pull and local use need none)"
        }
    );

    // Running server, if any.
    let client = reqwest::Client::new();
    let url = format!("{}/api/version", host.trim_end_matches('/'));
    match client
        .get(&url)
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            let body: serde_json::Value = resp.json().await.unwrap_or_default();
            println!(
                "{} server   {} at {host}",
                mark(true),
                body["version"].as_str().unwrap_or("?")
            );
        }
        _ => println!("-- server   not reachable at {host} (fine unless you need the API)"),
    }
    Ok(())
}

fn walk(dir: &std::path::Path) -> Vec<std::fs::DirEntry> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in rd.flatten() {
        if entry.path().is_dir() {
            out.extend(walk(&entry.path()));
        } else {
            out.push(entry);
        }
    }
    out
}
