//! Background server control: `strata daemon start|stop|status`.
//!
//! Ollama keeps a daemon so a CLI command never has to wait for weights. Strata
//! can do the same without a service install: a detached process, a PID file
//! and a log under `~/.strata`. `taskkill /T` on Windows takes the compute
//! child with it, so stopping does not leave an orphan holding the GPU.

use anyhow::{bail, Context, Result};
use std::process::{Command, Stdio};

fn dir() -> Result<std::path::PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .context("no home directory")?;
    let dir = std::path::Path::new(&home).join(".strata");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn pid_path() -> Result<std::path::PathBuf> { Ok(dir()?.join("daemon.pid")) }
fn log_path() -> Result<std::path::PathBuf> { Ok(dir()?.join("daemon.log")) }

fn read_pid() -> Option<u32> {
    std::fs::read_to_string(pid_path().ok()?)
        .ok()?
        .trim()
        .parse()
        .ok()
}

fn alive(pid: u32) -> bool {
    #[cfg(windows)]
    {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}")])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
            .unwrap_or(false)
    }
    #[cfg(not(windows))]
    {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}

async fn healthy(listen: &str) -> bool {
    let url = format!("http://{listen}/api/version");
    reqwest::Client::new()
        .get(url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

pub async fn start(listen: &str, model: &str, ctx: usize) -> Result<()> {
    if let Some(pid) = read_pid() {
        if alive(pid) && healthy(listen).await {
            println!("Strata is already running (PID {pid}) on http://{listen}");
            return Ok(());
        }
    }
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path()?)?;
    let exe = std::env::current_exe()?;
    let child = Command::new(exe)
        .args([
            "serve",
            "--model", model,
            "--ctx", &ctx.to_string(),
            "--listen", listen,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .spawn()
        .context("could not start the server")?;
    std::fs::write(pid_path()?, child.id().to_string())?;
    println!("Starting Strata (PID {}) — log at {}", child.id(), log_path()?.display());

    for _ in 0..120 {
        if healthy(listen).await {
            println!("Ready at http://{listen}");
            return Ok(());
        }
        if !alive(child.id()) {
            bail!("server exited during startup; see {}", log_path()?.display());
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
    bail!("server did not become ready in 120s; see {}", log_path()?.display())
}

pub fn stop(listen: &str) -> Result<()> {
    let Some(pid) = read_pid() else {
        println!("No daemon PID file; nothing to stop.");
        return Ok(());
    };
    if !alive(pid) {
        std::fs::remove_file(pid_path()?).ok();
        println!("Daemon was not running.");
        return Ok(());
    }
    #[cfg(windows)]
    {
        let out = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .output()?;
        if !out.status.success() {
            bail!("taskkill failed: {}", String::from_utf8_lossy(&out.stderr).trim());
        }
    }
    #[cfg(not(windows))]
    {
        Command::new("kill").arg(pid.to_string()).status()?;
    }
    std::fs::remove_file(pid_path()?).ok();
    let _ = listen;
    println!("Stopped Strata (PID {pid}).");
    Ok(())
}

pub async fn status(listen: &str) -> Result<()> {
    match read_pid() {
        Some(pid) => println!(
            "{} PID {pid}",
            if alive(pid) { "running" } else { "stale PID file" }
        ),
        None => println!("not started"),
    }
    println!(
        "server: {}",
        if healthy(listen).await { "responding" } else { "not responding" }
    );
    println!("log: {}", log_path()?.display());
    Ok(())
}
