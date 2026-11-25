//! `strata update` — pull the latest source and rebuild in place.
//!
//! Deliberately refuses to run with local edits: an update that silently
//! overwrites a user's work is worse than one that asks them to commit first.

use anyhow::{bail, Result};
use std::process::Command;

fn git(args: &[&str]) -> Result<std::process::Output> {
    let out = Command::new("git").args(args).output()?;
    Ok(out)
}

/// Write the global shim. It changes directory before running the binary, and
/// rebuilds first when any source file is newer — the same contract as the
/// project's `strata.ps1`, but usable from any directory.
pub fn write_shim() -> Result<()> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?;
    let bin = std::path::Path::new(&home).join(".strata").join("bin");
    std::fs::create_dir_all(&bin)?;
    let exe = std::env::current_exe()?;
    let project = std::env::current_dir()?;
    let lines = [
        "@echo off".to_string(),
        format!("cd /d \"{}\"", project.display()),
        "if not exist \"target\\release\\strata.exe\" cargo build --release".into(),
        "if errorlevel 1 exit /b 1".into(),
        "for /f %%i in ('powershell -NoProfile -Command \"(Get-ChildItem src -Recurse -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime -gt (Get-Item target\\release\\strata.exe).LastWriteTime\"') do if \"%%i\"==\"True\" cargo build --release".into(),
        "if errorlevel 1 exit /b 1".into(),
        format!("\"{}\" %*", exe.display()),
    ];
    std::fs::write(bin.join("strata.cmd"), lines.join("\r\n") + "\r\n")?;
    Ok(())
}

pub fn run(stash: bool) -> Result<()> {
    let status = git(&["status", "--porcelain"])?;
    if !status.status.success() {
        bail!("git status failed");
    }
    if !String::from_utf8_lossy(&status.stdout).trim().is_empty() {
        if !stash {
            bail!("local changes present. Commit or stash them, or use `strata update --stash`.");
        }
        let out = git(&["stash", "push", "-u", "strata update auto-stash"])?;
        if !out.status.success() {
            bail!(
                "git stash failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        println!("Auto-stashed local changes.");
    }

    println!("Fetching updates...");
    let out = git(&["pull", "--ff-only"])?;
    if !out.status.success() {
        bail!(
            "git pull failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    println!("{}", String::from_utf8_lossy(&out.stdout).trim());

    println!("Rebuilding the engine...");
    let out = Command::new("cargo")
        .args(["build", "--release"])
        .status()?;
    if !out.success() {
        bail!("cargo build --release failed");
    }

    if !std::path::Path::new("web/dist/index.html").exists() {
        println!("Building the console...");
        let out = Command::new("npm")
            .args(["run", "build"])
            .current_dir("web")
            .status()?;
        if !out.success() {
            bail!("npm run build failed");
        }
    }

    write_shim()?;
    println!("Refreshed the strata command shim.");
    println!("Strata is up to date.");
    Ok(())
}
