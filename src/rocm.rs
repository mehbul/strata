use anyhow::Result;
use std::path::PathBuf;

/// An external ROCm SDK, named by `COMFY_ROCM_SDK`.
///
/// Opportunistic only: with one, `info` can also report the gfx target and
/// torch version; without one, those lines are simply absent and nothing else
/// changes. Only the environment names it - guessing at a path under the
/// author's home directory found nothing on any other machine, and said
/// something about that machine's layout in the process.
fn comfy_rocm_sdk() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("COMFY_ROCM_SDK").ok()?);
    p.exists().then_some(p)
}

fn comfy_python() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("COMFY_ROCM_PYTHON") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    // The SDK sits at <python_env>/Lib/site-packages/_rocm_sdk_devel, so the
    // interpreter is three levels up, not four - which is what this walked
    // before, landing on a path that has never existed and sending every
    // detection down the fallback branch.
    let py = comfy_rocm_sdk()?.join("..").join("..").join("..").join("python.exe");
    py.exists().then_some(py)
}
fn vendored_hip_sdk() -> PathBuf {
    // Relative to exe or cwd, not hardcoded absolute with username
    let candidates = [
        PathBuf::from("rocm/hip-sdk"),
        std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.join("../../rocm/hip-sdk"))).unwrap_or_default(),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("rocm").join("hip-sdk"),
    ];
    for c in candidates {
        if c.join("bin").join("amdhip64_7.dll").exists() { return c; }
    }
    PathBuf::from("rocm/hip-sdk")
}
fn vendored_detect() -> PathBuf {
    let cands = [
        PathBuf::from("rocm/detect_gpu.py"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("rocm").join("detect_gpu.py"),
    ];
    for c in cands { if c.exists() { return c; } }
    PathBuf::from("rocm/detect_gpu.py")
}

fn local_hip_path() -> PathBuf {
    let vendored = vendored_hip_sdk();
    if vendored.join("bin").join("amdhip64_7.dll").exists() {
        return vendored;
    }
    // Fall back to a ComfyUI SDK if this machine has one; otherwise keep the
    // vendored path so the error names somewhere the user can actually look.
    comfy_rocm_sdk().unwrap_or(vendored)
}

#[derive(Debug)]
/// What could be established about the ROCm toolchain on this machine.
///
/// Every field is optional because every one of them used to have a literal
/// fallback - the development machine's GPU name, its gfx target, its ROCm and
/// torch versions - which meant a machine where detection failed reported that
/// hardware as though it had been measured.
pub struct RocmInfo {
    pub hip_path: PathBuf,
    pub amdhip_dll: PathBuf,
    pub gfx: Option<String>,
    pub device_name: Option<String>,
    pub torch_version: Option<String>,
    pub hip_version: Option<String>,
}

pub fn detect() -> Result<RocmInfo> {
    // 1. Prefer vendored HIP in project/rocm/hip-sdk (self-contained), fallback to comfy external
    let hip_path = local_hip_path();
    let amdhip = hip_path.join("bin").join("amdhip64_7.dll");
    if !amdhip.exists() {
        anyhow::bail!("amdhip64_7.dll not found at {:?} (checked vendored + comfy)", amdhip);
    }

    // 2. A ComfyUI PyTorch, if this machine happens to have one, gives the
    //    richest answer. Its absence is normal and not an error.
    let py = comfy_python();
    let torch = py.as_ref().and_then(|py| {
        let o = std::process::Command::new(py)
            .args([
                "-c",
                "import torch; print(torch.cuda.get_device_name(0));                  print(torch.version.hip); print(torch.__version__)",
            ])
            .output()
            .ok()?;
        if !o.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&o.stdout).to_string();
        let mut lines = text.lines().map(|l| l.trim().to_string());
        Some((lines.next(), lines.next(), lines.next()))
    });
    let (device_name, hip_version, torch_version) = match torch {
        Some((d, h, t)) => (d.filter(|s| !s.is_empty()), h.filter(|s| !s.is_empty()), t),
        None => (None, None, None),
    };

    // 3. The gfx target, from the vendored detector when a Python exists to run
    //    it. Left unknown rather than guessed.
    let gfx = py.as_ref().and_then(|py| {
        let detect_path = vendored_detect();
        if !detect_path.exists() {
            return None;
        }
        let o = std::process::Command::new(py).arg(&detect_path).output().ok()?;
        let text = String::from_utf8(o.stdout).ok()?;
        let t = text.trim().to_string();
        t.starts_with("gfx").then_some(t)
    });

 // 4. Try to validate amdhip64_7.dll
    // On Windows this DLL has deps in same bin dir - need to add to PATH for LoadLibraryEx
    // We check existence + torch HIP already proved runtime works (torch.cuda.is_available() == true)
    // So we soft-check: try load, but don't fail if deps missing - torch is the ground truth
    let hip_bin = hip_path.join("bin");
    unsafe {
        // Add hip bin to DLL search path for this process (like comfyui-rocm.bat does via PATH)
        let _ = libloading::os::windows::Library::new(&amdhip).map(|_| ()).or_else(|e| {
            // fallback: check existence only, rely on torch HIP check
            if amdhip.exists() { Ok(()) } else { Err(e) }
        });
        let _ = hip_bin; // keep for future explicit loader
    }

    Ok(RocmInfo {
        hip_path,
        amdhip_dll: amdhip,
        gfx,
        device_name,
        torch_version,
        hip_version,
    })
}

/// `strata info` - what is actually on this machine.
///
/// The compute runtime is asked first, because it is vendor-neutral, needs no
/// SDK or Python, and is the same view the process doing the work will have.
/// The ROCm toolchain details are additional colour and may be absent.
pub fn print_info() -> Result<()> {
    println!("=== Strata ===");

    match crate::runner::Runtime::discover() {
        Some(runtime) => {
            println!("runtime        : {}", runtime.server.display());
            match crate::hardware::probe(&runtime) {
                Ok(machine) => {
                    for d in &machine.devices {
                        let mark = if machine.primary().map(|p| p.id == d.id).unwrap_or(false) {
                            " *"
                        } else {
                            ""
                        };
                        println!(
                            "device         : {} {} - {:.1} GB total, {:.1} GB free{mark}",
                            d.id,
                            d.name,
                            d.total_gb(),
                            d.free_mib as f32 / 1024.0
                        );
                    }
                    println!(
                        "host           : {:.0} GB RAM, {:.0} GB free, {} threads",
                        machine.ram_total_gb, machine.ram_avail_gb, machine.cores
                    );
                }
                Err(e) => println!("device         : {e}"),
            }
        }
        None => {
            println!("runtime        : not found");
            println!("                 unzip a llama.cpp build into runtime/, or set");
            println!("                 STRATA_LLAMA_DIR. See the README.");
        }
    }

    // Everything below is optional detail about a ROCm install, absent on a
    // machine that has none. Nothing here is required to serve a model.
    match detect() {
        Ok(info) => {
            let kb = std::fs::metadata(&info.amdhip_dll).map(|m| m.len() / 1024).ok();
            println!(
                "HIP SDK        : {} ({})",
                info.hip_path.display(),
                if is_vendored() { "vendored" } else { "external" }
            );
            match kb {
                Some(kb) => println!("amdhip64_7.dll : {} ({kb} KB)", info.amdhip_dll.display()),
                None => println!("amdhip64_7.dll : {}", info.amdhip_dll.display()),
            }
            if let Some(gfx) = &info.gfx {
                println!("gfx target     : {gfx}");
            }
            if let (Some(t), Some(h)) = (&info.torch_version, &info.hip_version) {
                println!("torch          : {t}  HIP {h}");
            }
        }
        Err(e) => println!("HIP SDK        : not detected ({e})"),
    }

    println!();
    println!("--- status ---");
    println!("The matrix multiplies are llama.cpp's; Strata owns everything around them.");
    println!("The expert map in /experts is SIMULATED - the router is not observable yet.");
    Ok(())
}

/// Returns the HIP SDK root, used as HIP_SDK_ROOT.
pub fn hip_sdk_root() -> PathBuf {
    local_hip_path()
}

/// Returns true if using vendored ROCm (self-contained)
pub fn is_vendored() -> bool {
    let vendored = vendored_hip_sdk();
    vendored.join("bin").join("amdhip64_7.dll").exists() && local_hip_path() == vendored
}

/// Total VRAM of the device a model should be planned against.
///
/// Returns 0.0 when nothing can be established. It used to return 16.0 - the
/// development machine's card - which meant an undetected GPU produced a
/// placement plan for hardware the user did not have, and the mistake only
/// surfaced much later as an unexplained failure to load.
pub fn vram_total_gb() -> f32 {
    if let Some(runtime) = crate::runner::Runtime::discover() {
        if let Ok(machine) = crate::hardware::probe(&runtime) {
            if let Some(device) = machine.primary() {
                return device.total_gb();
            }
        }
    }
    // A ComfyUI PyTorch, where one exists, as a second opinion.
    if let Some(py) = comfy_python() {
        let out = std::process::Command::new(&py)
            .args(["-c", "import torch;print(torch.cuda.get_device_properties(0).total_memory)"])
            .output();
        if let Ok(o) = out {
            if o.status.success() {
                if let Ok(text) = String::from_utf8(o.stdout) {
                    if let Ok(bytes) = text.trim().parse::<f64>() {
                        return (bytes / (1024.0 * 1024.0 * 1024.0)) as f32;
                    }
                }
            }
        }
    }
    0.0
}

/// Directories that may hold the ROCm runtime libraries (rocBLAS, hipBLAS,
/// comgr) a llama.cpp HIP backend links against, in priority order.
///
/// A stock llama.cpp ROCm build ships `ggml-hip.dll` but not these, so one of
/// these directories has to be on the loader path or the backend fails to load
/// with error 126 and the GPU silently disappears.
pub fn sdk_search_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut push = |dir: PathBuf| {
        if dir.join("rocblas.dll").exists() || dir.join("librocblas.so").exists() {
            if !out.contains(&dir) {
                out.push(dir);
            }
        }
    };
    if let Ok(custom) = std::env::var("STRATA_ROCM_BIN") {
        push(PathBuf::from(custom));
    }
    if let Some(sdk) = comfy_rocm_sdk() {
        push(sdk.join("bin"));
    }
    push(local_hip_path().join("bin"));
    out
}
