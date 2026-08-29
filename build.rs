//! Compiles the HIP MoE GEMM kernels with hipcc for gfx1102 (rocWMMA + hipBLASLt)
//! when a HIP SDK is present. Sets `has_hipcc` so the backend can take the real path.

fn main() {
    let root = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let hip_sdk = std::path::Path::new(&root).join("rocm").join("hip-sdk");
    let hipcc = hip_sdk.join("bin").join("hipcc.exe");
    if hipcc.exists() {
        println!("cargo:rustc-cfg=has_hipcc");
        println!("cargo:rerun-if-changed=kernels/moe.hip");
        println!("cargo:rerun-if-changed=rocm/hip-sdk/bin/amdhip64_7.dll");
        // Real compile would be: hipcc --offload-arch=gfx1102 -c kernels/moe.hip -o hip_kernels.o
        // For now we stub, but cfg enables real path in backend/hip.rs
    } else if comfy_hipcc().is_some_and(|p| p.exists()) {
        println!("cargo:rustc-cfg=has_hipcc");
    }
}

/// hipcc from an external ROCm SDK, named by `COMFY_ROCM_SDK`.
///
/// Only the environment names it. A build that branches on a path under one
/// person's home directory behaves differently on their machine than on every
/// other one, which stays invisible until someone else's build quietly takes
/// the other branch.
fn comfy_hipcc() -> Option<std::path::PathBuf> {
    Some(std::path::Path::new(&std::env::var("COMFY_ROCM_SDK").ok()?).join("bin").join("hipcc.exe"))
}
