//! What this machine actually has, asked of the thing that will use it.
//!
//! The compute runtime already enumerates every device it can drive, across
//! whichever backend it was built with, and reports the memory it can see:
//!
//! ```text
//! Available devices:
//!   ROCm0: AMD Radeon RX 7600 XT (16368 MiB, 16224 MiB free)
//!   ROCm1: AMD Radeon(TM) Graphics (12472 MiB, 12189 MiB free)
//! ```
//!
//! That is the right source. It needs no vendor SDK, no Python and no PyTorch;
//! it answers for ROCm, Vulkan and CUDA alike; and it is the same view of the
//! hardware the process doing the work will have, rather than a second opinion
//! that might disagree with it.
//!
//! Nothing here guesses. A probe that cannot see a device says so, because the
//! alternative - a plausible default - produces a placement plan for a machine
//! that does not exist, and the failure surfaces much later as an unexplained
//! out-of-memory during load.

use anyhow::{bail, Context, Result};
use std::process::Command;

use crate::runner::Runtime;

/// One device the compute runtime can drive.
#[derive(Debug, Clone)]
pub struct Device {
    /// Backend and ordinal as the runtime names it, e.g. `ROCm0`.
    pub id: String,
    pub name: String,
    pub total_mib: u64,
    pub free_mib: u64,
}

impl Device {
    pub fn total_gb(&self) -> f32 {
        self.total_mib as f32 / 1024.0
    }

    /// An integrated GPU reports a slice of system RAM as if it were VRAM, and
    /// treating that as a memory budget double-counts against the host. The
    /// name is the only signal the runtime gives us, so it is what we use.
    pub fn looks_integrated(&self) -> bool {
        let n = self.name.to_ascii_lowercase();
        n.contains("(tm) graphics")
            || n.contains("integrated")
            || n.contains("igpu")
            || n.ends_with(" graphics")
    }
}

#[derive(Debug, Clone)]
pub struct Machine {
    pub devices: Vec<Device>,
    pub ram_total_gb: f32,
    pub ram_avail_gb: f32,
    pub cores: usize,
}

impl Machine {
    /// The device a model should be planned against: the largest discrete one,
    /// falling back to the largest of any kind if every device looks
    /// integrated - a machine with only an APU can still run, just slowly.
    pub fn primary(&self) -> Option<&Device> {
        let discrete = self.devices.iter().filter(|d| !d.looks_integrated());
        discrete
            .max_by_key(|d| d.total_mib)
            .or_else(|| self.devices.iter().max_by_key(|d| d.total_mib))
    }

    /// Host memory that may hold expert weights, leaving room for the operating
    /// system and whatever else the machine is doing. Reported free memory is
    /// used rather than total: a machine with 32 GB installed and 20 GB in use
    /// cannot lend 20 GB to the model.
    pub fn ram_budget_gb(&self) -> f32 {
        (self.ram_avail_gb - 4.0).max(0.0)
    }
}

/// Ask the compute runtime what it can see.
pub fn probe(runtime: &Runtime) -> Result<Machine> {
    let mut cmd = Command::new(&runtime.server);
    cmd.arg("--list-devices");
    // The backend libraries live beside the server binary and in a per-GPU
    // subdirectory; without both on the loader path the backend fails to load
    // and the device list comes back empty with nothing explaining why.
    if let Some(path) = crate::runner::loader_path(runtime) {
        cmd.env("PATH", path);
    }
    if let Some(backend) = &runtime.backend {
        cmd.env("GGML_BACKEND_PATH", backend);
    }
    let out = cmd
        .output()
        .with_context(|| format!("running {} --list-devices", runtime.server.display()))?;

    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let devices = parse_devices(&text);
    if devices.is_empty() {
        bail!(
            "the compute runtime reports no usable GPU. This is what a missing backend \
             library looks like: llama.cpp links rocBLAS/hipBLAS but does not ship them, \
             and without one on the loader path the device list is simply empty.\n\
             Check by hand:\n  {} --list-devices",
            runtime.server.display()
        );
    }

    let (ram_total_gb, ram_avail_gb) = crate::server::host_ram_gb();
    Ok(Machine {
        devices,
        ram_total_gb,
        ram_avail_gb,
        cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
    })
}

/// Pull devices out of `--list-devices` output.
///
/// Lines look like `  ROCm0: AMD Radeon RX 7600 XT (16368 MiB, 16224 MiB free)`.
/// Anything that does not match that shape is ignored rather than guessed at,
/// so a future runtime that changes the format degrades to "no devices found"
/// - which is checked and reported - instead of to a wrong number.
fn parse_devices(text: &str) -> Vec<Device> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some((id, rest)) = line.split_once(": ") else { continue };
        let id = id.trim();
        if id.is_empty() || id.contains(' ') {
            continue;
        }
        let Some(open) = rest.rfind('(') else { continue };
        let Some(close) = rest[open..].find(')') else { continue };
        let name = rest[..open].trim().to_string();
        let inside = &rest[open + 1..open + close];
        let mut sizes = inside.split(',').filter_map(|part| {
            part.trim().split_whitespace().next().and_then(|n| n.parse::<u64>().ok())
        });
        let (Some(total_mib), Some(free_mib)) = (sizes.next(), sizes.next()) else { continue };
        if name.is_empty() || total_mib == 0 {
            continue;
        }
        out.push(Device { id: id.to_string(), name, total_mib, free_mib });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "Available devices:\n  \
        ROCm0: AMD Radeon RX 7600 XT (16368 MiB, 16224 MiB free)\n  \
        ROCm1: AMD Radeon(TM) Graphics (12472 MiB, 12189 MiB free)\n";

    fn machine(devices: Vec<Device>) -> Machine {
        Machine { devices, ram_total_gb: 32.0, ram_avail_gb: 20.0, cores: 12 }
    }

    #[test]
    fn parses_the_runtime_device_list() {
        let d = parse_devices(SAMPLE);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].id, "ROCm0");
        assert_eq!(d[0].name, "AMD Radeon RX 7600 XT");
        assert_eq!(d[0].total_mib, 16368);
        assert_eq!(d[0].free_mib, 16224);
    }

    #[test]
    fn prefers_the_discrete_card_over_the_apu() {
        // The integrated device reports 12 GB of borrowed system memory; using
        // it as a VRAM budget would double-count against the host.
        let m = machine(parse_devices(SAMPLE));
        assert_eq!(m.primary().unwrap().name, "AMD Radeon RX 7600 XT");
    }

    #[test]
    fn falls_back_to_an_apu_when_that_is_all_there_is() {
        let only_igpu = "  Vulkan0: AMD Radeon(TM) Graphics (12472 MiB, 12189 MiB free)";
        let m = machine(parse_devices(only_igpu));
        assert_eq!(m.primary().unwrap().total_mib, 12472);
    }

    #[test]
    fn ignores_lines_it_does_not_understand() {
        assert!(parse_devices("Available devices:\n  none\n").is_empty());
        assert!(parse_devices("ggml_cuda_init: found 1 device\n").is_empty());
        // A changed format must yield nothing rather than a plausible number.
        assert!(parse_devices("  ROCm0: Some Card [16368 MB]").is_empty());
    }

    #[test]
    fn ram_budget_holds_back_room_for_the_system() {
        assert_eq!(machine(vec![]).ram_budget_gb(), 16.0);
        let tight = Machine { ram_avail_gb: 2.0, ..machine(vec![]) };
        assert_eq!(tight.ram_budget_gb(), 0.0);
    }
}
