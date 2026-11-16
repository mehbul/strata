//! Compute backend selection.
//!
//! llama.cpp is the universal backend. Specialized engines (Niko's Strata for
//! Qwen3.8, PolyStrata's strata-poly for the other supported architectures)
//! provide real expert caching but only for a fixed set of architectures.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    LlamaCpp,
    StrataNiko,
    StrataPoly,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::LlamaCpp => "llamacpp",
            Backend::StrataNiko => "niko",
            Backend::StrataPoly => "poly",
        }
    }

    pub fn binary(self) -> &'static str {
        match self {
            Backend::LlamaCpp => "llama-server",
            Backend::StrataNiko => "strata",       // Niko's engine binary
            Backend::StrataPoly => "strata-poly",  // PolyStrata's engine
        }
    }
}

/// Pick a backend from a GGUF architecture string.
pub fn select(arch: &str) -> Backend {
    let a = arch.to_lowercase();
    match a.as_str() {
        "qwen4exp" => Backend::StrataNiko,
        "glm5-next" | "glm5next" | "qwen35moe" | "clover_moe" | "gemma4"
        | "deepseek4" | "lfm2moe" | "flint" | "kolibri1" => Backend::StrataPoly,
        _ => Backend::LlamaCpp,
    }
}

/// Whether a backend is the universal fallback.
pub fn is_fallback(b: Backend) -> bool {
    b == Backend::LlamaCpp
}
