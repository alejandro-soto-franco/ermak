//! GPU backend guardrails (feature `gpu`).
//!
//! This GPU has only ~8 GiB of VRAM, so the device-side guardrail is essential:
//! a GPU ensemble batch is sized to a fraction of *free* VRAM (queried from
//! `nvidia-smi`), and the same [`crate::memory::MemoryBudget`] / batching
//! machinery streams the ensemble so the device is never over-committed.
//!
//! The on-device kernel itself (cuda-oxide or a Vulkan compute path) is the next
//! increment; this module lands the device-memory guardrail it will run under.

use crate::error::ErmakError;
use crate::memory::MemoryBudget;
use std::process::Command;

/// Parse free VRAM in MiB from `nvidia-smi` output (the first integer found,
/// tolerating units and trailing whitespace).
#[must_use]
pub fn parse_free_vram_mib(out: &str) -> Option<usize> {
    out.split(|c: char| !c.is_ascii_digit())
        .find(|s| !s.is_empty())
        .and_then(|s| s.parse().ok())
}

/// Query free VRAM via `nvidia-smi`.
///
/// # Errors
/// [`ErmakError::Gpu`] if `nvidia-smi` is missing or its output is unparseable.
pub fn free_vram_bytes() -> Result<usize, ErmakError> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|e| ErmakError::Gpu(format!("nvidia-smi failed to launch: {e}")))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mib = parse_free_vram_mib(&text)
        .ok_or_else(|| ErmakError::Gpu(format!("could not parse free VRAM from: {text:?}")))?;
    Ok(mib * 1024 * 1024)
}

/// A device-memory budget capped at `fraction` of free VRAM, so a GPU batch can
/// never claim all of the 8 GiB device. `fraction` is clamped to `(0, 1]`.
///
/// # Errors
/// Propagates [`free_vram_bytes`] errors.
pub fn device_budget(fraction: f64) -> Result<MemoryBudget, ErmakError> {
    let frac = fraction.clamp(f64::MIN_POSITIVE, 1.0);
    let free = free_vram_bytes()?;
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let cap = (free as f64 * frac) as usize;
    Ok(MemoryBudget::new(cap, "device VRAM"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_free_vram_with_or_without_units() {
        assert_eq!(parse_free_vram_mib("7636\n"), Some(7636));
        assert_eq!(parse_free_vram_mib(" 7636 MiB\n"), Some(7636));
        assert_eq!(parse_free_vram_mib("7636, 8151"), Some(7636));
        assert_eq!(parse_free_vram_mib("garbage"), None);
        assert_eq!(parse_free_vram_mib(""), None);
    }
}
