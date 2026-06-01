//! Error type for ermak. The memory-budget variants make refusing an
//! oversized allocation an explicit, recoverable error rather than an OOM kill.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErmakError {
    /// A requested allocation exceeded the configured memory budget. Returned
    /// instead of allocating, so an over-large request cannot OOM the machine.
    MemoryBudgetExceeded {
        what: &'static str,
        requested_bytes: usize,
        cap_bytes: usize,
    },
    /// The GPU backend was requested but is unavailable or failed.
    Gpu(String),
}

impl fmt::Display for ErmakError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErmakError::MemoryBudgetExceeded {
                what,
                requested_bytes,
                cap_bytes,
            } => write!(
                f,
                "{what}: requested {requested_bytes} bytes exceeds the {cap_bytes}-byte memory budget (raise ERMAK_MAX_BYTES or reduce the workload)"
            ),
            ErmakError::Gpu(msg) => write!(f, "gpu backend: {msg}"),
        }
    }
}

impl std::error::Error for ErmakError {}
