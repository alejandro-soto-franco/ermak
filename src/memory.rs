//! Explicit memory guardrails.
//!
//! This box has a history of OOM kills, and the GPU has only ~8 GiB of VRAM, so
//! ermak never allocates a buffer without first checking it against a budget.
//! Over-budget requests return [`ErmakError::MemoryBudgetExceeded`] instead of
//! allocating, and large ensembles are streamed in bounded batches so the peak
//! footprint is set by the batch size, not the requested workload size.

use crate::error::ErmakError;

/// A byte ceiling for a class of allocation (host buffers, or device VRAM).
#[derive(Debug, Clone, Copy)]
pub struct MemoryBudget {
    cap_bytes: usize,
    label: &'static str,
}

impl MemoryBudget {
    #[must_use]
    pub fn new(cap_bytes: usize, label: &'static str) -> Self {
        Self { cap_bytes, label }
    }

    /// Budget from `ERMAK_MAX_BYTES` (a plain byte count), else `default_bytes`.
    #[must_use]
    pub fn from_env_or(default_bytes: usize, label: &'static str) -> Self {
        let raw = std::env::var("ERMAK_MAX_BYTES").ok();
        Self::new(parse_max_bytes(raw.as_deref(), default_bytes), label)
    }

    #[must_use]
    pub fn cap_bytes(&self) -> usize {
        self.cap_bytes
    }

    /// `Ok` if `requested_bytes` fits, else an explicit budget error (no
    /// allocation happens).
    ///
    /// # Errors
    /// [`ErmakError::MemoryBudgetExceeded`] when `requested_bytes > cap`.
    pub fn ensure_fits(&self, requested_bytes: usize) -> Result<(), ErmakError> {
        if requested_bytes > self.cap_bytes {
            return Err(ErmakError::MemoryBudgetExceeded {
                what: self.label,
                requested_bytes,
                cap_bytes: self.cap_bytes,
            });
        }
        Ok(())
    }

    /// How many items of `bytes_per_item` fit under the cap (0 if the item size
    /// is 0). Used to size a streaming batch to the budget.
    #[must_use]
    pub fn max_items(&self, bytes_per_item: usize) -> usize {
        if bytes_per_item == 0 {
            return 0;
        }
        self.cap_bytes / bytes_per_item
    }
}

/// Parse `ERMAK_MAX_BYTES` (a plain integer byte count); fall back to
/// `default_bytes` when absent or unparseable.
#[must_use]
pub fn parse_max_bytes(raw: Option<&str>, default_bytes: usize) -> usize {
    raw.and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default_bytes)
}

/// Partition `total` items into contiguous `(start, len)` spans of at most
/// `batch` items each, so a caller can stream a large ensemble without ever
/// holding more than `batch` items in memory. A `batch` of 0 means "one span".
#[must_use]
pub fn batch_spans(total: usize, batch: usize) -> Vec<(usize, usize)> {
    if total == 0 {
        return Vec::new();
    }
    if batch == 0 {
        return vec![(0, total)];
    }
    let mut spans = Vec::with_capacity(total.div_ceil(batch));
    let mut start = 0;
    while start < total {
        let len = batch.min(total - start);
        spans.push((start, len));
        start += len;
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_fits_accepts_within_and_rejects_over() {
        let b = MemoryBudget::new(1000, "test buffer");
        assert!(b.ensure_fits(1000).is_ok());
        assert!(b.ensure_fits(500).is_ok());
        match b.ensure_fits(1001) {
            Err(ErmakError::MemoryBudgetExceeded {
                requested_bytes,
                cap_bytes,
                ..
            }) => {
                assert_eq!(requested_bytes, 1001);
                assert_eq!(cap_bytes, 1000);
            }
            other => panic!("expected budget error, got {other:?}"),
        }
    }

    #[test]
    fn max_items_is_floor_and_guards_zero() {
        let b = MemoryBudget::new(1000, "x");
        assert_eq!(b.max_items(40), 25);
        assert_eq!(b.max_items(30), 33); // floor(1000 / 30)
        assert_eq!(b.max_items(0), 0); // no divide-by-zero
    }

    #[test]
    fn batch_spans_partitions_total() {
        assert_eq!(batch_spans(10, 4), vec![(0, 4), (4, 4), (8, 2)]);
        assert_eq!(batch_spans(8, 4), vec![(0, 4), (4, 4)]);
        assert_eq!(batch_spans(0, 4), vec![]);
        assert_eq!(batch_spans(3, 0), vec![(0, 3)]); // batch 0 => single span
        let spans = batch_spans(101, 16);
        assert_eq!(spans.iter().map(|(_, l)| l).sum::<usize>(), 101);
        assert!(spans.iter().all(|(_, l)| *l <= 16));
    }

    #[test]
    fn parse_max_bytes_falls_back_on_missing_or_bad() {
        assert_eq!(parse_max_bytes(None, 999), 999);
        assert_eq!(parse_max_bytes(Some("512"), 999), 512);
        assert_eq!(parse_max_bytes(Some("garbage"), 999), 999);
    }
}
