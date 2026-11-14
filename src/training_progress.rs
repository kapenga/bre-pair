//! Opt-in CLI telemetry. Library callers remain silent unless explicitly enabled.
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

static ENABLED: AtomicBool = AtomicBool::new(false);
static COMPLETED: Mutex<Option<String>> = Mutex::new(None);

pub struct Scope(bool);

pub fn enable() -> Scope {
    *COMPLETED.lock().expect("progress snapshot lock") = None;
    Scope(ENABLED.swap(true, Ordering::Relaxed))
}

impl Drop for Scope {
    fn drop(&mut self) {
        ENABLED.store(self.0, Ordering::Relaxed);
    }
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Observe finished selection only: no counters or candidate ordering are mutated.
/// Counts include nonoverlapping replacements and remain 64-bit throughout.
pub(crate) fn wave(number: usize, pairs: &[crate::batch::Candidate]) {
    if enabled() {
        let selected_pairs = pairs.len();
        let replacements: u64 = pairs.iter().map(|p| p.count).sum();
        eprintln!("codec wave {{\"event\":\"wave\",\"wave\":{number},\"selected_pairs\":{selected_pairs},\"selected_replacements\":{replacements}}}");
    }
}

pub fn frontier(round: usize, candidates: usize, maximum: u64, threshold: u64) {
    if enabled() {
        eprintln!("codec frontier {{\"event\":\"frontier\",\"round\":{round},\"candidates\":{candidates},\"maximum_pair_count\":{maximum},\"stop_max_pair_count\":{threshold}}}");
    }
}

/// Names and units are internal constants, never corpus strings. Byte/symbol
/// counts remain integers, and events occur at chunk/byte-budget boundaries.
pub fn phase(name: &str, round: usize, done: u64, total: u64, unit: &str) {
    if enabled() {
        let state = COMPLETED.lock().expect("progress snapshot lock");
        let state = state.as_deref().unwrap_or("null");
        eprintln!("codec phase {{\"event\":\"phase\",\"phase\":\"{name}\",\"round\":{round},\"done\":{done},\"total\":{total},\"unit\":\"{unit}\",\"completed\":{state}}}");
    }
}

pub fn seed(grammar_rules: usize, symbols: u64, packed_bytes: usize, symbol_bits: u8) {
    if enabled() {
        let state = format!("{{\"round\":0,\"grammar_rules\":{grammar_rules},\"rules_created\":0,\"symbols\":{symbols},\"packed_bytes\":{packed_bytes},\"symbol_bits\":{symbol_bits},\"minimum_selected_count\":null,\"maximum_selected_count\":null}}");
        eprintln!("codec phase {{\"event\":\"seed\",\"completed\":{state}}}");
        *COMPLETED.lock().expect("progress snapshot lock") = Some(state);
    }
}

/// Retain only numeric counters, never a grammar, symbol buffer or pair table.
pub fn completed(round: &crate::engine::Round, created: usize) {
    if enabled() {
        *COMPLETED.lock().expect("progress snapshot lock") = Some(format!(
            "{{\"round\":{},\"grammar_rules\":{},\"rules_created\":{},\"symbols\":{},\"packed_bytes\":{},\"symbol_bits\":{},\"minimum_selected_count\":{},\"maximum_selected_count\":{}}}",
            round.index, round.total_rules, created, round.after_refill,
            round.packed_bytes, round.width, round.minimum_selected_count, round.maximum_selected_count,
        ));
    }
}
