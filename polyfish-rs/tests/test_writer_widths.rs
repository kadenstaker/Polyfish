//! Every training-data writer must derive the option-head width from
//! `mapper::NUM_MOVE_OPTIONS`, never from a literal.
//!
//! `tests/parity_widths.rs` ties that constant to `train.py`'s `pi_option`
//! head, so a writer holding its own copy of the number passes parity while
//! emitting stale-width files — the drift class behind the resolved
//! `NUM_ACTION_TYPES` 12-vs-11 trap (#3).

use polyfish::ai::mapper::NUM_MOVE_OPTIONS;
use regex::Regex;

const WRITERS: &[&str] = &[
    "src/replay/training.rs",
    "src/recorder.rs",
    "src/bin/self_play.rs",
];

fn read(rel: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

#[test]
fn writers_carry_no_literal_option_width() {
    let literal = Regex::new(&format!(r"\b{NUM_MOVE_OPTIONS}\b")).unwrap();
    let offenders: Vec<String> = WRITERS
        .iter()
        .flat_map(|rel| {
            read(rel)
                .lines()
                .enumerate()
                .filter(|(_, line)| literal.is_match(line))
                .map(|(i, line)| format!("  {rel}:{}: {}", i + 1, line.trim()))
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(
        offenders.is_empty(),
        "training-data writers must use mapper::NUM_MOVE_OPTIONS, not the literal \
         {NUM_MOVE_OPTIONS}:\n{}",
        offenders.join("\n")
    );
}
