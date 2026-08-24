//! The destructive Supabase one-shots must refuse to run unasked.

use polyfish::supabase::{confirmed, parse_content_range_total};

#[test]
fn no_answer_means_no() {
    assert!(!confirmed("delete everything", false, None));
}

#[test]
fn only_the_exact_phrase_confirms() {
    assert!(confirmed(
        "delete everything",
        false,
        Some("delete everything")
    ));
    assert!(confirmed(
        "delete everything",
        false,
        Some("delete everything\n")
    ));
    assert!(!confirmed("delete everything", false, Some("y")));
    assert!(!confirmed("delete everything", false, Some("")));
    assert!(!confirmed(
        "delete everything",
        false,
        Some("Delete Everything")
    ));
}

#[test]
fn yes_flag_short_circuits() {
    assert!(confirmed("delete everything", true, None));
    assert!(confirmed("delete everything", true, Some("no")));
}

#[test]
fn content_range_total_is_the_denominator() {
    assert_eq!(parse_content_range_total("0-0/1234"), Some(1234));
    assert_eq!(parse_content_range_total("*/0"), Some(0));
    assert_eq!(parse_content_range_total("0-24/*"), None);
    assert_eq!(parse_content_range_total(""), None);
}
