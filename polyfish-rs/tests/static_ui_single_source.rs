//! #55: the static UI had a second copy under `polyfish-ui/public/simulator`
//! and the server mounted both, so drift accumulated in both directions for a
//! month. These pin the single-source wiring, not any rendering behaviour.

use polyfish::web_static;
use std::path::Path;

#[test]
fn static_ui_has_exactly_one_copy() {
    assert!(
        !Path::new("../polyfish-ui/public/simulator").exists(),
        "the static UI fork is back; src/public is the only copy (#55)"
    );
    let root = Path::new(web_static::STATIC_UI);
    assert!(root.join("index.html").exists(), "missing the simulator");
    assert!(root.join("training.html").exists(), "missing the dashboard");
}

#[test]
fn simulator_mount_is_the_static_ui() {
    // polyfish-ui/src/App.tsx iframes /simulator/*; it must not read dist/.
    assert_eq!(web_static::STATIC_UI, "../src/public");
    let main_rs = std::fs::read_to_string("src/main.rs").unwrap();
    assert!(
        main_rs.contains(r#".nest_service("/simulator", ServeDir::new(web_static::STATIC_UI))"#),
        "/simulator must be served from the single static-UI root (#55)"
    );
}

#[test]
fn move_type_inference_reads_the_real_serde_fields() {
    // 89c5c1b fixed this in the fork only: src/public tested `move.type` four
    // times over, so every unlabelled move resolved to Research.
    for f in ["../src/public/js/main.js", "../src/public/js/map.js"] {
        let js = std::fs::read_to_string(f).unwrap();
        for field in ["move.techType", "move.structure", "move.reward"] {
            assert!(
                js.contains(&format!("{field} !== undefined")),
                "{f}: {field}"
            );
        }
    }
}

#[test]
fn replay_ui_targets_routes_that_exist() {
    // src/public's replay player POSTed /replay/analyze, which no route serves.
    let js = std::fs::read_to_string("../src/public/js/replay.js").unwrap();
    assert!(!js.contains("/replay/analyze"), "no such route (#55)");
    let main_rs = std::fs::read_to_string("src/main.rs").unwrap();
    for route in ["/replay/open", "/replay/state"] {
        assert!(js.contains(route), "replay.js no longer uses {route}");
        assert!(
            main_rs.contains(&format!("\"{route}\"")),
            "unrouted {route}"
        );
    }
}
