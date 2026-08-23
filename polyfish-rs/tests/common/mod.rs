//! Shared test support. Compiled into every crate that declares `mod common;`,
//! each of which uses a different subset, hence the blanket `dead_code` allows.

#![allow(dead_code)]

use polyfish::game::set_adversarial_search;
use std::sync::{Mutex, MutexGuard};

/// The adversarial switch is process-wide, so mode-sensitive tests must not
/// run concurrently. Every such test holds this for its whole body.
static MODE: Mutex<()> = Mutex::new(());

pub struct AdversarialModeGuard(#[allow(dead_code)] MutexGuard<'static, ()>);

impl AdversarialModeGuard {
    pub fn set(on: bool) -> Self {
        let g = MODE.lock().unwrap_or_else(|e| e.into_inner());
        set_adversarial_search(on);
        Self(g)
    }
}

impl Drop for AdversarialModeGuard {
    fn drop(&mut self) {
        set_adversarial_search(false);
    }
}
