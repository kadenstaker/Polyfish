//! Shared plumbing for the Supabase one-shot maintenance binaries: resolving
//! the target from the environment, counting what a wipe would touch, and
//! gating a destructive run behind an explicit confirmation.

use std::io::{IsTerminal, Write};

pub struct SupabaseTarget {
    pub url: String,
    pub key: String,
    pub bucket: String,
}

impl SupabaseTarget {
    /// Reads `.env` plus the process environment. The key is never printed.
    pub fn from_env() -> anyhow::Result<Self> {
        let _ = dotenvy::dotenv();
        let url = std::env::var("SUPABASE_URL").unwrap_or_default();
        let url = url.trim_end_matches('/').to_string();
        let key = std::env::var("SUPABASE_SERVICE_ROLE_KEY")
            .or_else(|_| std::env::var("SUPABASE_PUBLIC_ANON_KEY"))
            .unwrap_or_default();
        if url.is_empty() {
            anyhow::bail!("SUPABASE_URL is not set");
        }
        if key.is_empty() {
            anyhow::bail!("neither SUPABASE_SERVICE_ROLE_KEY nor SUPABASE_PUBLIC_ANON_KEY is set");
        }
        let bucket =
            std::env::var("SUPABASE_STORAGE_BUCKET").unwrap_or_else(|_| "games".to_string());
        Ok(Self { url, key, bucket })
    }

    pub fn rest(&self, path: &str) -> String {
        format!("{}/rest/v1/{}", self.url, path)
    }

    pub fn storage(&self, path: &str) -> String {
        format!("{}/storage/v1/{}", self.url, path)
    }

    pub fn describe(&self) {
        println!("Supabase target: {}", self.url);
        println!("Storage bucket:  {}", self.bucket);
    }
}

/// Total row count from a PostgREST `Content-Range` header (`0-24/1234`).
pub fn parse_content_range_total(value: &str) -> Option<u64> {
    value.rsplit('/').next()?.trim().parse().ok()
}

/// Exact row count for a table, or `None` when the count query did not answer.
pub async fn count_rows(
    client: &reqwest::Client,
    target: &SupabaseTarget,
    table: &str,
) -> Option<u64> {
    let res = client
        .get(target.rest(&format!("{table}?select=id")))
        .header("apikey", &target.key)
        .header("Authorization", format!("Bearer {}", target.key))
        .header("Prefer", "count=exact")
        .header("Range", "0-0")
        .send()
        .await
        .ok()?;
    if !res.status().is_success() {
        return None;
    }
    parse_content_range_total(res.headers().get("content-range")?.to_str().ok()?)
}

/// Whether a destructive run may proceed. `typed` is the operator's answer, or
/// `None` when there was no terminal to ask on.
pub fn confirmed(phrase: &str, yes: bool, typed: Option<&str>) -> bool {
    yes || typed.is_some_and(|answer| answer.trim() == phrase)
}

/// Prompts on a terminal and returns the verdict; refuses when there is none.
pub fn confirm_destructive(action: &str, phrase: &str, yes: bool) -> bool {
    if yes {
        return true;
    }
    if !std::io::stdin().is_terminal() {
        eprintln!("Refusing to {action} without --yes: no terminal to confirm on.");
        return false;
    }
    print!("About to {action}. Type \"{phrase}\" to proceed: ");
    let _ = std::io::stdout().flush();
    let mut typed = String::new();
    if std::io::stdin().read_line(&mut typed).is_err() {
        return false;
    }
    let ok = confirmed(phrase, false, Some(&typed));
    if !ok {
        eprintln!("Confirmation did not match; nothing was changed.");
    }
    ok
}
