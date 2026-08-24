use clap::Parser;
use polyfish::supabase::{SupabaseTarget, confirm_destructive, count_rows};

/// Clears `verified` on every row of the Supabase games table.
#[derive(Debug, Parser)]
#[command(about = "Reset `verified` to false on every row of the Supabase games table")]
struct Args {
    /// Skip the typed confirmation. Required for non-interactive runs.
    #[arg(long)]
    yes: bool,
    /// Print the resolved target and row count, then exit without writing.
    #[arg(long)]
    dry_run: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let target = SupabaseTarget::from_env()?;
    target.describe();

    let client = reqwest::Client::new();
    match count_rows(&client, &target, "games").await {
        Some(n) => println!("Rows that would be unverified: {n}"),
        None => println!("Rows that would be unverified: unknown (count query failed)"),
    }

    if args.dry_run {
        println!("--dry-run: nothing was changed.");
        return Ok(());
    }
    if !confirm_destructive("unverify every games row", "unverify all", args.yes) {
        std::process::exit(1);
    }

    println!("🧹 Unverifying all games (resetting to false)...");
    let res = client
        .patch(target.rest("games?id=not.is.null"))
        .header("apikey", &target.key)
        .header("Authorization", format!("Bearer {}", target.key))
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({ "verified": false }))
        .send()
        .await?;

    if res.status().is_success() {
        println!("✅ All games unverified.");
    } else {
        eprintln!("❌ Failed to unverify games: {}", res.text().await?);
    }

    Ok(())
}
