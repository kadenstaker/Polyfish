use clap::Parser;
use polyfish::supabase::{SupabaseTarget, confirm_destructive, count_rows};
use serde_json::Value;

/// Empties the Supabase storage bucket and the games table.
#[derive(Debug, Parser)]
#[command(about = "Empty the Supabase replay bucket and delete every games row")]
struct Args {
    /// Skip the typed confirmation. Required for non-interactive runs.
    #[arg(long)]
    yes: bool,
    /// Print the resolved target and everything that would be deleted, then exit.
    #[arg(long)]
    dry_run: bool,
}

/// One page of object names in the bucket, folder placeholders filtered out.
async fn list_bucket(
    client: &reqwest::Client,
    target: &SupabaseTarget,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let res = client
        .post(target.storage(&format!("object/list/{}", target.bucket)))
        .header("apikey", &target.key)
        .header("Authorization", format!("Bearer {}", target.key))
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({ "limit": 1000, "offset": 0 }))
        .send()
        .await?;

    if !res.status().is_success() {
        return Err(format!("failed to list bucket: {}", res.text().await?).into());
    }

    let files: Vec<Value> = res.json().await?;
    Ok(files
        .into_iter()
        .filter_map(|f| f["name"].as_str().map(|s| s.to_string()))
        .filter(|name| !name.is_empty())
        .collect())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let target = SupabaseTarget::from_env()?;
    target.describe();

    let client = reqwest::Client::new();
    let objects = list_bucket(&client, &target).await?;
    println!(
        "Objects in bucket (first page): {}{}",
        objects.len(),
        if objects.len() == 1000 { "+" } else { "" }
    );
    match count_rows(&client, &target, "games").await {
        Some(n) => println!("Rows in 'games' that would be deleted: {n}"),
        None => println!("Rows in 'games' that would be deleted: unknown (count query failed)"),
    }

    if args.dry_run {
        for name in &objects {
            println!("  would delete {name}");
        }
        println!("--dry-run: nothing was deleted.");
        return Ok(());
    }
    if !confirm_destructive(
        "delete every replay object and every games row",
        "delete everything",
        args.yes,
    ) {
        std::process::exit(1);
    }

    // --- 1. Empty Storage Bucket ---
    println!("\n🗑️  Emptying storage bucket '{}'...", target.bucket);
    let mut files_deleted = 0;

    loop {
        let filenames = list_bucket(&client, &target).await?;
        if filenames.is_empty() {
            println!("✅ Bucket is empty.");
            break;
        }

        // Bulk delete API
        let del_res = client
            .delete(target.storage(&format!("object/{}", target.bucket)))
            .header("apikey", &target.key)
            .header("Authorization", format!("Bearer {}", target.key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({ "prefixes": filenames }))
            .send()
            .await?;

        if del_res.status().is_success() {
            let count = filenames.len();
            files_deleted += count;
            println!("Deleted {} files... (Total: {})", count, files_deleted);
        } else {
            eprintln!(
                "❌ Failed to delete files from bucket: {}",
                del_res.text().await?
            );
            break;
        }
    }

    // --- 2. Clear Database Table ---
    println!("\n🗑️  Deleting all rows from 'games' table...");
    let res = client
        .delete(target.rest("games?id=not.is.null"))
        .header("apikey", &target.key)
        .header("Authorization", format!("Bearer {}", target.key))
        .send()
        .await?;

    if res.status().is_success() {
        println!("✅ Completely wiped all rows from 'games' table.");
    } else {
        eprintln!("❌ Failed to delete rows from table: {}", res.text().await?);
    }

    println!("\n🎉 Wipe complete. You are ready to re-scrape everything!");

    Ok(())
}
