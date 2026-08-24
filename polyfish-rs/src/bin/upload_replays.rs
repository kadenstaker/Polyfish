use polyfish::replay::{REPLAY_DIR, canonical_replay_file_name, is_canonical_replay_file};
use std::env;
use std::fs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let supabase_key = env::var("SUPABASE_SERVICE_ROLE_KEY")
        .or_else(|_| env::var("SUPABASE_PUBLIC_ANON_KEY"))
        .unwrap_or_default();
    let supabase_url = env::var("SUPABASE_URL").unwrap_or_default();

    if supabase_url.is_empty() || supabase_key.is_empty() {
        println!("Error: Supabase URL or Key not set in ENV.");
        return Ok(());
    }

    let bucket_name = env::var("SUPABASE_STORAGE_BUCKET").unwrap_or_else(|_| "games".to_string());
    let client = reqwest::Client::new();

    let entries = fs::read_dir(REPLAY_DIR)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if !path.is_file() || !is_canonical_replay_file(&path) {
            continue;
        }

        let content = fs::read_to_string(&path)?;
        let replay = match polyfish::replay::load_replay(&path) {
            Ok(replay) => replay,
            Err(e) => {
                println!("Skipping {:?}, invalid canonical replay: {}", path, e);
                continue;
            }
        };
        let game_name = if replay.initial_state.settings.game_name.is_empty() {
            replay.metadata.game_id.as_deref().unwrap_or("Unknown")
        } else {
            &replay.initial_state.settings.game_name
        };
        let seed = replay.initial_state.initial_seed;
        let uuid_val = replay.metadata.game_id.clone().unwrap_or_default();

        let db_url = if !uuid_val.is_empty() {
            format!(
                "{}/rest/v1/games?uuid=eq.{}&select=id",
                supabase_url.trim_end_matches('/'),
                uuid_val
            )
        } else {
            let safe_game_name = game_name.replace(" ", "%20");
            format!(
                "{}/rest/v1/games?seed=eq.{}&game_name=eq.{}&select=id",
                supabase_url.trim_end_matches('/'),
                seed,
                safe_game_name
            )
        };

        // 1. Check if it already exists
        let check_req = client
            .get(&db_url)
            .header("apikey", &supabase_key)
            .header("Authorization", format!("Bearer {}", supabase_key))
            .send()
            .await?;

        // Fail closed: anything short of a 2xx array body skips the upload rather
        // than falling through and creating a duplicate.
        let check_status = check_req.status();
        if !check_status.is_success() {
            let body = check_req.text().await.unwrap_or_default();
            println!("❌ Duplicate check failed for {game_name} ({check_status}): {body}");
            continue;
        }
        match check_req.json::<serde_json::Value>().await {
            Ok(json) => match json.as_array() {
                Some(rows) if !rows.is_empty() => {
                    println!("⚠️ Rejected duplicate game (UUID or Seed/Name): {game_name}");
                    continue;
                }
                Some(_) => {}
                None => {
                    println!("❌ Duplicate check returned a non-array body for {game_name}");
                    continue;
                }
            },
            Err(e) => {
                println!("❌ Duplicate check body unreadable for {game_name}: {e}");
                continue;
            }
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let file_name = canonical_replay_file_name(game_name, timestamp);
        let storage_url = format!(
            "{}/storage/v1/object/{}/{}",
            supabase_url.trim_end_matches('/'),
            bucket_name,
            file_name
        );

        // 2. Upload to storage
        let upload_res = client
            .post(&storage_url)
            .header("apikey", &supabase_key)
            .header("Authorization", format!("Bearer {}", supabase_key))
            .header("Content-Type", "application/json")
            .body(content)
            .send()
            .await?;

        if !upload_res.status().is_success() {
            let err_text = upload_res.text().await.unwrap_or_default();
            println!(
                "❌ Supabase Storage Upload Failed for {}: {}",
                game_name, err_text
            );
            continue;
        }

        // 3. Insert record into games table
        let insert_url = format!("{}/rest/v1/games", supabase_url.trim_end_matches('/'));
        let mut insert_payload = serde_json::json!({
            "seed": seed,
            "game_name": game_name,
            "storage_path": file_name,
            "verified": false
        });
        if !uuid_val.is_empty() {
            insert_payload
                .as_object_mut()
                .unwrap()
                .insert("uuid".into(), serde_json::json!(uuid_val));
        }

        let insert_res = client
            .post(&insert_url)
            .header("apikey", &supabase_key)
            .header("Authorization", format!("Bearer {}", supabase_key))
            .header("Content-Type", "application/json")
            .header("Prefer", "return=minimal")
            .json(&insert_payload)
            .send()
            .await?;

        if !insert_res.status().is_success() {
            let err_text = insert_res.text().await.unwrap_or_default();
            println!(
                "❌ Supabase DB Insert Failed for {}: {}",
                game_name, err_text
            );
            continue;
        }

        println!("✅ Successfully uploaded {} ({})", game_name, file_name);
    }

    println!("Finished processing replays directory.");
    Ok(())
}
