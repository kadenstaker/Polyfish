//! Read-only HTTP handlers for the training dashboard.

use axum::{
    Json,
    extract::Query,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use safetensors::SafeTensors;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const CSV_PATH: &str = "training_log.csv";
const MOVES_PATH: &str = "moves_by_turn.json";
const VALUE_DIST_PATH: &str = "value_distribution.json";
const LADDER_PATH: &str = "ladder.json";
const RATINGS_PATH: &str = "elo_ratings.json";

/// Columns kept as strings even when they parse as numbers: `run_id` is a unix
/// timestamp the dashboard compares and formats as text.
const CSV_TEXT_COLUMNS: &[&str] = &[
    "run_id",
    "iter_started_at",
    "run_started_at",
    "games_file",
    "match_type",
];

/// Every column of `training_log.csv` verbatim, numbers where the cell parses
/// and null for blanks. Reading the header instead of a fixed struct means a
/// column added to the CSV reaches every consumer without a change here; the
/// fixed-struct reader this replaced silently dropped 23 of them.
pub fn training_csv_rows() -> Vec<Value> {
    parse_training_csv(&std::fs::read_to_string(CSV_PATH).unwrap_or_default())
}

fn parse_training_csv(content: &str) -> Vec<Value> {
    let mut lines = content.lines().filter(|l| !l.trim().is_empty());
    let Some(header) = lines.next() else {
        return Vec::new();
    };
    let headers: Vec<&str> = header.split(',').collect();
    lines
        .filter(|line| line.split(',').count() >= 5)
        .map(|line| {
            let cells: Vec<&str> = line.split(',').collect();
            let row: serde_json::Map<String, Value> = headers
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let cell = cells.get(i).copied().unwrap_or("").trim();
                    let value = if CSV_TEXT_COLUMNS.contains(name) {
                        Value::from(cell)
                    } else if cell.is_empty() {
                        Value::Null
                    } else {
                        cell.parse::<f64>()
                            .map(Value::from)
                            .unwrap_or_else(|_| Value::from(cell))
                    };
                    ((*name).to_string(), value)
                })
                .collect();
            Value::Object(row)
        })
        .collect()
}

fn cell_str<'a>(row: &'a Value, name: &str) -> &'a str {
    row.get(name).and_then(Value::as_str).unwrap_or_default()
}

/// A blank or unparseable cell reads 0, as it did through the fixed-struct
/// reader's parse fallback, so the runs list keeps its shape.
fn cell_f64(row: &Value, name: &str) -> f64 {
    row.get(name).and_then(Value::as_f64).unwrap_or(0.0)
}

#[derive(Debug, Serialize)]
struct RunSummary {
    run_id: String,
    run_started_at: String,
    iter_count: usize,
    iter_min: i64,
    iter_max: i64,
    last_loss: f64,
    best_score: f64,
}

fn run_summaries(rows: &[Value]) -> Vec<RunSummary> {
    let mut by_run: HashMap<&str, Vec<&Value>> = HashMap::new();
    for row in rows {
        by_run.entry(cell_str(row, "run_id")).or_default().push(row);
    }

    let mut summaries: Vec<RunSummary> = by_run
        .into_iter()
        .map(|(run_id, mut run_rows)| {
            run_rows.sort_by(|a, b| cell_f64(a, "iteration").total_cmp(&cell_f64(b, "iteration")));
            let first = run_rows[0];
            let last = run_rows[run_rows.len() - 1];
            let started = match cell_str(first, "iter_started_at") {
                "" => cell_str(first, "run_started_at"),
                v => v,
            };
            RunSummary {
                run_id: run_id.to_string(),
                run_started_at: started.to_string(),
                iter_count: run_rows.len(),
                iter_min: cell_f64(first, "iteration") as i64,
                iter_max: cell_f64(last, "iteration") as i64,
                last_loss: cell_f64(last, "loss"),
                best_score: run_rows
                    .iter()
                    .map(|r| cell_f64(r, "max_score"))
                    .fold(0.0_f64, f64::max),
            }
        })
        .collect();

    summaries.sort_by(|a, b| b.run_id.cmp(&a.run_id));
    summaries
}

pub async fn api_runs() -> Json<Value> {
    Json(json!(run_summaries(&training_csv_rows())))
}

#[derive(Debug, serde::Deserialize)]
pub struct RunFilter {
    pub run: Option<String>,
}

pub async fn api_training_metrics(Query(q): Query<RunFilter>) -> Json<Value> {
    let rows: Vec<Value> = training_csv_rows()
        .into_iter()
        .filter(|r| {
            q.run
                .as_ref()
                .is_none_or(|id| cell_str(r, "run_id") == id.as_str())
        })
        .collect();
    Json(Value::Array(rows))
}

pub async fn api_moves_by_turn(Query(q): Query<RunFilter>) -> Json<Value> {
    let content = std::fs::read_to_string(MOVES_PATH).unwrap_or_else(|_| "{}".to_string());
    let all: Value = serde_json::from_str(&content).unwrap_or(json!({}));
    if let Some(run_id) = &q.run {
        if let Some(run_data) = all.get(run_id) {
            return Json(run_data.clone());
        }
        return Json(json!({}));
    }
    Json(all)
}

/// The ladder plus elo.py's joint fit under `ratings`, once the loop has
/// written one. A reading's own `elo_est` is one match chained onto one
/// anchor's number; the fit is every recorded match at once.
fn ladder_with_ratings(ladder: &str, ratings: Option<String>) -> Value {
    let mut all: Value =
        serde_json::from_str(ladder).unwrap_or_else(|_| json!({ "anchors": [], "readings": [] }));
    let fitted = ratings.and_then(|text| serde_json::from_str::<Value>(&text).ok());
    if let (Some(obj), Some(fitted)) = (all.as_object_mut(), fitted) {
        obj.insert("ratings".to_string(), fitted);
    }
    all
}

pub async fn api_elo_ladder() -> Json<Value> {
    let ladder = std::fs::read_to_string(LADDER_PATH).unwrap_or_else(|_| "{}".to_string());
    Json(ladder_with_ratings(
        &ladder,
        std::fs::read_to_string(RATINGS_PATH).ok(),
    ))
}

#[derive(Debug, serde::Deserialize)]
pub struct ValueDistQuery {
    pub file: Option<String>,
    pub run: Option<String>,
    pub iteration: Option<i64>,
}

#[derive(Debug, Serialize)]
struct ValueDistribution {
    file: String,
    n: usize,
    stats: ValueStats,
    hist: HistData,
    abs_hist: HistData,
    buckets: BucketData,
    samples: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct ValueStats {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    weak_pct: f64,
    moderate_pct: f64,
    strong_pct: f64,
    saturation_pct: f64,
    in_target_range_pct: f64,
}

#[derive(Debug, Serialize)]
struct HistData {
    bins: Vec<f64>,
    counts: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct BucketData {
    weak: f64,
    moderate: f64,
    strong: f64,
    saturation: f64,
}

fn list_games_files() -> Vec<String> {
    let mut files = Vec::new();
    for dir in ["", "archive"] {
        let path = Path::new(dir);
        if !path.exists() {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("games_") && name.ends_with(".safetensors") {
                    let rel = if dir.is_empty() {
                        name
                    } else {
                        format!("{dir}/{name}")
                    };
                    files.push(rel);
                }
            }
        }
    }
    files.sort_by(|a, b| b.cmp(a));
    files
}

fn resolve_games_path(file: &str) -> PathBuf {
    let p = PathBuf::from(file);
    if p.exists() {
        return p;
    }
    PathBuf::from("archive").join(file.trim_start_matches("archive/"))
}

fn load_values(path: &Path) -> anyhow::Result<Vec<f32>> {
    let data = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&data)?;
    let view = st
        .tensor("values")
        .map_err(|e| anyhow::anyhow!("no 'values' tensor: {e}"))?;
    let bytes = view.data();
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn compute_distribution(values: &[f32]) -> ValueDistribution {
    let n = values.len();
    let mean = if n > 0 {
        values.iter().map(|v| *v as f64).sum::<f64>() / n as f64
    } else {
        0.0
    };
    let variance = if n > 1 {
        values
            .iter()
            .map(|v| {
                let d = *v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64
    } else {
        0.0
    };
    let std = variance.sqrt();
    let min = values.iter().copied().fold(f32::INFINITY, f32::min) as f64;
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;

    let mut weak = 0usize;
    let mut moderate = 0usize;
    let mut strong = 0usize;
    let mut saturation = 0usize;
    let mut in_target = 0usize;

    for v in values {
        let av = v.abs();
        if av < 0.1 {
            weak += 1;
        } else if av < 0.3 {
            moderate += 1;
        } else if av < 0.5 {
            strong += 1;
        } else {
            saturation += 1;
        }
        if av >= 0.1 && av <= 0.5 {
            in_target += 1;
        }
    }

    let pct = |c: usize| {
        if n > 0 {
            100.0 * c as f64 / n as f64
        } else {
            0.0
        }
    };

    const HIST_BINS: usize = 80;
    let mut hist_counts = vec![0usize; HIST_BINS];
    let mut abs_hist_counts = vec![0usize; HIST_BINS];
    for v in values {
        let idx = (((*v + 1.0) / 2.0).clamp(0.0, 0.9999) * HIST_BINS as f32) as usize;
        hist_counts[idx] += 1;
        let av = v.abs().clamp(0.0, 1.0);
        let aidx = ((av * 0.9999) * HIST_BINS as f32) as usize;
        abs_hist_counts[aidx] += 1;
    }

    let hist_bins: Vec<f64> = (0..HIST_BINS)
        .map(|i| -1.0 + (2.0 * (i as f64 + 0.5) / HIST_BINS as f64))
        .collect();
    let abs_bins: Vec<f64> = (0..HIST_BINS)
        .map(|i| (i as f64 + 0.5) / HIST_BINS as f64)
        .collect();

    let max_samples = 8000usize;
    let samples: Vec<f32> = if values.len() <= max_samples {
        values.to_vec()
    } else {
        let step = values.len() / max_samples;
        values.iter().step_by(step.max(1)).copied().collect()
    };

    ValueDistribution {
        file: String::new(),
        n,
        stats: ValueStats {
            mean,
            std,
            min,
            max,
            weak_pct: pct(weak),
            moderate_pct: pct(moderate),
            strong_pct: pct(strong),
            saturation_pct: pct(saturation),
            in_target_range_pct: pct(in_target),
        },
        hist: HistData {
            bins: hist_bins,
            counts: hist_counts,
        },
        abs_hist: HistData {
            bins: abs_bins,
            counts: abs_hist_counts,
        },
        buckets: BucketData {
            weak: pct(weak),
            moderate: pct(moderate),
            strong: pct(strong),
            saturation: pct(saturation),
        },
        samples,
    }
}

fn load_value_dist_cache(run_id: Option<&str>, iteration: Option<i64>) -> Option<Value> {
    let content = std::fs::read_to_string(VALUE_DIST_PATH).ok()?;
    let all: Value = serde_json::from_str(&content).ok()?;
    let run_id = run_id?;
    let iteration = iteration?;
    all.get(run_id)?.get(iteration.to_string()).cloned()
}

pub async fn api_value_distribution(
    Query(q): Query<ValueDistQuery>,
) -> Result<Json<Value>, ApiError> {
    if q.file.is_none() && q.run.is_none() {
        return Ok(Json(json!({ "files": list_games_files() })));
    }

    if let (Some(run_id), Some(iteration)) = (&q.run, q.iteration) {
        if let Some(cached) = load_value_dist_cache(Some(run_id), Some(iteration)) {
            return Ok(Json(cached));
        }
    }

    let file = q.file.as_deref().unwrap_or("");
    if file.is_empty() {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            "value distribution not found (no cache entry)".into(),
        ));
    }

    let path = resolve_games_path(file);
    if path.exists() {
        let values = load_values(&path).map_err(|e| {
            ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to read values: {e}"),
            )
        })?;
        let mut dist = compute_distribution(&values);
        dist.file = file.to_string();
        return Ok(Json(serde_json::to_value(dist).unwrap_or(json!({}))));
    }

    if let Some(cached) = load_value_dist_cache(q.run.as_deref(), q.iteration) {
        return Ok(Json(cached));
    }

    Err(ApiError(
        StatusCode::NOT_FOUND,
        format!("file not found: {file}"),
    ))
}

pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CSV: &str = concat!(
        "run_id,run_started_at,iter_started_at,iteration,loss,max_score,brand_new_column\n",
        "1755,2026-08-01T00:00:00Z,2026-08-01T01:00:00Z,2,,12,7\n",
        "1755,2026-08-01T00:00:00Z,,1,0.5,15,\n",
    );

    #[test]
    fn a_column_the_reader_has_never_heard_of_still_reaches_the_dashboard() {
        let rows = parse_training_csv(CSV);
        assert_eq!(rows[0]["brand_new_column"], 7.0);
    }

    #[test]
    fn blanks_are_null_and_text_columns_stay_text() {
        let rows = parse_training_csv(CSV);
        assert_eq!(rows[0]["run_id"], "1755");
        assert_eq!(rows[1]["iter_started_at"], "");
        assert!(rows[0]["loss"].is_null());
        assert!(rows[1]["brand_new_column"].is_null());
    }

    #[test]
    fn a_run_summary_reads_a_blank_cell_as_zero_and_falls_back_for_its_start() {
        let summaries = run_summaries(&parse_training_csv(CSV));
        assert_eq!(summaries.len(), 1);
        let run = &summaries[0];
        assert_eq!(run.run_started_at, "2026-08-01T00:00:00Z");
        assert_eq!((run.iter_min, run.iter_max, run.iter_count), (1, 2, 2));
        assert_eq!(run.last_loss, 0.0);
        assert_eq!(run.best_score, 15.0);
    }

    #[test]
    fn the_joint_fit_rides_along_with_the_ladder() {
        let out = ladder_with_ratings(
            r#"{"anchors":[],"readings":[]}"#,
            Some(r#"{"greedy":{"elo":0.0}}"#.to_string()),
        );
        assert_eq!(out["ratings"]["greedy"]["elo"], 0.0);
        assert!(out["readings"].is_array());
    }

    #[test]
    fn a_missing_or_unreadable_fit_leaves_the_ladder_intact() {
        for ratings in [None, Some("not json".to_string())] {
            let out = ladder_with_ratings(r#"{"anchors":[{"name":"greedy"}]}"#, ratings);
            assert_eq!(out["anchors"][0]["name"], "greedy");
            assert!(out.get("ratings").is_none());
        }
    }
}
