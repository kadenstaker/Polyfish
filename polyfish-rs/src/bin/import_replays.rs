use clap::{Parser, Subcommand};
use polyfish::replay::training::{TrainingCollector, TrainingSample, write_training_files};
use polyfish::replay::{
    DivergenceVerifier, MAX_SUPPORTED_GAME_VERSION, MIN_SUPPORTED_GAME_VERSION, PairObserver,
    ReplayError, ReplayExecutor, VersionSupport, is_canonical_replay_file, load_replay,
    validate_training_eligibility_with,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Parser)]
#[command(about = "Validate canonical Polyfish replays and export behavior-cloning data")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Validate(Common),
    ExportTraining {
        #[command(flatten)]
        common: Common,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 50_000)]
        samples_per_file: usize,
    },
}

#[derive(Debug, clap::Args)]
struct Common {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    recursive: bool,
    #[arg(long)]
    fail_fast: bool,
    #[arg(long)]
    error_report: Option<PathBuf>,
    /// Import a capture whose game version is outside the supported range anyway.
    #[arg(long)]
    allow_version_drift: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FileFailure {
    file: String,
    stage: &'static str,
    version: Option<i32>,
    error: String,
}

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct Summary {
    discovered_files: usize,
    valid_files: usize,
    invalid_files: usize,
    training_eligible_files: usize,
    derived_result_files: usize,
    training_samples: usize,
    version_drift_files: usize,
    source_score_drifts: usize,
    output_files: Vec<String>,
    failures_by_version: BTreeMap<String, usize>,
    failures: Vec<FileFailure>,
}

fn replay_files(input: &Path, recursive: bool) -> anyhow::Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if !input.is_dir() {
        anyhow::bail!(
            "input {} is neither a file nor a directory",
            input.display()
        );
    }
    let max_depth = if recursive { usize::MAX } else { 1 };
    let mut files: Vec<_> = WalkDir::new(input)
        .min_depth(1)
        .max_depth(max_depth)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| is_canonical_replay_file(path))
        .collect();
    files.sort();
    Ok(files)
}

fn record_failure(
    summary: &mut Summary,
    path: &Path,
    stage: &'static str,
    error: impl ToString,
    version: Option<i32>,
) {
    let failure = FileFailure {
        file: path.display().to_string(),
        stage,
        version,
        error: error.to_string(),
    };
    eprintln!("INVALID [{}] {}: {}", stage, failure.file, failure.error);
    summary.invalid_files += 1;
    *summary
        .failures_by_version
        .entry(version.map_or_else(|| "unknown".to_string(), |version| version.to_string()))
        .or_default() += 1;
    summary.failures.push(failure);
}

fn execution_stage(error: &ReplayError) -> &'static str {
    match error {
        ReplayError::SourceDivergence { .. } => "sourceDivergence",
        _ => "execute",
    }
}

fn report_score_notes(summary: &mut Summary, path: &Path, verifier: &DivergenceVerifier) {
    for note in verifier.score_notes() {
        summary.source_score_drifts += 1;
        eprintln!("SOURCE-SCORE-DRIFT {}: {note}", path.display());
    }
}

fn write_report(path: Option<&Path>, summary: &Summary) -> anyhow::Result<()> {
    if let Some(path) = path {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, serde_json::to_vec_pretty(summary)?)?;
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let (common, export) = match &args.command {
        Command::Validate(common) => (common, None),
        Command::ExportTraining {
            common,
            output,
            samples_per_file,
        } => (common, Some((output.as_path(), *samples_per_file))),
    };
    let files = replay_files(&common.input, common.recursive)?;
    let mut summary = Summary {
        discovered_files: files.len(),
        ..Default::default()
    };
    let mut all_samples: Vec<TrainingSample> = Vec::new();

    for path in &files {
        let replay = match load_replay(path) {
            Ok(replay) => replay,
            Err(error) => {
                record_failure(&mut summary, path, "load", error, None);
                if common.fail_fast {
                    break;
                }
                continue;
            }
        };
        let version = Some(replay.initial_state.settings.version);
        let checkpoints = match replay.metadata.end_turn_checkpoints() {
            Ok(checkpoints) => checkpoints,
            Err(error) => {
                record_failure(&mut summary, path, "sourceDiagnostics", error, version);
                if common.fail_fast {
                    break;
                }
                continue;
            }
        };

        if export.is_some() {
            let eligibility =
                match validate_training_eligibility_with(&replay, common.allow_version_drift) {
                    Ok(eligibility) => eligibility,
                    Err(error) => {
                        record_failure(&mut summary, path, "trainingEligibility", error, version);
                        if common.fail_fast {
                            break;
                        }
                        continue;
                    }
                };
            if eligibility.version_support != VersionSupport::Supported {
                summary.version_drift_files += 1;
                eprintln!(
                    "WARN [versionDrift] {}: game version {} is outside the supported range {}..={}",
                    path.display(),
                    eligibility.game_version,
                    MIN_SUPPORTED_GAME_VERSION,
                    MAX_SUPPORTED_GAME_VERSION
                );
            }
            let mut observers = PairObserver(
                TrainingCollector::new_with(&replay, common.allow_version_drift)?,
                DivergenceVerifier::new(checkpoints),
            );
            let had_result = replay.result.is_some();
            match ReplayExecutor::execute_with_observer(&replay, &mut observers) {
                Ok(game) => {
                    let PairObserver(collector, verifier) = observers;
                    report_score_notes(&mut summary, path, &verifier);
                    match collector.finish(&game, replay.result.as_ref(), path) {
                        Ok(mut samples) => {
                            summary.valid_files += 1;
                            summary.training_eligible_files += 1;
                            summary.training_samples += samples.len();
                            if !had_result {
                                summary.derived_result_files += 1;
                                println!("DERIVED-RESULT {}", path.display());
                            }
                            all_samples.append(&mut samples);
                            println!(
                                "VALID {} ({} samples)",
                                path.display(),
                                replay.command_count()
                            );
                        }
                        Err(error) => {
                            record_failure(&mut summary, path, "trainingLabels", error, version);
                            if common.fail_fast {
                                break;
                            }
                        }
                    }
                }
                Err(error) => {
                    record_failure(&mut summary, path, execution_stage(&error), error, version);
                    if common.fail_fast {
                        break;
                    }
                }
            }
        } else {
            let mut verifier = DivergenceVerifier::new(checkpoints);
            match ReplayExecutor::execute_with_observer(&replay, &mut verifier) {
                Ok(_) => {
                    report_score_notes(&mut summary, path, &verifier);
                    summary.valid_files += 1;
                    println!(
                        "VALID {} ({} commands)",
                        path.display(),
                        replay.command_count()
                    );
                }
                Err(error) => {
                    record_failure(&mut summary, path, execution_stage(&error), error, version);
                    if common.fail_fast {
                        break;
                    }
                }
            }
        }
    }

    if let Some((output, samples_per_file)) = export {
        if !all_samples.is_empty() {
            summary.output_files = write_training_files(&all_samples, output, samples_per_file)?
                .into_iter()
                .map(|path| path.display().to_string())
                .collect();
        }
    }
    write_report(common.error_report.as_deref(), &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if summary.invalid_files > 0 {
        anyhow::bail!(
            "{} of {} replay files failed; see the summary or --error-report",
            summary.invalid_files,
            summary.discovered_files
        )
    }
    Ok(())
}
