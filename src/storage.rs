use crate::model::RunResult;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

fn base_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("cloudflare-speed-cli")
}

fn runs_dir() -> PathBuf {
    base_dir().join("runs")
}

pub fn ensure_dirs() -> Result<()> {
    std::fs::create_dir_all(runs_dir()).context("create runs dir")?;
    Ok(())
}

pub fn save_run(result: &RunResult) -> Result<PathBuf> {
    ensure_dirs()?;
    let path = get_run_path(result)?;
    let data = serde_json::to_vec_pretty(result)?;
    std::fs::write(&path, data).context("write run json")?;
    Ok(path)
}

pub fn get_run_path(result: &RunResult) -> Result<PathBuf> {
    let ts = &result.timestamp_utc;
    let safe_ts = ts
        .replace(':', "-")
        .replace('T', "_")
        .replace('Z', "Z");
    Ok(runs_dir().join(format!("run-{safe_ts}-{}.json", result.meas_id)))
}

pub fn export_json(path: &Path, result: &RunResult) -> Result<()> {
    let data = serde_json::to_vec_pretty(result)?;
    std::fs::write(path, data).context("write export json")?;
    Ok(())
}

pub fn export_csv(path: &Path, result: &RunResult) -> Result<()> {
    let mut out = String::new();
    out.push_str("timestamp_utc,base_url,meas_id,server,download_mbps,upload_mbps,idle_p50_ms,idle_p90_ms,idle_p99_ms,idle_loss,dl_loaded_p50_ms,dl_loaded_p90_ms,dl_loaded_p99_ms,dl_loaded_loss,ul_loaded_p50_ms,ul_loaded_p90_ms,ul_loaded_p99_ms,ul_loaded_loss\n");
    out.push_str(&format!(
        "{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6},{:.3},{:.3},{:.3},{:.6},{:.3},{:.3},{:.3},{:.6}\n",
        csv_escape(&result.timestamp_utc),
        csv_escape(&result.base_url),
        csv_escape(&result.meas_id),
        csv_escape(result.server.as_deref().unwrap_or("")),
        result.download.mbps,
        result.upload.mbps,
        result.idle_latency.p50_ms.unwrap_or(f64::NAN),
        result.idle_latency.p90_ms.unwrap_or(f64::NAN),
        result.idle_latency.p99_ms.unwrap_or(f64::NAN),
        result.idle_latency.loss,
        result.loaded_latency_download.p50_ms.unwrap_or(f64::NAN),
        result.loaded_latency_download.p90_ms.unwrap_or(f64::NAN),
        result.loaded_latency_download.p99_ms.unwrap_or(f64::NAN),
        result.loaded_latency_download.loss,
        result.loaded_latency_upload.p50_ms.unwrap_or(f64::NAN),
        result.loaded_latency_upload.p90_ms.unwrap_or(f64::NAN),
        result.loaded_latency_upload.p99_ms.unwrap_or(f64::NAN),
        result.loaded_latency_upload.loss,
    ));
    std::fs::write(path, out).context("write export csv")?;
    Ok(())
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

pub fn load_recent(limit: usize) -> Result<Vec<RunResult>> {
    ensure_dirs()?;
    let dir = runs_dir();
    let mut entries: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    for e in std::fs::read_dir(&dir).context("read runs dir")? {
        let e = e?;
        let p = e.path();
        if p.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let m = e.metadata()?;
        let mt = m.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        entries.push((mt, p));
    }
    entries.sort_by_key(|(t, _)| *t);
    entries.reverse();

    let mut out = Vec::new();
    for (_, p) in entries.into_iter().take(limit) {
        let data = std::fs::read(&p).with_context(|| format!("read {}", p.display()))?;
        let r: RunResult = serde_json::from_slice(&data).with_context(|| format!("parse {}", p.display()))?;
        out.push(r);
    }
    Ok(out)
}



