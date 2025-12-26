use crate::cli::{build_config, Cli};
use crate::engine::{EngineControl, TestEngine};
use crate::model::{Phase, RunResult, TestEvent};
use anyhow::{Context, Result};
use crossterm::{
    event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::StreamExt;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Sparkline, Tabs},
    Terminal,
};
use std::{io, time::Duration, time::Instant};
use tokio::sync::mpsc;

mod widgets;

struct UiState {
    tab: usize,
    paused: bool,
    phase: Phase,
    info: String,

    dl_series: Vec<u64>,
    ul_series: Vec<u64>,
    idle_lat_series: Vec<u64>,
    loaded_dl_lat_series: Vec<u64>,
    loaded_ul_lat_series: Vec<u64>,

    // Time-series for charts (seconds since run start, value)
    run_start: Instant,
    dl_points: Vec<(f64, f64)>,
    ul_points: Vec<(f64, f64)>,
    idle_lat_points: Vec<(f64, f64)>,
    loaded_dl_lat_points: Vec<(f64, f64)>,
    loaded_ul_lat_points: Vec<(f64, f64)>,

    dl_mbps: f64,
    ul_mbps: f64,
    dl_avg_mbps: f64,
    ul_avg_mbps: f64,
    dl_bytes_total: u64,
    ul_bytes_total: u64,
    dl_phase_start: Option<Instant>,
    ul_phase_start: Option<Instant>,

    // Live latency samples for real-time stats
    idle_latency_samples: Vec<f64>,
    loaded_dl_latency_samples: Vec<f64>,
    loaded_ul_latency_samples: Vec<f64>,
    idle_latency_sent: u64,
    idle_latency_received: u64,
    loaded_dl_latency_sent: u64,
    loaded_dl_latency_received: u64,
    loaded_ul_latency_sent: u64,
    loaded_ul_latency_received: u64,

    last_result: Option<RunResult>,
    history: Vec<RunResult>,
    ip: Option<String>,
    colo: Option<String>,
    server: Option<String>,
    asn: Option<String>,
    as_org: Option<String>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            tab: 0,
            paused: false,
            phase: Phase::IdleLatency,
            info: String::new(),
            dl_series: Vec::new(),
            ul_series: Vec::new(),
            idle_lat_series: Vec::new(),
            loaded_dl_lat_series: Vec::new(),
            loaded_ul_lat_series: Vec::new(),
            run_start: Instant::now(),
            dl_points: Vec::new(),
            ul_points: Vec::new(),
            idle_lat_points: Vec::new(),
            loaded_dl_lat_points: Vec::new(),
            loaded_ul_lat_points: Vec::new(),
            dl_mbps: 0.0,
            ul_mbps: 0.0,
            dl_avg_mbps: 0.0,
            ul_avg_mbps: 0.0,
            dl_bytes_total: 0,
            ul_bytes_total: 0,
            dl_phase_start: None,
            ul_phase_start: None,
            idle_latency_samples: Vec::new(),
            loaded_dl_latency_samples: Vec::new(),
            loaded_ul_latency_samples: Vec::new(),
            idle_latency_sent: 0,
            idle_latency_received: 0,
            loaded_dl_latency_sent: 0,
            loaded_dl_latency_received: 0,
            loaded_ul_latency_sent: 0,
            loaded_ul_latency_received: 0,
            last_result: None,
            history: Vec::new(),
            ip: None,
            colo: None,
            server: None,
            asn: None,
            as_org: None,
        }
    }
}

impl UiState {
    fn push_series(series: &mut Vec<u64>, v: u64) {
        const MAX: usize = 120;
        series.push(v);
        if series.len() > MAX {
            let _ = series.drain(0..(series.len() - MAX));
        }
    }

    fn push_point(points: &mut Vec<(f64, f64)>, x: f64, y: f64) {
        const MAX: usize = 1200; // ~2 min at 10Hz
        points.push((x, y));
        if points.len() > MAX {
            let _ = points.drain(0..(points.len() - MAX));
        }
    }

    fn compute_live_latency_stats(
        samples: &[f64],
        sent: u64,
        received: u64,
    ) -> crate::model::LatencySummary {
        use hdrhistogram::Histogram;
        let loss = if sent == 0 {
            0.0
        } else {
            ((sent - received) as f64) / (sent as f64)
        };

        if samples.is_empty() {
            return crate::model::LatencySummary {
                sent,
                received,
                loss,
                min_ms: None,
                p50_ms: None,
                p90_ms: None,
                p99_ms: None,
                max_ms: None,
                jitter_ms: None,
            };
        }

        // Compute jitter (stddev) from samples
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let jitter_ms = Some(variance.sqrt());

        // Use HDRHistogram for percentiles
        let mut h = Histogram::<u64>::new_with_bounds(1, 60_000_000, 3).unwrap();
        for &ms in samples {
            let us = (ms * 1000.0).round().clamp(1.0, 60_000_000.0) as u64;
            let _ = h.record(us);
        }

        crate::model::LatencySummary {
            sent,
            received,
            loss,
            min_ms: Some((h.min() as f64) / 1000.0),
            p50_ms: Some((h.value_at_quantile(0.50) as f64) / 1000.0),
            p90_ms: Some((h.value_at_quantile(0.90) as f64) / 1000.0),
            p99_ms: Some((h.value_at_quantile(0.99) as f64) / 1000.0),
            max_ms: Some((h.max() as f64) / 1000.0),
            jitter_ms,
        }
    }
}

pub async fn run(args: Cli) -> Result<()> {
    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).ok();

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("create terminal")?;
    terminal.clear().ok();

    let mut state = UiState {
        phase: Phase::IdleLatency,
        ..Default::default()
    };
    state.history = crate::storage::load_recent(20).unwrap_or_default();

    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(100));

    // Start first run.
    let mut run_ctx = start_run(&args).await?;

    let res = loop {
        tokio::select! {
            _ = tick.tick() => {
                terminal.draw(|f| draw(f.area(), f, &state)).ok();
            }
            maybe_ev = events.next() => {
                let Some(Ok(ev)) = maybe_ev else { continue };
                if let Event::Key(k) = ev {
                    if k.kind != KeyEventKind::Press {
                        continue;
                    }
                    match (k.modifiers, k.code) {
                        (_, KeyCode::Char('q')) | (KeyModifiers::CONTROL, KeyCode::Char('c')) => {
                            run_ctx.ctrl_tx.send(EngineControl::Cancel).await.ok();
                            break Ok(());
                        }
                        (_, KeyCode::Char('p')) => {
                            state.paused = !state.paused;
                            run_ctx.ctrl_tx.send(EngineControl::Pause(state.paused)).await.ok();
                        }
                        (_, KeyCode::Char('r')) => {
                            state.info = "Restarting…".into();
                            run_ctx.ctrl_tx.send(EngineControl::Cancel).await.ok();
                            if let Some(h) = run_ctx.handle.take() {
                                let _ = h.await;
                            }
                            state.last_result = None;
                            state.run_start = Instant::now();
                            state.dl_series.clear();
                            state.ul_series.clear();
                            state.idle_lat_series.clear();
                            state.loaded_dl_lat_series.clear();
                            state.loaded_ul_lat_series.clear();
                            state.dl_points.clear();
                            state.ul_points.clear();
                            state.idle_lat_points.clear();
                            state.loaded_dl_lat_points.clear();
                            state.loaded_ul_lat_points.clear();
                            state.dl_mbps = 0.0;
                            state.ul_mbps = 0.0;
                            state.dl_avg_mbps = 0.0;
                            state.ul_avg_mbps = 0.0;
                            state.dl_bytes_total = 0;
                            state.ul_bytes_total = 0;
                            state.dl_phase_start = None;
                            state.ul_phase_start = None;
                            state.paused = false;
                            run_ctx = start_run(&args).await?;
                        }
                        (_, KeyCode::Char('s')) => {
                            if let Some(r) = state.last_result.as_ref() {
                                match save_result_json(r) {
                                    Ok(_) => {
                                        state.info = "Saved. (r rerun, q quit)".into();
                                    }
                                    Err(e) => {
                                        state.info = format!("Save failed: {e:#}");
                                    }
                                }
                            } else {
                                state.info = "No completed run to save yet.".into();
                            }
                        }
                        (_, KeyCode::Tab) => {
                            state.tab = (state.tab + 1) % 3;
                        }
                        (_, KeyCode::Char('?')) => {
                            state.tab = 2; // help
                        }
                        _ => {}
                    }
                }
            }
            maybe_engine_ev = run_ctx.event_rx.recv() => {
                match maybe_engine_ev {
                    None => {
                        // engine finished; wait for result
                        if let Some(h) = run_ctx.handle.take() {
                            match h.await {
                                Ok(Ok(r)) => {
                                    crate::storage::save_run(&r).ok();
                                    if let Some(meta) = r.meta.as_ref() {
                                        // Try multiple possible field names for IP
                                        state.ip = meta
                                            .get("clientIp")
                                            .or_else(|| meta.get("ip"))
                                            .or_else(|| meta.get("clientIP"))
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());
                                        state.colo = meta
                                            .get("colo")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());
                                        // Extract ASN and organization
                                        state.asn = meta
                                            .get("asn")
                                            .and_then(|v| v.as_i64())
                                            .map(|n| n.to_string())
                                            .or_else(|| meta.get("asn").and_then(|v| v.as_str()).map(|s| s.to_string()));
                                        state.as_org = meta
                                            .get("asOrganization")
                                            .or_else(|| meta.get("asnOrg"))
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());
                                    }
                                    // Server should be set from RunResult.server
                                    if r.server.is_some() {
                                        state.server = r.server.clone();
                                    }
                                    state.last_result = Some(r);
                                    state.history = crate::storage::load_recent(20).unwrap_or_default();
                                    state.info = "Done. (r rerun, q quit)".into();
                                }
                                Ok(Err(e)) => state.info = format!("Run failed: {e:#}"),
                                Err(e) => state.info = format!("Run join failed: {e}"),
                            }
                        }
                    }
                    Some(ev) => apply_event(&mut state, ev),
                }
            }
        }
    };

    // Restore terminal.
    disable_raw_mode().ok();
    let mut stdout = io::stdout();
    execute!(stdout, LeaveAlternateScreen).ok();
    res
}

struct RunCtx {
    ctrl_tx: mpsc::Sender<EngineControl>,
    event_rx: mpsc::Receiver<TestEvent>,
    handle: Option<tokio::task::JoinHandle<Result<RunResult>>>,
}

async fn start_run(args: &Cli) -> Result<RunCtx> {
    let cfg = build_config(args);
    let (event_tx, event_rx) = mpsc::channel::<TestEvent>(4096);
    let (ctrl_tx, ctrl_rx) = mpsc::channel::<EngineControl>(32);
    let engine = TestEngine::new(cfg);
    let handle = tokio::spawn(async move { engine.run(event_tx, ctrl_rx).await });
    Ok(RunCtx {
        ctrl_tx,
        event_rx,
        handle: Some(handle),
    })
}

fn apply_event(state: &mut UiState, ev: TestEvent) {
    match ev {
        TestEvent::PhaseStarted { phase } => {
            state.phase = phase;
            state.info = format!("Phase: {phase:?}");
            match phase {
                Phase::IdleLatency => {
                    // Reset idle latency tracking
                    state.idle_latency_samples.clear();
                    state.idle_latency_sent = 0;
                    state.idle_latency_received = 0;
                }
                Phase::Download => {
                    state.dl_phase_start = Some(Instant::now());
                    state.dl_bytes_total = 0;
                    state.dl_avg_mbps = 0.0;
                    // Reset loaded DL latency tracking
                    state.loaded_dl_latency_samples.clear();
                    state.loaded_dl_latency_sent = 0;
                    state.loaded_dl_latency_received = 0;
                }
                Phase::Upload => {
                    state.ul_phase_start = Some(Instant::now());
                    state.ul_bytes_total = 0;
                    state.ul_avg_mbps = 0.0;
                    // Reset loaded UL latency tracking
                    state.loaded_ul_latency_samples.clear();
                    state.loaded_ul_latency_sent = 0;
                    state.loaded_ul_latency_received = 0;
                }
                _ => {}
            }
        }
        TestEvent::Info { message } => state.info = message,
        TestEvent::MetaInfo { meta } => {
            // Extract IP and colo from meta
            state.ip = meta
                .get("clientIp")
                .or_else(|| meta.get("ip"))
                .or_else(|| meta.get("clientIP"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            state.colo = meta
                .get("colo")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            
            // Extract ASN and organization
            state.asn = meta
                .get("asn")
                .and_then(|v| {
                    if let Some(n) = v.as_i64() {
                        Some(n.to_string())
                    } else {
                        v.as_str().map(|s| s.to_string())
                    }
                });
            state.as_org = meta
                .get("asOrganization")
                .or_else(|| meta.get("asnOrg"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            
            // Extract city for server location (if available, use it directly)
            if let Some(city) = meta.get("city").and_then(|v| v.as_str()) {
                // If we have city, use it for server location
                // Could combine with country if available
                if let Some(country) = meta.get("country").and_then(|v| v.as_str()) {
                    state.server = Some(format!("{}, {}", city, country));
                } else {
                    state.server = Some(city.to_string());
                }
            } else if state.colo.is_some() && state.server.is_none() {
                // If we have colo but no city, we'll wait for RunResult.server
                // which comes from map_colo_to_server
            }
        }
        TestEvent::LatencySample { phase, during, rtt_ms, ok } => {
            let t = state.run_start.elapsed().as_secs_f64();
            match (phase, during) {
                (Phase::IdleLatency, _) => {
                    state.idle_latency_sent += 1;
                    if ok {
                        state.idle_latency_received += 1;
                        if let Some(ms) = rtt_ms {
                            let v = ms.round().clamp(0.0, 5000.0) as u64;
                            UiState::push_series(&mut state.idle_lat_series, v);
                            UiState::push_point(&mut state.idle_lat_points, t, ms);
                            state.idle_latency_samples.push(ms);
                            // Keep reasonable sample size
                            if state.idle_latency_samples.len() > 10000 {
                                state.idle_latency_samples.drain(0..(state.idle_latency_samples.len() - 10000));
                            }
                        }
                    }
                }
                (Phase::Download, Some(Phase::Download)) => {
                    state.loaded_dl_latency_sent += 1;
                    if ok {
                        state.loaded_dl_latency_received += 1;
                        if let Some(ms) = rtt_ms {
                            let v = ms.round().clamp(0.0, 5000.0) as u64;
                            UiState::push_series(&mut state.loaded_dl_lat_series, v);
                            UiState::push_point(&mut state.loaded_dl_lat_points, t, ms);
                            state.loaded_dl_latency_samples.push(ms);
                            if state.loaded_dl_latency_samples.len() > 10000 {
                                state.loaded_dl_latency_samples.drain(0..(state.loaded_dl_latency_samples.len() - 10000));
                            }
                        }
                    }
                }
                (Phase::Upload, Some(Phase::Upload)) => {
                    state.loaded_ul_latency_sent += 1;
                    if ok {
                        state.loaded_ul_latency_received += 1;
                        if let Some(ms) = rtt_ms {
                            let v = ms.round().clamp(0.0, 5000.0) as u64;
                            UiState::push_series(&mut state.loaded_ul_lat_series, v);
                            UiState::push_point(&mut state.loaded_ul_lat_points, t, ms);
                            state.loaded_ul_latency_samples.push(ms);
                            if state.loaded_ul_latency_samples.len() > 10000 {
                                state.loaded_ul_latency_samples.drain(0..(state.loaded_ul_latency_samples.len() - 10000));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        TestEvent::ThroughputTick { phase, bytes_total, bps_instant } => {
            let mbps = (bps_instant * 8.0) / 1_000_000.0;
            let t = state.run_start.elapsed().as_secs_f64();
            match phase {
                Phase::Download => {
                    state.dl_mbps = mbps;
                    state.dl_bytes_total = bytes_total;
                    if let Some(t0) = state.dl_phase_start {
                        let secs = t0.elapsed().as_secs_f64().max(1e-9);
                        state.dl_avg_mbps = ((bytes_total as f64) / secs) * 8.0 / 1_000_000.0;
                    }
                    let v = state.dl_mbps.round().clamp(0.0, 10_000.0) as u64;
                    UiState::push_series(&mut state.dl_series, v);
                    UiState::push_point(&mut state.dl_points, t, state.dl_mbps.max(0.0));
                }
                Phase::Upload => {
                    state.ul_mbps = mbps;
                    state.ul_bytes_total = bytes_total;
                    if let Some(t0) = state.ul_phase_start {
                        let secs = t0.elapsed().as_secs_f64().max(1e-9);
                        state.ul_avg_mbps = ((bytes_total as f64) / secs) * 8.0 / 1_000_000.0;
                    }
                    let v = state.ul_mbps.round().clamp(0.0, 10_000.0) as u64;
                    UiState::push_series(&mut state.ul_series, v);
                    UiState::push_point(&mut state.ul_points, t, state.ul_mbps.max(0.0));
                }
                _ => {}
            }
        }
    }
}

fn draw(area: Rect, f: &mut ratatui::Frame, state: &UiState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)].as_ref())
        .split(area);

    let tabs = Tabs::new(vec![
        Line::from("Dashboard"),
        Line::from("History"),
        Line::from("Help"),
    ])
        .select(state.tab)
        .block(Block::default().borders(Borders::ALL).title("cloudflare-speed-cli"))
        .highlight_style(Style::default().fg(Color::Yellow));
    f.render_widget(tabs, chunks[0]);

    match state.tab {
        0 => draw_dashboard(chunks[1], f, state),
        1 => draw_history(chunks[1], f, state),
        _ => draw_help(chunks[1], f),
    }
}

fn draw_dashboard(area: Rect, f: &mut ratatui::Frame, state: &UiState) {
    // Small terminal: keep the compact dashboard (gauges + sparklines).
    // Large terminal: show full charts (like the website) alongside the live cards.
    if area.height < 28 {
        return draw_dashboard_compact(area, f, state);
    }

    let main = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Length(3),  // Speed boxes row (side-by-side)
                Constraint::Length(12), // Throughput charts row (side-by-side)
                Constraint::Length(8),  // Latency stats row (idle + loaded DL + loaded UL)
                Constraint::Min(0),     // Status + shortcuts
            ]
            .as_ref(),
        )
        .split(area);

    // Speed boxes at the top: DL left, UL right
    let speed_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(main[0]);
    
    // Download speed box
    let dl_info = Paragraph::new(Line::from(vec![
        Span::styled(format!("inst {:.1} / avg {:.1}", state.dl_mbps, state.dl_avg_mbps), Style::default().fg(Color::Green)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green))
            .title("Download Speed")
    );
    f.render_widget(dl_info, speed_row[0]);
    
    // Upload speed box
    let ul_info = Paragraph::new(Line::from(vec![
        Span::styled(format!("inst {:.1} / avg {:.1}", state.ul_mbps, state.ul_avg_mbps), Style::default().fg(Color::Cyan)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title("Upload Speed")
    );
    f.render_widget(ul_info, speed_row[1]);

    // Throughput charts side-by-side: DL left, UL right
    let thr_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(main[1]);

    let (_x_min, x_max) = x_bounds(&state.dl_points, &state.ul_points);
    
    // Download throughput chart (left)
    let y_dl_max = max_y(&state.dl_points).max(10.0);
    let y_dl_max = (y_dl_max * 1.10).min(10_000.0);
    let dl_ds = Dataset::default()
        .graph_type(GraphType::Line)
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(Color::Green))
        .data(window(&state.dl_points, x_max, 60.0));
    let dl_chart = Chart::new(vec![dl_ds])
        .block(Block::default().borders(Borders::ALL).title("Download Throughput (last 60s)"))
        .x_axis(
            Axis::default()
                .title("s")
                .bounds([(x_max - 60.0).max(0.0), x_max.max(1.0)]),
        )
        .y_axis(Axis::default().title("Mbps").bounds([0.0, y_dl_max]));
    f.render_widget(dl_chart, thr_row[0]);

    // Upload throughput chart (right)
    let y_ul_max = max_y(&state.ul_points).max(10.0);
    let y_ul_max = (y_ul_max * 1.10).min(10_000.0);
    let ul_ds = Dataset::default()
        .graph_type(GraphType::Line)
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(Color::Cyan))
        .data(window(&state.ul_points, x_max, 60.0));
    let ul_chart = Chart::new(vec![ul_ds])
        .block(Block::default().borders(Borders::ALL).title("Upload Throughput (last 60s)"))
        .x_axis(
            Axis::default()
                .title("s")
                .bounds([(x_max - 60.0).max(0.0), x_max.max(1.0)]),
        )
        .y_axis(Axis::default().title("Mbps").bounds([0.0, y_ul_max]));
    f.render_widget(ul_chart, thr_row[1]);

    // Latency stats (numeric, not charts): Idle, Loaded DL, Loaded UL
    let lat_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(33), Constraint::Percentage(33), Constraint::Percentage(34)].as_ref())
        .split(main[2]);

    // Helper to format latency stats
    let format_latency = |lat: &crate::model::LatencySummary| -> Vec<Line> {
        vec![
            Line::from(vec![
                Span::styled("p50: ", Style::default().fg(Color::Gray)),
                Span::raw(format!("{:.1} ms", lat.p50_ms.unwrap_or(f64::NAN))),
            ]),
            Line::from(vec![
                Span::styled("p90: ", Style::default().fg(Color::Gray)),
                Span::raw(format!("{:.1} ms", lat.p90_ms.unwrap_or(f64::NAN))),
            ]),
            Line::from(vec![
                Span::styled("p99: ", Style::default().fg(Color::Gray)),
                Span::raw(format!("{:.1} ms", lat.p99_ms.unwrap_or(f64::NAN))),
            ]),
            Line::from(vec![
                Span::styled("Jitter: ", Style::default().fg(Color::Gray)),
                Span::raw(format!("{:.1} ms", lat.jitter_ms.unwrap_or(f64::NAN))),
            ]),
            Line::from(vec![
                Span::styled("Loss: ", Style::default().fg(Color::Gray)),
                Span::raw(format!("{:.2}%", lat.loss * 100.0)),
            ]),
        ]
    };

    // Idle latency stats (live from samples)
    let idle_lat = if state.idle_latency_samples.is_empty() && state.idle_latency_sent == 0 {
        None
    } else {
        Some(UiState::compute_live_latency_stats(
            &state.idle_latency_samples,
            state.idle_latency_sent,
            state.idle_latency_received,
        ))
    };
    let idle_stats = Paragraph::new(
        idle_lat.as_ref().map(format_latency).unwrap_or_else(|| {
            vec![Line::from("Waiting for data...")]
        })
    )
    .block(Block::default().borders(Borders::ALL).title("Idle Latency"));
    f.render_widget(idle_stats, lat_row[0]);

    // Loaded latency during download stats (live from samples)
    let dl_loaded_lat = if state.loaded_dl_latency_samples.is_empty() && state.loaded_dl_latency_sent == 0 {
        None
    } else {
        Some(UiState::compute_live_latency_stats(
            &state.loaded_dl_latency_samples,
            state.loaded_dl_latency_sent,
            state.loaded_dl_latency_received,
        ))
    };
    let dl_loaded_stats = Paragraph::new(
        dl_loaded_lat.as_ref().map(format_latency).unwrap_or_else(|| {
            vec![Line::from("Waiting for data...")]
        })
    )
    .block(Block::default().borders(Borders::ALL).title("Loaded Latency (Download)"));
    f.render_widget(dl_loaded_stats, lat_row[1]);

    // Loaded latency during upload stats (live from samples)
    let ul_loaded_lat = if state.loaded_ul_latency_samples.is_empty() && state.loaded_ul_latency_sent == 0 {
        None
    } else {
        Some(UiState::compute_live_latency_stats(
            &state.loaded_ul_latency_samples,
            state.loaded_ul_latency_sent,
            state.loaded_ul_latency_received,
        ))
    };
    let ul_loaded_stats = Paragraph::new(
        ul_loaded_lat.as_ref().map(format_latency).unwrap_or_else(|| {
            vec![Line::from("Waiting for data...")]
        })
    )
    .block(Block::default().borders(Borders::ALL).title("Loaded Latency (Upload)"));
    f.render_widget(ul_loaded_stats, lat_row[2]);

    // Combined Status and Controls panel
    let saved_path = state.last_result.as_ref()
        .and_then(|r| crate::storage::get_run_path(r).ok());
    
    // Determine IP version
    let ip_version = state.ip.as_deref()
        .map(|ip| if ip.contains(':') { "IPv6" } else { "IPv4" })
        .unwrap_or("-");
    
    let combined = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Phase: ", Style::default().fg(Color::Gray)),
            Span::raw(format!("{:?}", state.phase)),
            Span::raw("   "),
            Span::styled("Paused: ", Style::default().fg(Color::Gray)),
            Span::raw(format!("{}", state.paused)),
        ]),
        Line::from(vec![
            Span::styled("Connected via: ", Style::default().fg(Color::Gray)),
            Span::raw(ip_version),
        ]),
        Line::from(vec![
            Span::styled("Server location: ", Style::default().fg(Color::Gray)),
            Span::raw(state.server.as_deref().unwrap_or("-")),
        ]),
        Line::from(vec![
            Span::styled("Your network: ", Style::default().fg(Color::Gray)),
            Span::raw(
                match (state.as_org.as_deref(), state.asn.as_deref()) {
                    (Some(org), Some(asn)) => format!("{} (AS{})", org, asn),
                    (Some(org), None) => org.to_string(),
                    (None, Some(asn)) => format!("AS{}", asn),
                    (None, None) => "-".to_string(),
                }
            ),
        ]),
        Line::from(vec![
            Span::styled("Your IP address: ", Style::default().fg(Color::Gray)),
            Span::raw(state.ip.as_deref().unwrap_or("-")),
        ]),
        Line::from(vec![
            Span::styled("Info: ", Style::default().fg(Color::Gray)),
            Span::raw(&state.info),
        ]),
        Line::from(""),
        Line::from("Keyboard Shortcuts:"),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("q", Style::default().fg(Color::Magenta)),
            Span::raw(" / "),
            Span::styled("Ctrl-C", Style::default().fg(Color::Magenta)),
            Span::raw("  Quit"),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("r", Style::default().fg(Color::Magenta)),
            Span::raw("           Rerun test"),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("p", Style::default().fg(Color::Magenta)),
            Span::raw("           Pause/Resume"),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("s", Style::default().fg(Color::Magenta)),
            Span::raw("           Save JSON"),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("tab", Style::default().fg(Color::Magenta)),
            Span::raw("         Switch tabs"),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("?", Style::default().fg(Color::Magenta)),
            Span::raw("           Help"),
        ]),
        Line::from(vec![
            Span::styled("Saved JSON: ", Style::default().fg(Color::Gray)),
            Span::raw(
                saved_path
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .unwrap_or("none")
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Source: ", Style::default().fg(Color::Gray)),
            Span::styled("https://speed.cloudflare.com/", Style::default().fg(Color::Blue)),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).title("Status & Controls"));
    f.render_widget(combined, main[3]);
}

fn draw_dashboard_compact(area: Rect, f: &mut ratatui::Frame, state: &UiState) {
    let content = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(8)].as_ref())
        .split(area);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(content[0]);

    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3), Constraint::Min(0)].as_ref())
        .split(cols[0]);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3), Constraint::Min(0)].as_ref())
        .split(cols[1]);

    // Download sparkline with inst/avg box below
    f.render_widget(
        Sparkline::default()
            .block(Block::default().borders(Borders::ALL).title("DL sparkline (Mbps)"))
            .data(&state.dl_series)
            .style(Style::default().fg(Color::Green)),
        left[0],
    );
    let dl_info = Paragraph::new(Line::from(vec![
        Span::styled(format!("inst {:.1} / avg {:.1}", state.dl_mbps, state.dl_avg_mbps), Style::default().fg(Color::Green)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green))
            .title("Download Speed")
    );
    f.render_widget(dl_info, left[1]);

    // Upload sparkline with inst/avg box below
    f.render_widget(
        Sparkline::default()
            .block(Block::default().borders(Borders::ALL).title("UL sparkline (Mbps)"))
            .data(&state.ul_series)
            .style(Style::default().fg(Color::Cyan)),
        right[0],
    );
    let ul_info = Paragraph::new(Line::from(vec![
        Span::styled(format!("inst {:.1} / avg {:.1}", state.ul_mbps, state.ul_avg_mbps), Style::default().fg(Color::Cyan)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title("Upload Speed")
    );
    f.render_widget(ul_info, right[1]);

    f.render_widget(
        Sparkline::default()
            .block(Block::default().borders(Borders::ALL).title("Idle latency (ms)"))
            .data(&state.idle_lat_series)
            .style(Style::default().fg(Color::Magenta)),
        right[2],
    );

    let meta = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Phase: ", Style::default().fg(Color::Gray)),
            Span::raw(format!("{:?}", state.phase)),
            Span::raw("   "),
            Span::styled("Paused: ", Style::default().fg(Color::Gray)),
            Span::raw(format!("{}", state.paused)),
        ]),
        Line::from(vec![
            Span::styled("IP/Colo: ", Style::default().fg(Color::Gray)),
            Span::raw(format!(
                "{} / {}",
                state.ip.as_deref().unwrap_or("-"),
                state.colo.as_deref().unwrap_or("-")
            )),
        ]),
        Line::from(vec![
            Span::styled("Info: ", Style::default().fg(Color::Gray)),
            Span::raw(&state.info),
        ]),
        Line::from(vec![
            Span::styled("Server: ", Style::default().fg(Color::Gray)),
            Span::raw(state.server.as_deref().unwrap_or("-")),
        ]),
        Line::from(""),
        Line::from("Keys: q quit | r rerun | p pause | s save json | tab switch | ? help"),
    ])
    .block(Block::default().borders(Borders::ALL).title("Status"));
    f.render_widget(meta, content[1]);
}

fn draw_help(area: Rect, f: &mut ratatui::Frame) {
    let p = Paragraph::new(vec![
        Line::from("Keybinds:"),
        Line::from("  q / Ctrl-C  Quit"),
        Line::from("  r           Rerun"),
        Line::from("  p           Pause/Resume"),
        Line::from("  s           Save last run JSON"),
        Line::from("  tab         Switch tabs"),
        Line::from("  ?           Show this help"),
        Line::from(""),
    ])
    .block(Block::default().borders(Borders::ALL).title("Help"));
    f.render_widget(p, area);
}

fn save_result_json(r: &RunResult) -> Result<std::path::PathBuf> {
    crate::storage::save_run(r)
}

fn draw_history(area: Rect, f: &mut ratatui::Frame, state: &UiState) {
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from("Most recent runs:"));
    lines.push(Line::from(""));
    for r in state.history.iter().take(10).rev() {
        // Parse and format timestamp to human-readable format (yyyy-mm-dd hh:mm:ss)
        // Try parsing RFC3339 format (e.g., "2024-12-19T17:30:45Z")
        let timestamp_str: String = {
            // Simple manual parsing for RFC3339 format
            let s = &r.timestamp_utc;
            if s.len() >= 19 && s.contains('T') {
                // Extract "2024-12-19T17:30:45" part
                let date_time: String = s.chars().take(19).collect();
                if let Some(t_pos) = date_time.find('T') {
                    let date_part = &date_time[..t_pos];
                    let time_part = &date_time[t_pos+1..];
                    format!("{} {}", date_part, time_part)
                } else {
                    r.timestamp_utc.clone()
                }
            } else {
                r.timestamp_utc.clone()
            }
        };
        
        lines.push(Line::from(vec![
            Span::styled(timestamp_str, Style::default().fg(Color::Gray)),
            Span::raw("  "),
            Span::styled(format!("DL {:>7.2} Mbps", r.download.mbps), Style::default().fg(Color::Green)),
            Span::raw("  "),
            Span::styled(format!("UL {:>7.2} Mbps", r.upload.mbps), Style::default().fg(Color::Cyan)),
            Span::raw("  "),
            Span::raw(format!("Idle p50 {:>6.1} ms", r.idle_latency.p50_ms.unwrap_or(f64::NAN))),
        ]));
    }
    let p = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("History"));
    f.render_widget(p, area);
}

fn max_y(points: &[(f64, f64)]) -> f64 {
    points
        .iter()
        .map(|(_, y)| *y)
        .fold(0.0, |a, b| a.max(b))
}

fn x_bounds(a: &[(f64, f64)], b: &[(f64, f64)]) -> (f64, f64) {
    let xmax = a
        .last()
        .map(|(x, _)| *x)
        .unwrap_or(0.0)
        .max(b.last().map(|(x, _)| *x).unwrap_or(0.0));
    let xmin = 0.0;
    (xmin, xmax)
}

fn window<'a>(points: &'a [(f64, f64)], x_max: f64, secs: f64) -> &'a [(f64, f64)] {
    let x0 = (x_max - secs).max(0.0);
    let idx = points
        .iter()
        .position(|(x, _)| *x >= x0)
        .unwrap_or(points.len());
    &points[idx..]
}


