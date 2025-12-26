use crate::model::LatencySummary;
use hdrhistogram::Histogram;

#[derive(Debug, Default, Clone)]
pub struct OnlineStats {
    n: u64,
    mean: f64,
    m2: f64,
}

impl OnlineStats {
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / (self.n as f64);
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn stddev(&self) -> Option<f64> {
        if self.n < 2 {
            None
        } else {
            Some((self.m2 / ((self.n - 1) as f64)).sqrt())
        }
    }
}

pub fn latency_summary_from_samples(
    sent: u64,
    received: u64,
    samples_ms: &[f64],
    jitter_ms: Option<f64>,
) -> LatencySummary {
    let loss = if sent == 0 {
        0.0
    } else {
        ((sent - received) as f64) / (sent as f64)
    };

    if samples_ms.is_empty() {
        return LatencySummary {
            sent,
            received,
            loss,
            min_ms: None,
            p50_ms: None,
            p90_ms: None,
            p99_ms: None,
            max_ms: None,
            jitter_ms,
        };
    }

    // HDRHistogram wants integer values; store microseconds to preserve precision.
    let mut h = Histogram::<u64>::new_with_bounds(1, 60_000_000, 3).unwrap();
    for &ms in samples_ms {
        let us = (ms * 1000.0).round().clamp(1.0, 60_000_000.0) as u64;
        let _ = h.record(us);
    }

    let min_ms = Some((h.min() as f64) / 1000.0);
    let max_ms = Some((h.max() as f64) / 1000.0);
    let p50_ms = Some((h.value_at_quantile(0.50) as f64) / 1000.0);
    let p90_ms = Some((h.value_at_quantile(0.90) as f64) / 1000.0);
    let p99_ms = Some((h.value_at_quantile(0.99) as f64) / 1000.0);

    LatencySummary {
        sent,
        received,
        loss,
        min_ms,
        p50_ms,
        p90_ms,
        p99_ms,
        max_ms,
        jitter_ms,
    }
}


