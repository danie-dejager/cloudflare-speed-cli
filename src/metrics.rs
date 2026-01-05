/// Compute metrics (mean, median, 25th percentile, 75th percentile) from samples.
/// Takes a slice to avoid unnecessary allocations; sorts a temporary copy internally.
pub fn compute_metrics(samples: &[f64]) -> Option<(f64, f64, f64, f64)> {
    if samples.len() < 2 {
        return None;
    }
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;

    // Sort a copy for percentile calculations
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = sorted[n / 2];
    let p25 = sorted[n / 4];
    let p75 = sorted[3 * n / 4];
    Some((mean, median, p25, p75))
}

/// Compute jitter (standard deviation) from latency samples.
pub fn compute_jitter(samples: &[f64]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Some(variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_metrics_basic() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, median, p25, p75) = compute_metrics(&samples).unwrap();
        assert!((mean - 3.0).abs() < 0.001);
        assert!((median - 3.0).abs() < 0.001);
        assert!((p25 - 2.0).abs() < 0.001);
        assert!((p75 - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_metrics_insufficient_samples() {
        assert!(compute_metrics(&[1.0]).is_none());
        assert!(compute_metrics(&[]).is_none());
    }

    #[test]
    fn test_compute_metrics_two_samples() {
        let samples = vec![10.0, 20.0];
        let result = compute_metrics(&samples);
        assert!(result.is_some());
        let (mean, _, _, _) = result.unwrap();
        assert!((mean - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_metrics_unsorted_input() {
        let samples = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let (mean, median, p25, p75) = compute_metrics(&samples).unwrap();
        assert!((mean - 3.0).abs() < 0.001);
        assert!((median - 3.0).abs() < 0.001);
        assert!((p25 - 2.0).abs() < 0.001);
        assert!((p75 - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_jitter_basic() {
        // samples: [1, 2, 3, 4, 5], mean = 3, variance = 10/4 = 2.5, stddev = sqrt(2.5) ≈ 1.58
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let jitter = compute_jitter(&samples).unwrap();
        assert!((jitter - 1.5811).abs() < 0.001);
    }

    #[test]
    fn test_compute_jitter_insufficient_samples() {
        assert!(compute_jitter(&[1.0]).is_none());
        assert!(compute_jitter(&[]).is_none());
    }
}
