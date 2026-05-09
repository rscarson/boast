use crate::experiments::export_results;
use rand::{Rng, rng};

/// Experiment to model the effect of our 2 high-level parameters (ratio_pass and p_s) on bias
///
/// The results should that worse-case bias scales roughly linearly with ratio_pass, with p_fail having no measurable effect until ratio_pass is very close to 1.0
pub fn experiment_1() {
    const FILENAME: &str = "bias_experiment1.csv";
    const HEADINGS: [&str; 10] = [
        "p_fail",
        "meanmean_p",
        "meanstdev_p",
        "stdevmean_p",
        "stdevstdev_p",
        "meanmean_d",
        "meanstdev_d",
        "stdevmean_d",
        "stdevstdev_d",
        "raw_decision_bias_stdevstdev",
    ];

    let ratio_passs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99];
    let mut results = Vec::new();
    println!("Starting experiment 1");
    // Since $p_{\text{fail}} = 1 - (1 - p')^n$
    // and $p' = 1 / (1 + C \cdot n \cdot p)$ where C=1.864
    // and p == 1e-6
    // We can calculate n to get a set of p_fail (0.05, 0.1, 0.15, ..., 0.5)
    let p_fail_values = (5..=50).map(|x| x as f64 / 100.0);
    for p_fail in p_fail_values {
        let n = (-(1.0 - p_fail).ln() / (1.0 / (1.0 + 1.864 * 1e-6 * 1.0))).ceil() as usize;

        let mut points = Vec::new();
        for ratio_pass in ratio_passs {
            let result = trial(ratio_pass, n, 1.0, 1e-6);

            let mean_prediction_bias = result.prediction_bias.iter().cloned().sum::<f64>()
                / (result.prediction_bias.len() as f64);
            let stdev_prediction_bias = (result
                .prediction_bias
                .iter()
                .cloned()
                .map(|b| (b - mean_prediction_bias).powi(2))
                .sum::<f64>()
                / (result.prediction_bias.len() as f64))
                .sqrt();

            let mean_decision_bias = result.decision_bias.iter().cloned().sum::<f64>()
                / (result.decision_bias.len() as f64);
            let stdev_decision_bias = (result
                .decision_bias
                .iter()
                .cloned()
                .map(|b| (b - mean_decision_bias).powi(2))
                .sum::<f64>()
                / (result.decision_bias.len() as f64))
                .sqrt();

            // Now also get the stdev of the raw decision bias
            let mean_raw_decision_bias = result.raw_decision_bias.iter().cloned().sum::<f64>()
                / (result.raw_decision_bias.len() as f64);
            let stdev_raw_decision_bias = (result
                .raw_decision_bias
                .iter()
                .cloned()
                .map(|b| (b - mean_raw_decision_bias).powi(2))
                .sum::<f64>()
                / (result.raw_decision_bias.len() as f64))
                .sqrt();

            points.push((
                mean_prediction_bias,
                stdev_prediction_bias,
                mean_decision_bias,
                stdev_decision_bias,
                stdev_raw_decision_bias,
            ));
        }

        // Get the mean-mean, the mean-stdev, the stddev-mean, and the stddev-stddev
        // That is, the mean of all means, the stdev of all means, etc
        let mean_mean_pred = points.iter().map(|p| p.0).sum::<f64>() / (points.len() as f64);
        let mean_stdev_pred = (points
            .iter()
            .map(|p| (p.0 - mean_mean_pred).powi(2))
            .sum::<f64>()
            / (points.len() as f64))
            .sqrt();
        let stdev_mean_pred = points.iter().map(|p| p.1).sum::<f64>() / (points.len() as f64);
        let stdev_stdev_pred = (points
            .iter()
            .map(|p| (p.1 - stdev_mean_pred).powi(2))
            .sum::<f64>()
            / (points.len() as f64))
            .sqrt();

        // Now for decision bias
        let mean_mean_dec = points.iter().map(|p| p.2).sum::<f64>() / (points.len() as f64);
        let mean_stdev_dec = (points
            .iter()
            .map(|p| (p.2 - mean_mean_dec).powi(2))
            .sum::<f64>()
            / (points.len() as f64))
            .sqrt();
        let stdev_mean_dec = points.iter().map(|p| p.3).sum::<f64>() / (points.len() as f64);
        let stdev_stdev_dec = (points
            .iter()
            .map(|p| (p.3 - stdev_mean_dec).powi(2))
            .sum::<f64>()
            / (points.len() as f64))
            .sqrt();

        // And now for the raw decision bias stdev stdev
        let mean_stdev_raw_decision_bias =
            points.iter().map(|p| p.4).sum::<f64>() / (points.len() as f64);
        let stdev_stdev_raw_decision_bias = (points
            .iter()
            .map(|p| (p.4 - mean_stdev_raw_decision_bias).powi(2))
            .sum::<f64>()
            / (points.len() as f64))
            .sqrt();

        results.push([
            p_fail,
            mean_mean_pred,
            mean_stdev_pred,
            stdev_mean_pred,
            stdev_stdev_pred,
            mean_mean_dec,
            mean_stdev_dec,
            stdev_mean_dec,
            stdev_stdev_dec,
            stdev_stdev_raw_decision_bias,
        ]);
    }

    export_results(FILENAME, HEADINGS, &results);
    println!("Done; Exporting results to {FILENAME}");
}

/// Now that we have seen the lack of link between the input parameters and bias, we can do a simpler experiment against p_s alone
pub fn experiment_2_inner() -> Vec<[f64; 3]> {
    let (mut p_s_mul, p_s_max) = (0.0, 12.0);
    let mut results = Vec::new();
    while p_s_mul <= p_s_max {
        let p_s = 10f64.powf(p_s_mul);
        p_s_mul += 0.5;

        let result = trial(0.8, 100_000, p_s, 1e-6);
        let mean_bias = result.prediction_bias.iter().cloned().sum::<f64>()
            / (result.prediction_bias.len() as f64);
        let stdev_bias = (result
            .prediction_bias
            .iter()
            .cloned()
            .map(|b| (b - mean_bias).powi(2))
            .sum::<f64>()
            / (result.prediction_bias.len() as f64))
            .sqrt();

        results.push([p_s, mean_bias, stdev_bias]);
    }

    results
}

/// Now that we have seen the lack of link between the input parameters and bias, we can do a simpler experiment against p_s alone
pub fn experiment_2() {
    const FILENAME: &str = "bias_experiment2.csv";
    const HEADINGS: [&str; 3] = ["p_s", "mean_bias", "stdev_bias"];

    println!("Starting experiment 2");
    let results = experiment_2_inner();

    println!("Exporting results to {FILENAME}");
    export_results(FILENAME, HEADINGS, &results);
}

/// Now we explore the effect of p_s on the number of iterations performed in each simulation
pub fn experiment_3_inner() -> Vec<[f64; 3]> {
    let (mut p_s_mul, p_s_max) = (0.0, 12.0);
    let mut results = Vec::new();
    while p_s_mul <= p_s_max {
        let p_s = 10f64.powf(p_s_mul);
        p_s_mul += 0.5;

        let result = trial(0.8, 100_000, p_s, 1e-6);
        let iteration_fractions = result
            .iterations
            .iter()
            .map(|it| it / (result.initial_iterations as f64));
        let iterations: Vec<f64> = iteration_fractions.collect();

        let mean_iterations = iterations.iter().cloned().sum::<f64>() / (iterations.len() as f64);
        let stdev_iterations = (iterations
            .iter()
            .cloned()
            .map(|b| (b - mean_iterations).powi(2))
            .sum::<f64>()
            / (iterations.len() as f64))
            .sqrt();

        results.push([p_s, mean_iterations, stdev_iterations]);
    }

    results
}

/// Now we explore the effect of p_s on the number of iterations performed in each simulation
pub fn experiment_3() {
    const FILENAME: &str = "bias_experiment3.csv";
    const HEADINGS: [&str; 3] = ["p_s", "mean_iterations", "stdev_iterations"];

    println!("Starting experiment 3");
    let results = experiment_3_inner();

    println!("Done; Exporting results to {FILENAME}");
    export_results(FILENAME, HEADINGS, &results);
}

/// Now - experiments 2 and 3 are doing a curve for against the standard error of bias
/// So we need to run the whole experiment ~20 to 100 times to get an error estimate on that standard error
///
/// This will produce a table of [p_s, mean_stdev_bias, stdev_stdev_bias, mean_mean_iterations, stdev_mean_iterations]
pub fn get_error_for_2_3() {
    const TRIALS: usize = 50;
    const HEADINGS: [&str; 5] = [
        "p_s",
        "mean_stdev_bias",
        "stdev_stdev_bias",
        "mean_mean_iterations",
        "stdev_mean_iterations",
    ];
    const FILENAME: &str = "bias_experiment2_3_errors.csv";

    let mut results = Vec::new();
    println!("Starting error estimation for experiments 2 and 3");
    for i in 0..TRIALS {
        println!("  Trial {} / {TRIALS}", i + 1);
        let bias_results = experiment_2_inner();
        let iteration_results = experiment_3_inner();

        for i in 0..bias_results.len() {
            let p_s = bias_results[i][0];
            let stdev_bias = bias_results[i][2];
            let mean_iterations = iteration_results[i][1];

            if results.len() <= i {
                results.push((p_s, Vec::new(), Vec::new()));
            }

            results[i].1.push(stdev_bias);
            results[i].2.push(mean_iterations);
        }
    }

    // Now compute the mean and stdev for each p_s
    let mut final_results = Vec::new();
    for (p_s, stdev_biases, mean_iterations) in results {
        let mean_stdev_bias =
            stdev_biases.iter().cloned().sum::<f64>() / (stdev_biases.len() as f64);
        let stdev_stdev_bias = (stdev_biases
            .iter()
            .cloned()
            .map(|b| (b - mean_stdev_bias).powi(2))
            .sum::<f64>()
            / (stdev_biases.len() as f64))
            .sqrt();

        let mean_mean_iterations =
            mean_iterations.iter().cloned().sum::<f64>() / (mean_iterations.len() as f64);
        let stdev_mean_iterations = (mean_iterations
            .iter()
            .cloned()
            .map(|b| (b - mean_mean_iterations).powi(2))
            .sum::<f64>()
            / (mean_iterations.len() as f64))
            .sqrt();

        final_results.push([
            p_s,
            mean_stdev_bias,
            stdev_stdev_bias,
            mean_mean_iterations,
            stdev_mean_iterations,
        ]);
    }

    println!("Done; Exporting results to {FILENAME}");
    export_results(FILENAME, HEADINGS, &final_results);
}

struct TrialResult {
    // The number of iterations initially planned by BOAST before any evidence is gathered, which we can use to normalize the number of iterations performed in each trial
    initial_iterations: usize,

    // The prediction_bias, for each trial iteration, which we can use to compute the mean and stdev of bias across all iterations
    // prediction_bias is the difference between the observed pass ratio and the true pass ratio, which captures the bias in BOAST's estimate of the failure probability
    prediction_bias: Vec<f64>,

    // The decision_bias, for each trial iteration, which we can use to compute the mean and stdev of bias across all iterations
    // decision_bias is the difference between the observed pass ratio and the pass ratio threshold, which captures the bias in BOAST's decision to stop or continue iterating
    decision_bias: Vec<f64>,

    // Without the piecewise function to establish that the function is the source of the variance in decision bias SE SE
    raw_decision_bias: Vec<f64>,

    // The number of iterations performed in each trial iteration, which we can use to compute the mean and stdev of iterations across all iterations
    iterations: Vec<f64>,
}

/// Returns (p_fail, [bias]) for a trial
fn trial(ratio_pass: f64, n: usize, p_s: f64, outlier_probability: f64) -> TrialResult {
    const ITERS: usize = 10_000;

    let mut rng = rng();
    let mut state = boast::State::new(
        boast::Options {
            confidence: 0.95,
            outlier_probability,
            pass_ratio: ratio_pass,
            timeout: None,
            prior_strength: Some(p_s),
            damping_constant: None,
        },
        n,
    );

    let mut pbias_results = Vec::new();
    let mut dbias_results = Vec::new();
    let mut raw_dbias = Vec::new();
    let mut iteration_results = Vec::new();
    for _ in 0..ITERS {
        let true_fail = next_fail_probability(&mut rng);
        state.reset();

        while !state.should_stop() {
            let passed = rng.random_bool(1.0 - true_fail);
            state.record_result(passed);
        }

        let obs_pass = state.pass_ratio();
        let true_pass = 1.0 - true_fail;

        let prediction_bias = obs_pass - true_pass;
        let raw_decision_bias = ratio_pass - true_pass;
        let decision_bias = match raw_decision_bias {
            x if x < 0.0 && obs_pass >= ratio_pass => 0.0, // True positive
            x if x > 0.0 && obs_pass < ratio_pass => 0.0,  // True negative
            x if true_pass >= ratio_pass => x,             // False positive
            x => x,                                        // False negative
        };

        pbias_results.push(prediction_bias);
        dbias_results.push(decision_bias);
        raw_dbias.push(raw_decision_bias);
        iteration_results.push(state.iterations() as f64);
    }

    TrialResult {
        initial_iterations: state.initial_k(),
        prediction_bias: pbias_results,
        decision_bias: dbias_results,
        raw_decision_bias: raw_dbias,
        iterations: iteration_results,
    }
}

fn next_fail_probability(rng: &mut impl Rng) -> f64 {
    let (min, max) = match rng.random::<f64>() {
        x if x < 0.10 => (0.001, 0.01), // Very rare failures
        x if x < 0.30 => (0.01, 0.05),  // Rare failures
        x if x < 0.65 => (0.05, 0.20),  // Occasional failures
        x if x < 0.90 => (0.20, 0.50),  // Frequent failures
        _ => (0.50, 0.999),             // Very frequent failures
    };

    // uniformly sample between min and max
    rng.random_range(min..=max)
}

pub fn simulate_true_fail_probabilities() {
    const ITERS: usize = 100_000;
    const FILENAME: &str = "p_fail_histogram.csv";
    const HEADINGS: [&str; 2] = ["p_fail", "count"];

    let mut rng = rng();
    let mut probabilities = Vec::new();

    for _ in 0..ITERS {
        probabilities.push(next_fail_probability(&mut rng));
    }

    let mut histogram: Vec<[f64; 2]> = Vec::new();
    for p in &probabilities {
        // First find out if any entry is already <epsilon from the current value
        let entry = histogram
            .iter_mut()
            .find(|row| f64::abs(*p - row[0]) < 0.001);
        if let Some(row) = entry {
            row[1] += 1.0;
        } else {
            histogram.push([*p, 1.0]);
        }
    }

    println!("Done; Exporting results to {FILENAME}");
    export_results(FILENAME, HEADINGS, &histogram);
}
