use rand_distr::Distribution;
use statrs::distribution::ContinuousCDF;

use crate::experiments::export_results;

/// Compare estimated wald p_fail interval over various n to predicted p_fail from BOAST
pub fn experiment_1() {
    const HEADINGS: [&str; 6] = [
        "n",
        "p_fail_simulated",
        "p_fail_sampled",
        "p_fail_lower",
        "p_fail_upper",
        "ci_width",
    ];

    //
    // Rare failure (p_fail = 0.01)
    let results_rare = experiment_1_for_fail(0.01);
    println!("Done with rare failure");
    export_results("estimate_experiment1_rare.csv", HEADINGS, &results_rare);

    //
    // Common failure (p_fail = 0.3)
    let results_common = experiment_1_for_fail(0.3);
    println!("Done with common failure");
    export_results("estimate_experiment1_common.csv", HEADINGS, &results_common);

    // Very common failure (p_fail = 0.9)
    let results_very_common = experiment_1_for_fail(0.9);
    println!("Done with very common failure");
    export_results(
        "estimate_experiment1_very_common.csv",
        HEADINGS,
        &results_very_common,
    );
}

/// Compare estimated wald p_fail interval over various n to predicted p_fail from BOAST
pub fn experiment_1_for_fail(p_fail: f64) -> Vec<[f64; 6]> {
    // Normal distribution with mean 0 and stddev 1
    // Outliers are points with abs(value) > 3.0
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::rng();

    let c = c_for_pfail(p_fail); // Get the corresponding c value for the desired p_fail

    let (mut n_mul, n_max) = (0.0, 5.0);
    let mut results = Vec::new();
    println!("Starting experiment 1");
    while n_mul <= n_max {
        //
        // Determine n, stepping by powers of 10^0.5
        let n = 10f64.powf(n_mul) as usize;
        n_mul += 0.5;

        print!("\rTesting n = {n}...");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        // Determine how many datasets we need
        let state = boast::State::new(
            boast::Options {
                confidence: 0.95,
                outlier_probability: 0.0027, // Approximate probability of a point being outside 3 std devs in a normal distribution
                pass_ratio: 0.0,
                timeout: None,
            },
            n,
        );
        let k = state.current_k();

        // Generate all datasets ahead of time to ensure fair comparison
        // We will make it so that the max sample size k is 1% of the total number
        let total_datasets = k * 100;
        let datasets: Vec<Vec<f64>> = (0..total_datasets)
            .map(|_| (0..n).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        // Determine true p_fail
        let mut pass_count = 0;
        for dataset in &datasets {
            if z_test(c, dataset) {
                pass_count += 1;
            }
        }
        let p_fail = 1.0 - (pass_count as f64 / total_datasets as f64);

        // Determine sampled p_fail from BOAST
        let data = boast::DataSource::iterable(n, datasets.into_iter());
        let runner = boast::Runner::new(boast::Options::new(0.95, 0.0027), data, |data| {
            let passed = z_test(c, data);
            assert!(passed, "Test failed: dataset did not pass");
        });
        let stats = runner.try_run().unwrap().into_stats();
        let p_fail_sampled = 1.0 - stats.observed_pass_ratio;

        let (p_fail_lower, p_fail_upper) = (stats.p_fail_lower_bound, stats.p_fail_upper_bound);
        let half_width = (p_fail_upper - p_fail_lower) / 2.0;

        results.push([
            n as f64,
            p_fail,
            p_fail_sampled,
            p_fail_lower,
            p_fail_upper,
            half_width,
        ]);
    }

    results
}

fn z_test(c: f64, data: &[f64]) -> bool {
    // Z = (1/sqrt(n)) * sum(x)
    // fail if |Z| > c
    let n = data.len() as f64;
    let z = data.iter().sum::<f64>() / n.sqrt();
    z.abs() <= c
}

fn c_for_pfail(p_fail: f64) -> f64 {
    // c = inverse_cdf(1 - p_fail/2) for a standard normal distribution
    let normal = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
    normal.inverse_cdf(1.0 - p_fail / 2.0)
}
