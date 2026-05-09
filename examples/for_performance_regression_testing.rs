use boast::{DataSource, Options, Runner};
use rand::prelude::*;

fn main() {
    //
    // This is an example of how you might use BOAST to do performance regression testing.
    // We have two versions of a function (function_a and function_b) that we want to make sure have similar performance characteristics - we don't want to regress on performance when we update the function.
    //
    // We will use BOAST to run both functions on the same data and compare their runtimes, and we will set it up so that if function_b is significantly slower than function_a, the test will fail.
    //

    // First we need a data source - it will provide the data sets for each trial
    // Here we are generating datasets from scratch using the DataSource::generated constructor, which takes in a length and a generator function
    let datasource = DataSource::generated(100, |dst, seed| {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let data: Vec<u64> = (0..100).map(|_| rng.random_range(0..10000)).collect();
        dst.copy_from_slice(&data);
    });

    //
    // Now we set up the options for BOAST
    // We are asking for a confidence level of 95% (confidence = 0.95) that we'd have observed a failure if one existed
    // We think that <0.0001% of the points in our data will be strange enough to contribute to a failure (outlier_rate = 1e-4)
    // And we are ok with up to 10% of the tests failing due to random noise, so we set a pass ratio of 90% (pass_ratio = 0.90)
    let options = Options::new(0.95, 1e-6).with_pass_ratio(0.90);

    // Create the BOAST runner with the options, data source, and a test function
    // The test function runs both versions of the function on the same data and compares their runtimes - if function_b is significantly slower than function_a, this will count as a failure for this trial
    let runner = Runner::new(options, datasource, |data: &[u64]| {
        let start_a = std::time::Instant::now();
        function_a(data);
        let duration_a = start_a.elapsed();

        let start_b = std::time::Instant::now();
        function_b(data);
        let duration_b = start_b.elapsed();

        // Compare the durations and assert that function_b is not slower than function_a
        assert!(
            duration_b <= duration_a,
            "function_b is slower than function_a"
        );
    });

    // Run the test and get the results
    // We don't want to panic on fail because we intend to use the results
    println!("Running performance regression test...");
    let result = runner
        .try_run()
        .expect("Internal error during test execution");
    let stats = result.into_stats();

    let mut rng = rand::rngs::StdRng::from_os_rng();
    let data: Vec<u64> = (0..1000).map(|_| rng.random_range(0..10000)).collect();

    // Now we run both functions a bunch of times to get a distribution of its runtimes
    println!("Running functions to get runtime distributions...");
    let mut runtimes_a = Vec::new();
    let mut runtimes_b = Vec::new();
    for _ in 0..100 {
        let start = std::time::Instant::now();
        function_a(&data);
        runtimes_a.push(start.elapsed().as_secs_f64());

        let start = std::time::Instant::now();
        function_b(&data);
        runtimes_b.push(start.elapsed().as_secs_f64());
    }

    let (stdev_a, mean_a) = polyfit::statistics::stddev_and_mean(runtimes_a.into_iter());
    let (stdev_b, mean_b) = polyfit::statistics::stddev_and_mean(runtimes_b.into_iter());

    println!(
        "function_a: mean = {:.0}ms, stdev = {:.0}ms",
        mean_a * 1e3,
        stdev_a * 1e3
    );
    println!(
        "function_b: mean = {:.0}ms, stdev = {:.0}ms",
        mean_b * 1e3,
        stdev_b * 1e3
    );

    let success_confidence = (
        stats.p_fail_lower_bound * 100.0,
        stats.p_fail_upper_bound * 100.0,
    );
    println!(
        "Confidence that function_b is slower than function_a: {:.2}% - {:.2}% ({} / {} trials failed)",
        success_confidence.0, success_confidence.1, stats.observed_failures, stats.total_iterations
    );
}

// Represents some older version of a function with known performance characteristics that we want to make sure we don't regress on
pub fn function_a(data: &[u64]) {
    for &x in data {
        let mut sum: u64 = 0;
        for i in 0..x {
            sum = sum.wrapping_add(i ^ 0x9E3779B97F4A7C15);
        }
        std::hint::black_box(sum);
    }
}

// An updated version of the function that we want to test against the old one to make sure we don't regress on performance
pub fn function_b(data: &[u64]) {
    for &x in data {
        let sum = (0..x)
            .map(|i| i ^ 0x9E3779B97F4A7C15)
            .fold(0u64, |acc, v| acc.wrapping_add(v));
        std::hint::black_box(sum);
    }
}

#[cfg(test)]
mod test {
    use super::{function_a, function_b};

    use boast::DataSource;
    use rand::prelude::*;

    fn datasource() -> DataSource<u64> {
        DataSource::generated(100, |dst, seed| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let data: Vec<u64> = (0..100).map(|_| rng.random_range(0..10000)).collect();
            dst.copy_from_slice(&data);
        })
    }

    #[boast::test(q = 0.95, p = 1e-6, timeout = 60, pass_ratio = 0.90)]
    fn performance_regression_test(#[src = datasource] data: &[u64]) {
        let start_a = std::time::Instant::now();
        function_a(data);
        let duration_a = start_a.elapsed();

        let start_b = std::time::Instant::now();
        function_b(data);
        let duration_b = start_b.elapsed();

        // Compare the durations and assert that function_b is not slower than function_a
        assert!(
            duration_b <= duration_a,
            "function_b is slower than function_a"
        );
    }
}
