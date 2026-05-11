This crate is an implementation of the [BOAST](https://github.com/rscarson/boast/blob/master/boast.draft.pdf) testing algorithm for same-size transformed data sets

It is mostly designed for use in numerical library or statistical property testing

## Limitations

- The paper assumes a normal or uniform distribution for outliers in your data
- Tests must be fully independent and consume different sets of the same length

## Usage

Here is a basic example of how to use this crate using the macros. You can also use the underlying functions directly if you want more control, but the macros are designed to be pretty flexible and easy to use.

```rust
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

    //
    // The parameters look scary but they're actually pretty simple
    // confidence / q = how sure you want to be that if a bug exists, you'll find it
    // outlier_rate / p = how likely is it that any value in your data is an `outlier` that could contribute to a failure
    // timeout = how long to run the test before giving up
    // pass_ratio = what fraction of the tests must pass for the overall test to be considered a success (default 100%)
    //
    // The outlier_rate is really just a guess, as the paper demonstrates that the algorithm is pretty robust to misestimation AND
    // this is just an upper bound - try to underestimate it - 1e-4 to 1e-6 is usually a good range.
    #[boast::test(confidence = 0.95, outlier_rate = 1e-6, timeout = 60, pass_ratio = 0.90)]
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
```

Here is an example that uses polyfit to ease random transformations

```rust
use polyfit::{
    assert_residuals_normal, function,
    MonomialFit, statistics::DegreeBound, score::Aic, 
    transforms::{ApplyNoise, Strength}
};

use boast::{Options, DataSource, assert_randomized};

/// Given a data set, apply gaussian noise with a stdev of 0.01
fn transform_data(data: &[(f64, f64)], seed: u64) -> Vec<(f64, f64)> {
    data.clone().apply_normal_noise(Strength::Absolute(0.01), Some(seed))
}

/// Given a data set, fail if the data's p-value for residual normality is not at least 0.01
fn test(data: &[(f64, f64)]) {
    let fit = MonomialFit::new_auto(data, DegreeBound::Relaxed, &Aic).unwrap();
    assert_residuals_normal!(&fit, 0.01);
}

fn main() {
    // Just a polynomial function to generate some data
    function!(const poly(x) = 1.0 + 2.0 x^1 + 3.0 x^2);

    // The test is going to use pre-generated data that gets transformed at each step
    let data = DataSource::transformed(polysolve_range(0.0..=1000.0, 1.0), transform_data);

    // We believe that each point has more than 0.01% chance of being an `outlier` that could cause problems
    // We want to be 95% sure that if a bug exists, we found it
    // By default this also means 95% of the tests must pass
    let options = Options::new(0.95, 1e-4);

    // Run the test
    // If it fails, you'll have an estimate of the true failure rate, the last seed to fail, and how many tests were run
    assert_randomized(options, data, test);
}
```