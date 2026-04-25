This crate is an implementation of the [BIOD](https://github.com/rscarson/baysian_iterative_outlier_detection/blob/master/bayesian_iterative_outlier_detection.draft.pdf) testing algorithm for same-size transformed data sets

It is mostly designed for use in numerical library or statistical property testing

## Limitations

- The paper assumes a normal or uniform distribution for extreme outliers in your data
- Tests must be fully independent and consume different sets of the same length

## Usage

Here is an example that uses polyfit to ease random transformations

```rust
use polyfit::{
    assert_residuals_normal, function,
    MonomialFit, statistics::DegreeBound, score::Aic, 
    transforms::{ApplyNoise, Strength}
};

use biod::{Options, DataSource, assert_randomized};

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

    // We believe that each point has more than 0.01% chance of being an `extreme outlier` that could cause problems
    // We want to be 95% sure that if a bug exists, we found it
    // By default this also means 95% of the tests must pass
    let options = Options::new(0.95, 1e-4);

    // Run the test
    // If it fails, you'll have an estimate of the true failure rate, the last seed to fail, and how many tests were run
    assert_randomized(options, data, test);
}
```