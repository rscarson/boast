use polyfit::transforms::{NoiseTransform, Strength, Transform};

fn main() {
    //
    // The easy, canonical way to use this is the #[boast::test] macro you see below
    // But you can use the Runner directly if you want more control, or to actually analyze the results instead of just panicking on failure or printing them out
    //

    // First we need a data source - it will provide the data sets for each trial
    // Here we are generating datasets from scratch using the DataSource::generated constructor, which takes in a length and a generator function
    let datasource = boast::DataSource::generated(1000, |container, seed| {
        let transform = NoiseTransform::CorrelatedGaussian {
            rho: 0.0,
            strength: Strength::Relative(0.1),
            seed: Some(seed),
        };
        transform.apply(container.iter_mut());
    });

    //
    // Now we set up the options for BOAST
    // We are asking for a confidence level of 95% (confidence = 0.95) that we'd have observed a failure if one existed
    // We think that <0.0001% of the points in our data will be strange enough to contribute to a failure (outlier_rate = 1e-6)
    // And we are ok with up to 10% of the tests failing due to random noise, so we set a pass ratio of 90% (pass_ratio = 0.90)
    let options = boast::Options::new(0.95, 1e-6).with_pass_ratio(0.90);

    // Create the BOAST runner with the options, data source, and a test function
    // the function just makes sure that the normally distributed data we generated actually is normally distributed!
    let runner = boast::Runner::new(options, datasource, |data: &[f64]| {
        let normality = polyfit::statistics::residual_normality(data);
        assert!(
            normality > 0.05,
            "Residuals are not normal: p-value = {}",
            normality
        );
    });

    // Run the test and get the results
    let results = runner
        .try_run()
        .expect("Internal error during test execution");
    println!("Test results:\n{}", results);
}

#[cfg(test)]
mod tests {
    use polyfit::{
        ChebyshevFit, MonomialFit, assert_fits, assert_residuals_normal, function,
        score::Aic,
        statistics::DegreeBound,
        transforms::{NoiseTransform, Strength, Transform},
    };

    function!(const POLY(x) = 1.0 + 2.0 x^1 + 3.0 x^2);

    fn data_src() -> boast::DataSource<(f64, f64)> {
        //
        // This is the function that generates the data for our tests.
        //
        // It uses a polynomial function (POLY) to create the initial dataset
        // The DataSource for the tests then applies a transformation to the data that adds some Normal (Gaussian) noise to the y-values
        let data = POLY.solve_range(0.0..=1000.0, 1.0);

        boast::DataSource::transformed(data, |d, seed| {
            let transform = NoiseTransform::CorrelatedGaussian {
                rho: 0.0,
                strength: Strength::Relative(0.1),
                seed: Some(seed),
            };
            transform.apply(d.iter_mut().map(|(_, y)| y));
        })
    }

    #[boast::test(q = 0.95, p = 1e-4, timeout = 60, pass_ratio = 0.90)]
    fn test_normal_distribution(#[src = data_src] data: &[(f64, f64)]) {
        //
        // This test exists basically verbatim in polyfit - it's used to test the assert_residuals_normal macro!
        //
        // Check out the parameters we used:
        // #[boast::test(q = 0.95, p = 1e-4, timeout = 60, pass_ratio = 0.90)]
        //
        // So we are asking for a confidence level of 95% (q = 0.95) that we'd have observed a failure if one existed
        // We think that <0.1% of the points in our data will be strange enough to contribute to a failure (p = 1e-4)
        // We want to stop the test after 60 seconds if it hasn't already stopped (timeout = 60)
        // And we know that this will fail about 10% of the time due to random noise, so we want to require at least a 90% pass ratio
        let fit = MonomialFit::new_auto(data, DegreeBound::Relaxed, &Aic).unwrap();
        assert_residuals_normal!(&fit, 0.05);
    }

    #[boast::test(confidence = 0.95, outlier_rate = 1e-6)]
    fn test_can_fit(#[src = data_src] data: &[(f64, f64)]) {
        //
        // This one a data recovery test - if we use a function and add noise, can my Chebyshev implementation recover the original function?
        //
        // I used a different setup here:
        // #[boast::test(confidence = 0.95, outlier_rate = 1e-6)]
        //
        // So here we are asking for a confidence level of 95% (confidence = 0.95) that we'd have observed a failure if one existed
        // We think that <0.0001% of the points in our data will be strange enough to contribute to a failure (outlier_rate = 1e-6)
        //   -> It is VERY hard to confuse Chebyshev
        // We are not setting a timeout or pass ratio, so any failure will be a real failure
        // It runs until a failure happens or we have enough evidence to be confident that it won't fail
        let fit = ChebyshevFit::new(data, 2).unwrap();
        assert_fits!(POLY, fit);
    }
}
