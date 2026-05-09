mod options;
pub use options::Options;

mod state;
pub use state::State;

mod data_src;
pub use data_src::DataSource;

mod runner;
pub use runner::{Error, Runner, Stats, TestResult};

/// Convenience function to run a test with the given options and data source, and print the results.
///
/// This is the equivalent of:
/// ```rust
/// let runner = Runner::new(options, data.into(), test);
/// runner.run();
/// ```
///
/// # Panics
/// This function will panic if the test fails, returning the last seed to fail.
///
/// If the test passes, it will print the confidence interval for the failure rate and the number of iterations performed.
pub fn assert_randomized<T, FTest>(options: Options, data: impl Into<DataSource<T>>, test: FTest)
where
    T: std::panic::RefUnwindSafe + Clone,
    FTest: Fn(&[T]) + std::panic::RefUnwindSafe,
{
    let runner = Runner::new(options, data.into(), test);
    runner.run();
}

/// Convenience function to run a test with the given options and data source, and return the results, without panicking or printing anything.
///
/// This is the equivalent of:
/// ```rust
/// let runner = Runner::new(options, data.into(), test);   
/// runner.try_run();
/// ```
///
/// # Errors
/// This function will return an error something goes wrong during the test, such as a timeout, or issue with the data source.
pub fn test<T, FTest>(
    options: Options,
    data: impl Into<DataSource<T>>,
    test: FTest,
) -> Result<TestResult, Error>
where
    T: Clone + std::panic::RefUnwindSafe,
    FTest: Fn(&[T]) + std::panic::RefUnwindSafe,
{
    let runner = Runner::new(options, data.into(), test);
    runner.try_run()
}

pub use boast_macros::test;
