#![allow(clippy::manual_is_multiple_of)]

use rand::{Rng, rngs::ThreadRng};

use crate::{DataSource, Options, State};

struct PanicHookGuard {
    original_hook: Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static>,
    payload: std::sync::Arc<std::sync::Mutex<String>>,
}

/// Summary statistics from a completed BIOD run
#[derive(Debug, Clone, PartialEq)]
pub struct Stats {
    /// Estimated lower bound on the failure probability
    pub p_fail_lower_bound: f64,

    /// Estimated upper bound on the failure probability
    pub p_fail_upper_bound: f64,

    /// Approximate observed ratio of passing tests to total tests
    pub observed_pass_ratio: f64,

    /// Number of observed failures during testing
    pub observed_failures: usize,

    /// Total number of iterations performed during testing
    pub total_iterations: usize,

    /// The seeds of all failing tests observed during the run
    pub failing_seeds: Vec<u64>,
}
impl Stats {
    pub fn from_state(state: &State, failing_seeds: Vec<u64>) -> Option<Self> {
        let (p_fail_lower, p_fail_upper) = state.p_fail_interval();
        Some(Self {
            p_fail_lower_bound: p_fail_lower,
            p_fail_upper_bound: p_fail_upper,
            observed_pass_ratio: state.pass_ratio(),
            observed_failures: state.iterations() - state.passes(),
            total_iterations: state.iterations(),
            failing_seeds,
        })
    }
}

/// Current stage of the BIOD process
#[derive(Debug, Clone, PartialEq)]
pub enum TestResult {
    /// Test has completed successfully, meeting the required confidence and pass ratio
    Passed(Stats),

    /// Test has completed unsuccessfully, failing to meet the required confidence or pass ratio
    Failed(Stats),
}
impl TestResult {
    pub fn stats(&self) -> &Stats {
        match self {
            TestResult::Passed(stats) | TestResult::Failed(stats) => stats,
        }
    }

    /// Returns true if the test passed, false if it failed
    pub fn is_passed(&self) -> bool {
        matches!(self, TestResult::Passed(_))
    }

    /// Convert the TestResult into its underlying Stats, discarding the pass/fail distinction
    pub fn into_stats(self) -> Stats {
        match self {
            TestResult::Passed(stats) | TestResult::Failed(stats) => stats,
        }
    }
}

/// Errors that can occur during the BIOD process
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    TimedOut,
    DataSourceExhausted,
    Internal(String),
}
impl std::error::Error for Error {}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::TimedOut => write!(f, "Test timed out"),
            Error::DataSourceExhausted => write!(f, "Data source exhausted"),
            Error::Internal(msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

pub struct Runner<'src, T, FTest>
where
    T: std::panic::RefUnwindSafe,
    FTest: FnMut(&[T]) + std::panic::RefUnwindSafe,
{
    state: State,

    data: DataSource<'src, T>,
    test: FTest,

    rng: ThreadRng,
    failing_seeds: Vec<u64>,
    last_err: Option<(Box<dyn std::any::Any + Send>, u64)>,
}
impl<'src, T, FTest> Runner<'src, T, FTest>
where
    T: std::panic::RefUnwindSafe,
    FTest: Fn(&[T]) + std::panic::RefUnwindSafe,
{
    pub fn new(options: Options, data: DataSource<'src, T>, test: FTest) -> Self {
        Self {
            state: State::new(options, data.len()),
            data,
            test,

            rng: rand::rng(),
            failing_seeds: Vec::new(),
            last_err: None,
        }
    }

    /// Run the BIOD process to completion, panicking if the test fails
    ///
    /// Will print output during the process,
    /// including the final confidence and failure rate estimates,
    /// and the seed of the last failing test if applicable.
    pub fn run(self) {
        self.inner_run(true, true).expect("Test failed");
    }

    /// Get a reference to the current state of the BIOD process, which includes iteration counts,
    /// pass counts, and Bayesian parameters.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Run the BIOD process to completion, returning a Result indicating the outcome
    ///
    /// Will not print any output during the process,
    /// but will still capture the seed of the last failing test if applicable.
    pub fn try_run(self) -> Result<TestResult, Error> {
        self.inner_run(false, false)
    }

    pub fn inner_run(
        mut self,
        panic_on_fail: bool,
        print_output: bool,
    ) -> Result<TestResult, Error> {
        let hook = Self::start_panic_capture();
        let state = &self.state;

        if print_output {
            println!(
                "Starting with initial pass count k = {}, confidence = {:.2}%, required pass ratio = {:.2}%",
                state.initial_k(),
                state.q() * 100.0,
                state.options().pass_ratio * 100.0
            );
        }

        if print_output {
            print!(
                "\rFinished 0 / {} iterations. 0 failures reported.",
                state.current_k(),
            );
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }

        loop {
            let result = match self.step() {
                Some(r) => r,
                None => {
                    if print_output
                        && (self.state.current_k() < 1_000 || self.state.iterations() % 10 == 0)
                    {
                        let i = self.state.iterations();
                        let k = self.state.current_k().max(i);
                        let failures = i - self.state.passes();
                        let p_fail = self.state.p_fail() * 100.0;
                        print!(
                            "\rFinished {i} / {k} iterations. {failures} failures reported. Estimated >={p_fail:.2}% of sets induce failure."
                        );
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }

                    continue;
                }
            };

            match &result {
                Err(Error::TimedOut) if print_output => {
                    eprintln!("Test ended due to timeout before sufficient evidence was gathered.");
                }

                Err(Error::DataSourceExhausted) if print_output => {
                    eprintln!(
                        "Test ended because the data source was exhausted before sufficient evidence was gathered."
                    );
                }

                Err(Error::Internal(msg)) if print_output => {
                    eprintln!("Internal error: {msg}");
                }

                Ok(TestResult::Passed(stats)) | Ok(TestResult::Failed(stats)) if print_output => {
                    let fail_low = stats.p_fail_lower_bound * 100.0;
                    let fail_high = stats.p_fail_upper_bound * 100.0;
                    println!(
                        "\n\nWith 95% confidence, the true failure rate is between {fail_low:.2}% and {fail_high:.2}%.",
                    );

                    let failures = stats.observed_failures;
                    let total_iters = self.state.iterations();
                    let ratio = stats.observed_pass_ratio * 100.0;
                    eprintln!("{failures}/{total_iters} tests failed ({ratio:.2}% pass)\n");

                    if let Ok(TestResult::Failed(_)) = result
                        && let Some((_, seed)) = &self.last_err
                    {
                        let payload = hook.payload.lock().unwrap().clone();
                        eprintln!("Test failed.");
                        eprintln!("Last failing seed: 0x{seed:0X}");
                        eprintln!("{payload}");
                    }

                    if panic_on_fail {
                        if let Ok(TestResult::Failed(_)) = result {
                            if let Some((err, _)) = self.last_err {
                                std::panic::resume_unwind(err);
                            }
                        }
                    }
                }

                _ => {}
            }

            Self::stop_panic_capture(hook);
            return result;
        }
    }

    fn result(&self) -> Option<Result<TestResult, Error>> {
        if !self.state.should_stop() {
            return None;
        }

        let stats = match Stats::from_state(&self.state, self.failing_seeds.clone()) {
            Some(s) => s,
            None => {
                return Some(Err(Error::Internal(
                    "Failed to compute test statistics from state.".to_string(),
                )));
            }
        };

        if self.state.has_timed_out() {
            Some(Err(Error::TimedOut))
        } else if self.state.has_passed() {
            Some(Ok(TestResult::Passed(stats)))
        } else {
            Some(Ok(TestResult::Failed(stats)))
        }
    }

    fn step(&mut self) -> Option<Result<TestResult, Error>> {
        if self.result().is_none() {
            let seed: u64 = self.rng.random();
            let transformed_data = match self.data.get_data(seed) {
                Ok(data) => data,
                Err(err) => return Some(Err(err)),
            };
            let result = std::panic::catch_unwind(|| {
                (self.test)(&transformed_data);
            });

            let passed = result.is_ok();
            self.state.record_result(passed);

            if let Err(err) = result {
                self.last_err = Some((err, seed));
                self.failing_seeds.push(seed);
            }
        }

        self.result()
    }

    fn start_panic_capture() -> PanicHookGuard {
        let default_hook = std::panic::take_hook();
        let last_payload = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let hook_payload = std::sync::Arc::clone(&last_payload);

        std::panic::set_hook(Box::new(move |info| {
            let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = info.payload().downcast_ref::<String>() {
                s.clone()
            } else {
                String::new()
            };

            *hook_payload.lock().unwrap() = payload;
        }));

        PanicHookGuard {
            original_hook: default_hook,
            payload: last_payload,
        }
    }

    fn stop_panic_capture(guard: PanicHookGuard) {
        std::panic::set_hook(guard.original_hook);
    }
}
