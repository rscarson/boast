use crate::Options;

/// State for Bayesian Outlier-Aware Sequential Testing (BOAST)
///
/// Keeps track of the current state of the BOAST process, including iteration counts,
/// pass counts, and Bayesian parameters.
#[derive(Debug, Clone, Copy)]
pub struct State {
    /// The size of the data set under test
    n: usize,

    /// The prior-belief strength (`p_s`) in the initial outlier probability
    /// See the paper for details on the recommended value for this parameter
    p_s: f64,

    /// The constant C used to tune the model
    /// See the paper for details on the recommended value for this parameter
    damping_constant: f64,

    /// Options for the BOAST process
    options: Options,

    /// The time at which the BOAST process started. Used for timeout calculations.
    start_time: std::time::Instant,

    //
    // Iteration counts
    initial_k: usize,
    current_k: usize,
    iterations: usize,

    //
    // Results counts
    passes: usize,
    unreported_passes: usize,

    //
    // Bayesian parameters
    a: f64,
    b: f64,
}
impl State {
    const DAMPING_CONSTANT: f64 = 1.864;
    const PRIOR_STRENGTH: f64 = 3.35e6;

    /// Creates a new BOAST state instance with the given options and data set size.
    #[must_use]
    pub fn new(options: Options, n: usize) -> Self {
        let p_s = options.prior_strength.unwrap_or(Self::PRIOR_STRENGTH);
        let c = options.damping_constant.unwrap_or(Self::DAMPING_CONSTANT);

        let mut boast = Self {
            n,
            p_s,
            damping_constant: c,
            options,
            start_time: std::time::Instant::now(),

            initial_k: 0,
            current_k: 0,
            iterations: 0,

            passes: 0,
            unreported_passes: 0,

            a: 0.0,
            b: 0.0,
        };

        //  Calculate `p_fail`, the probability of a single data set failing:
        //  `p_fail = 1 - (1 - p')^n`
        //
        //  And use `p_s` to calculate the prior alpha and beta values for the Bayesian update
        let p_prime = boast.p_prime();
        let p_fail = 1.0 - (1.0 - p_prime).powi(boast.n as _);

        // Calculate initial k
        boast.initial_k = boast.k(p_fail);
        boast.current_k = boast.initial_k;

        boast.a = p_fail * boast.p_s();
        boast.b = (1.0 - p_fail) * boast.p_s();
        boast
    }

    pub fn reset(&mut self) {
        self.start_time = std::time::Instant::now();

        self.iterations = 0;
        self.passes = 0;
        self.unreported_passes = 0;

        let p_fail = self.p_fail();
        self.current_k = self.k(p_fail);
    }

    /// Calculates a damping constant C based on a desired floor for k and the lowest conservative q value.
    ///
    /// k_floor is the theoretical minimum number of iterations required to achieve confidence q as n approaches infinity
    /// lowest_conservative_q is the lowest value of q you consider to be resonably conservative
    ///
    /// The recommended value is based on k_floor of 3, and q of 0.8, which results in a damping constant of 1.864.
    ///
    /// This constant encodes the universe of reasonable parameters for the model, so:
    /// - For any q above the one set, the iterations initially estimated will always be above the k_floor
    /// - The lower the q provided, the more aggressively the model increases iterations as q grows
    pub fn calculate_damping_constant(
        &mut self,
        desired_k_floor: f64,
        lowest_conservative_q: f64,
    ) -> f64 {
        desired_k_floor / (-(1.0 - lowest_conservative_q).ln())
    }

    /// Returns the options used for this BOAST state.
    #[must_use]
    pub fn options(&self) -> Options {
        self.options
    }

    /// Calculate k, the correct number of iterations to achieve q given p
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn k(&self, p_fail: f64) -> usize {
        ((1.0 - self.q()).ln() / (1.0 - p_fail).ln()).abs().ceil() as usize
    }

    /// The pointwise probability of an extreme outlier causing a failure
    #[must_use]
    pub fn p(&self) -> f64 {
        self.options.outlier_probability
    }

    /// The required confidence (`q`) in the initial number of iterations being above
    #[must_use]
    pub fn q(&self) -> f64 {
        self.options.confidence
    }

    /// The prior-belief strength (`p_s`) in the initial outlier probability
    #[must_use]
    pub fn p_s(&self) -> f64 {
        self.p_s
    }

    /// The alpha parameter of the Beta distribution used in Bayesian updating
    #[must_use]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// The beta parameter of the Beta distribution used in Bayesian updating
    #[must_use]
    pub fn b(&self) -> f64 {
        self.b
    }

    /// The initially estimated required number of iterations to achieve confidence `q`
    #[must_use]
    pub fn initial_k(&self) -> usize {
        self.initial_k
    }

    /// The currently estimated required number of iterations to achieve confidence `q`
    #[must_use]
    pub fn current_k(&self) -> usize {
        self.current_k
    }

    /// The total number of iterations performed so far
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// The number of passing iterations performed so far
    #[must_use]
    pub fn passes(&self) -> usize {
        self.passes
    }

    /// To reflect the diminishing impact of additional data points on the likelihood of failure due to extreme outliers, we use a damping factor to adjust p based on n.
    /// This takes the form of a linear fractional damping function `1 / (1 + Cx)`, where C is a constant chosen to reflect the diminishing impact of additional data points
    /// and x is the expected number of extreme outliers in the data set, `n * p`.
    #[must_use]
    pub fn damping_fraction(&self) -> f64 {
        1.0 / (1.0 + (self.damping_constant * self.n as f64 * self.p()))
    }

    /// Calculate p', the adjusted value for p, the probability of observing an extreme outlier that can trigger a failure
    #[must_use]
    pub fn p_prime(&self) -> f64 {
        let damping_fraction = self.damping_fraction();
        self.p() * damping_fraction
    }

    /// Calculate `p_fail`, the probability of a single data set failing:
    /// `p_fail = a / (a + b)`
    #[must_use]
    pub fn p_fail(&self) -> f64 {
        self.a / (self.a + self.b)
    }

    /// Records the result of a test iteration, updating the internal state accordingly.
    pub fn record_result(&mut self, passed: bool) {
        self.iterations += 1;
        if passed {
            self.passes += 1;
            self.unreported_passes += 1;
        }

        if self.passes < self.iterations {
            self.a += 1.0;
            self.b += self.unreported_passes as f64;
            self.unreported_passes = 0;
        }

        let p_fail = self.p_fail();
        self.current_k = self.k(p_fail);
    }

    /// Calculates the current pass ratio.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[must_use]
    pub fn pass_ratio(&self) -> f64 {
        self.passes as f64 / self.iterations as f64
    }

    /// Determines whether the BOAST process has timed out based on the configured timeout option.
    #[must_use]
    pub fn has_timed_out(&self) -> bool {
        if let Some(timeout) = self.options.timeout {
            let elapsed = self.start_time.elapsed();
            elapsed >= timeout
        } else {
            false
        }
    }

    /// Determines whether the test end, based on current state and options.
    #[must_use]
    pub fn should_stop(&self) -> bool {
        if self.options.pass_ratio >= 1.0 && self.passes < self.iterations {
            // If pass_ratio is 1.0 or greater and we have a failure, stop immediately
            return true;
        }

        // Otherwise, stop if we have enough evidence
        self.iterations >= self.current_k
    }

    /// Determines whether there is enough evidence to fulfill q, and the pass ratio requirement.
    #[must_use]
    pub fn has_passed(&self) -> bool {
        self.pass_ratio() >= self.options.pass_ratio && self.has_sufficient_evidence()
    }

    /// Determines whether there is sufficient evidence to fulfill q.
    #[must_use]
    pub fn has_sufficient_evidence(&self) -> bool {
        let min_iterations = self.initial_k.min(self.current_k);
        self.iterations >= min_iterations
    }

    /// Calculates the bounds for the estimated true failure rate at a 95% confidence level using the Wilson score interval.
    #[must_use]
    pub fn p_fail_interval(&self) -> (f64, f64) {
        // center = (p̂ + z²/(2n)) / (1 + z²/n)
        // half_width =
        //  z * sqrt( (p̂(1−p̂)/n) + z²/(4n²) )
        //  / (1 + z²/n)
        // p = 1 - pass_ratio
        let p = 1.0 - self.pass_ratio();
        let k = self.iterations as f64;
        let z = 1.96; // for 95% confidence
        let center = (p + z * z / (2.0 * k)) / (1.0 + z * z / k);
        let half_width =
            z * ((p * (1.0 - p) / k) + (z * z / (4.0 * k * k))).sqrt() / (1.0 + z * z / k);

        let p_fail_lower = (center - half_width).max(0.0);
        let p_fail_upper = (center + half_width).min(1.0);

        (p_fail_lower, p_fail_upper)
    }
}
