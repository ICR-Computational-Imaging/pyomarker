functions {
    vector dirichlet_multinomial(array[,] int y, vector mu, real prec) {
        int N = dims(y)[1];
        int K = dims(y)[2];
        vector[N] y_sum = to_vector(rep_array(0, N));
        vector[N] p_sum = to_vector(rep_array(0, N));
        for (k in 1:K) {
            vector[N] y_col = to_vector(y[:, k]);
            y_sum += y_col;
            p_sum += lgamma(y_col + mu[k] * prec);
        }
        return lgamma(prec) - lgamma(y_sum + prec) + p_sum - sum(lgamma(mu * prec));
    }

    real dirichlet_multinomial_lpmf(array[,] int y, vector mu, real prec) {
        return sum(dirichlet_multinomial(y, mu, prec));
    }
}

data {
    int <lower=2> K;                 // Number of components
    int <lower=1> Nb;                // Number of repeat baseline data
    int <lower=1> Np;                // Number of post-treatment data
    simplex[K] mu0;                  // Known mean baseline value
    array[Nb, K] int <lower=0> yb;   // Repeat baseline measurements
    array[Np, K] int <lower=0> yp;   // Study post-treatment measurements
    real <lower=0> prec_prior;       // The scale for the precision/concentration prior
}

parameters {
    real<lower=0> prec;              // Precision of the measurement
    real<lower=0> conc;              // Concentration of the post-treatment changes
    simplex[K] mu1;                  // Population average of post-treatment changes
    real<lower=0, upper=1> lambda;   // Proportion of measurements demonstrating significant change
}

model {
    // Priors
    prec ~ cauchy(0, prec_prior);
    conc ~ cauchy(0, prec_prior);
    lambda ~ uniform(0, 1);
    mu1 ~ dirichlet(rep_vector(1, K));

    // Repeat Baseline Likelihood
    yb ~ dirichlet_multinomial(mu0, prec);

    // Post-treatment Likelihood
    vector[Np] lp0;
    vector[Np] lp1;
    for (n in 1:Np) {
        lp0[n] = log1m(lambda) + dirichlet_multinomial_lpmf(yp[n] | mu0 * prec);
        lp1[n] = log(lambda) + dirichlet_multinomial_lpmf(yp[n] | mu1 * conc);
    }
    target += sum(log_sum_exp(lp0, lp1));
}

generated quantities {
    vector[Np] z;           // Model labels samples
    for (n in 1:Np) {
        real lp0 = log1m(lambda) + dirichlet_multinomial_lpmf(yp[n] | mu0 * prec);
        real lp1 = log(lambda) + dirichlet_multinomial_lpmf(yp[n] | mu1 * conc);
        z[n] = categorical_rng([exp(lp0 - log_sum_exp(lp0, lp1)),  exp(lp1 - log_sum_exp(lp0, lp1))]');
    }
}