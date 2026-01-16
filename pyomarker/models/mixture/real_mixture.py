""" The fixed baseline habitat model """

from numpy.typing import NDArray
import numpy as np
import xarray

from src.stan_models.mixture_model import MixtureModel

__model_baseline__ = """
data {
    int <lower=1> Nb;          // Number of repeat baseline data
    vector[Nb] yb1;            // Baseline measurement 1
    vector[Nb] yb2;            // Baseline measurement 2
    real <lower=0> sd_prior;   // Std-dev for prior of standard deviations
    real <lower=0> mu_prior;   // Limit for maximum allowable ADC
}

transformed data {
    vector[Nb] db = yb2 - yb1;
    vector[Nb] mb = (yb1 + yb2) / 2;
}

parameters {
    real<lower=0> sdr;          // Measurement error
    real<lower=0> sd0;          // Std-dev of baseline measurements
    real mu0;                   // Population baseline value
}

model {
    // Priors
    sdr ~ cauchy(0, sd_prior);
    sd0 ~ cauchy(0, sd_prior);
    mu0 ~ normal(0, mu_prior);

    // Repeat Baseline Likelihood
    db ~ normal(0, sqrt(2) * sdr);
    mb ~ normal(mu0, sqrt(square(sd0) + square(sdr)/2));
}
"""

__model_post_treatment__ = """
data {
    int <lower=1> Nb;          // Number of repeat baseline data
    int <lower=1> Np;          // Number of post-treatment data
    vector[Nb] yb1;            // Baseline measurement 1
    vector[Nb] yb2;            // Baseline measurement 2
    vector[Np] yp1;            // Post-treatment measurement 1
    vector[Np] yp2;            // Post-treatment measurement 2
    real <lower=0> sd_prior;   // Std-dev for prior of standard deviations
    real <lower=0> mu_prior;   // Limit for maximum allowable ADC
}

transformed data {
    vector[Nb] db = yb2 - yb1;
    vector[Nb] mb = (yb1 + yb2) / 2;
    vector[Np] dp = yp2 - yp1;
}

parameters {
    real <lower=0> sdr;              // Measurement error
    real <lower=0> sd0;              // Std-dev of baseline measurements
    real <lower=0> sdd;              // Std-dev of differences
    real mu0;                        // Population baseline value
    real mud;                        // Population mean difference
    real <lower=0, upper=1> lambda;  // Proportion of measurements demonstrating significant change
}

model {
    // Priors
    sdr ~ cauchy(0, sd_prior);
    sd0 ~ cauchy(0, sd_prior);
    sdd ~ cauchy(0, sd_prior);
    mu0 ~ normal(0, mu_prior);
    mud ~ normal(0, mu_prior);
    lambda ~ uniform(0, 1);

    // repeat baseline likelihood
    db ~ normal(0, sqrt(2) * sdr);
    mb ~ normal(mu0, sqrt(square(sd0) + square(sdr)/2));

    // post-treatment likelihood
    vector[Np] lp0;
    vector[Np] lp1;
    for (n in 1:Np) {
        lp0[n] = log1m(lambda) + normal_lpdf(dp[n] | 0, sqrt(2) * sdr);
        lp1[n] = log(lambda) + normal_lpdf(dp[n] | mud, sqrt(square(sdd) + square(sdr) * 2));
    }
    target += sum(log_sum_exp(lp0, lp1));
}

generated quantities {
    vector[Np] z; 
    for (n in 1:Np){
        real lp0 = log1m(lambda) + normal_lpdf(dp[n] | 0, sqrt(2) * sdr);
        real lp1 = log(lambda) + normal_lpdf(dp[n] | mud, sqrt(square(sdd) + square(sdr) * 2));
        z[n] = categorical_rng([exp(lp0 - log_sum_exp(lp0, lp1)),  exp(lp1 - log_sum_exp(lp0, lp1))]');
    }
}
"""


class ADCModel(MixtureModel):
    """ Assumes that the baseline measurement is fixed (i.e. mu0 is fixed).

    Assumptions of this model:
        1. The baseline value is fixed (mu0 is known).
        2. Measurement error is modelled as a dirichlet-multinomial distribution.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yb1 = None
        self.yb2 = None
        self.yp1 = None
        self.yp2 = None
        self.sd_prior = None
        self.mu_prior = None
        self.labels_post = None

    @staticmethod
    def baseline_model() -> str:
        """ Return the stan model string. """
        return __model_baseline__

    @staticmethod
    def post_treatment_model() -> str:
        """ Return the stan model string. """
        return __model_post_treatment__

    def baseline_stan_data_structure(self) -> dict:
        """ Return the baseline data as a dictionary for passing into Stan sampling. """
        if self.yb1 is None:
            raise ValueError("No data has yet been set.")
        return {"Nb": self.yb1.shape[0], "yb1": self.yb1, "yb2": self.yb2, "sd_prior": self.sd_prior,
                "mu_prior": self.mu_prior}

    def post_treatment_stan_data_structure(self) -> dict:
        """ Return the post-treatment data as a dictionary for passing into Stan sampling. """
        if self.yp1 is None:
            raise ValueError("No data has yet been set.")
        return {"Nb": self.yb1.shape[0], "yb1": self.yb1, "yb2": self.yb2, "Np": self.yp1.shape[0], "yp1": self.yp1,
                "yp2": self.yp2, "sd_prior": self.sd_prior, "mu_prior": self.mu_prior}

    def set_data(self, yb1: NDArray, yb2: NDArray, yp1: NDArray, yp2: NDArray, labels_post: NDArray = None,
                 sd_prior: float = 20, mu_prior: float = 100) -> None:
        """
        Args:
            yb1 (NDArray): A shape (Nb,) array representing the first baseline data.
            yb2 (NDArray): A shape (Nb,) array representing the second baseline data.
            yp1 (NDArray): A shape (Np,) array representing the first post-treatment data.
            yp2 (NDArray): A shape (Np,) array representing the second post-treatment data.
            labels_post (NDArray): A shape (Np,) array representing the post-treatment labels with
                each element being a string. Default is None, in which case labels are numbered.
            sd_prior (float): The standard deviation of the standard deviation priors.
            mu_prior (float): The standard deviation of the mean deviation priors.
        """

        # Store the data
        self.yb1 = np.array(yb1).astype("float")
        self.yb2 = np.array(yb2).astype("float")
        self.yp1 = np.array(yp1).astype("float")
        self.yp2 = np.array(yp2).astype("float")
        self.sd_prior = sd_prior
        self.mu_prior = mu_prior
        if labels_post is None:
            labels_post = [i + 1 for i in range(self.yp1.shape[0])]
        self.labels_post = np.array(labels_post)

    def post_processing(self):
        """ Compute the ICC and IMS. """
        sdr = np.array(self.post_treatment_az_data.posterior["sdr"])  # (n_chains, n_samples)
        sdd = np.array(self.post_treatment_az_data.posterior["sdd"])  # (n_chains, n_samples)
        sd0 = np.array(self.post_treatment_az_data.posterior["sd0"])  # (n_chains, n_samples)
        icc = sd0 ** 2 / (sd0 ** 2 + sdr ** 2)
        ims = sdd ** 2 / (sdd ** 2 + sdr ** 2)

        # Add to the az posterior as instances of xarray (don't forget to add an empty axis for single parameters)
        self.post_treatment_az_data.posterior["icc"] = xarray.DataArray(icc, dims=("chain", "draw"))
        self.post_treatment_az_data.posterior["ims"] = xarray.DataArray(ims, dims=("chain", "draw"))
