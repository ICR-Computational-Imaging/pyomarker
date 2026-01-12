""" The fixed baseline habitat model """

import os
import sys
import pickle

from numpy.typing import NDArray
import numpy as np
from cmdstanpy import CmdStanModel
import arviz as az


class MixtureModel:
    """ A base class for performing mixture modelling based on double baseline measurements.

    Properties:
        num_chains (int): The number of MCMC chains to sample. Default is 3.
        num_samples (int): The number of MCMC samples per chain. Default is 1000.
        num_warmup (int): The number of MCMC samples to remove for burn-in.
        num_thin (int): The number of MCMC to discard for thinning.
        random_seed (int): The random seed. Default is 1984.

    """
    def __init__(self, num_chains: int = 3, num_samples: int = 1000, num_warmup: int = 500,
                 num_thin: int = 1, random_seed: int = 1984):
        self.num_chains = num_chains
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.num_thin = num_thin
        self.random_seed = random_seed
        self.__sampler__ = None
        self.__samples__ = None
        self.az_data = None

    def save_pickle(self, fname: str) -> None:
        """ Save to pickle file
        """
        if not fname.split(".")[-1] == "pkl":
            raise ValueError("fname must have .pkl extension")
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, fname: str):
        """ Load a saved model
        """
        if not os.path.exists(fname):
            raise FileNotFoundError(f"No file named {fname} found.")
        with open(fname, "rb") as f:
            model = pickle.load(f)
        return model

    def log_posterior_odds(self) -> NDArray:
        """ Estimated posterior odds from the output parameters

        Returns:
            NDArray: Shape (Np,) array of the estimated posterior odds
        """
        z = np.array(self.az_data.posterior["z"])  # Shape (num_chains, num_samples, Np)
        p0 = np.mean(z == 1, axis=(0, 1))
        p1 = np.mean(z == 2, axis=(0, 1))
        log_odds = np.log(p1) - np.log(p0)
        return log_odds

    def change_probability(self) -> NDArray:
        """ Estimated probability of change

        Returns:
            NDArray: Shape (Np,) array of estimated probabilities
        """
        z = np.array(self.az_data.posterior["z"])  # Shape (num_chains, num_samples, Np)
        p1 = np.mean(z == 2, axis=(0, 1))
        return p1

    @staticmethod
    def baseline_model() -> str:
        """ Return the stan model string. """
        raise NotImplementedError

    @staticmethod
    def stan_file() -> str:
        """ Return the stan file location. """
        raise NotImplementedError

    def stan_data_structure(self) -> dict:
        """ Return the data as a dictionary for passing into Stan sampling. """
        raise NotImplementedError

    def post_processing(self) -> None:
        """ Perform any necessary post-processing (e.g. computation of transformed parameters). """
        return

    def fit(self, quiet: bool = True) -> None:
        """ Fit the model.

        Args:
            quiet (bool): Whether to ignore output to screen during stan sampling.
        """
        try:
            with open("__baseline_model__.stan", "w") as f:
                f.write(self.baseline_model())
            self.__baseline_sampler__ = CmdStanModel(stan_file="__baseline_model__.stan")
            self.__baseline_stan_samples__ = self.__baseline_sampler__.sample(data=self.baseline_stan_data_structure(),
                                                                              chains=self.num_chains,
                                                                              iter_sampling=self.num_samples,
                                                                              thin=self.num_thin,
                                                                              iter_warmup=self.num_warmup,
                                                                              seed=self.random_seed)
            self.baseline_az_data = az.from_cmdstanpy(self.__baseline_stan_samples__)

            with open("__post_treatment_model__.stan", "w") as f:
                f.write(self.post_treatment_model())
            self.__post_treatment_sampler__ = CmdStanModel(stan_file="__post_treatment_model__.stan")
            self.__post_treatment_stan_samples__ = self.__post_treatment_sampler__.sample(
                data=self.post_treatment_stan_data_structure(),
                chains=self.num_chains,
                iter_sampling=self.num_samples,
                thin=self.num_thin,
                iter_warmup=self.num_warmup,
                seed=self.random_seed)
            self.post_treatment_az_data = az.from_cmdstanpy(self.__post_treatment_stan_samples__)

            self.post_processing()

        finally:
            if quiet:
                self.__teardown_stdout__()
