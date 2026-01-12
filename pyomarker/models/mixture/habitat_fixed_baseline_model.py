""" The fixed baseline habitat model """

from numpy.typing import NDArray
import numpy as np
import xarray
from scipy.stats import dirichlet, multinomial

from mixture_model import MixtureModel


class HabitatFixedBaselineModel(MixtureModel):
    """ Assumes that the baseline measurement is fixed (i.e. mu0 is fixed).

    Assumptions of this model:
        1. The baseline value is fixed (mu0 is known).
        2. Measurement error is modelled as a dirichlet-multinomial distribution.

    """
    def __init__(self, full_model: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu0 = None
        self.prec_prior = None
        self.labels_post = None

    @staticmethod
    def model_definition() -> str:
        """ Return the stan model string. """
        return __model_baseline__

    def post_treatment_model(self) -> str:
        """ Return the stan model string. """
        if self.full_model:
            return __model_post_treatment_full__
        else:
            return __model_post_treatment__

    def baseline_stan_data_structure(self) -> dict:
        """ Return the baseline data as a dictionary for passing into Stan sampling. """
        if self.yb is None:
            raise ValueError("No data has yet been set.")
        return {"Nb": self.yb.shape[0], "K": self.yb.shape[1], "yb": self.yb, "mu0": self.mu0,
                "prec_prior": self.prec_prior}

    def post_treatment_stan_data_structure(self) -> dict:
        """ Return the post-treatment data as a dictionary for passing into Stan sampling. """
        if self.yp is None:
            raise ValueError("No data has yet been set.")
        return {"Nb": self.yb.shape[0], "yb": self.yb, "Np": self.yp.shape[0], "yp": self.yp,
                "K": self.yb.shape[1], "mu0": self.mu0, "prec_prior": self.prec_prior}

    def posterior_predictive_samples(self, n_voxels: int, normalised=True) -> NDArray:
        """ Posterior samples from the Null model.

        Args:
            n_voxels (int): Assumed number of voxels.
            normalised (bool): Whether to normalise the posterior samples.

        Returns:
            NDArray: Posterior samples, shape (n_samples, k).
        """
        samples = []
        prec_samples = np.array(self.baseline_az_data.posterior["prec"]).ravel()
        for prec in prec_samples:
            x_pred = dirichlet.rvs(self.mu0 * prec)[0]
            y_pred = multinomial.rvs(n_voxels, x_pred)
            if normalised:
                y_pred[y_pred == 0] = 1
                y_pred = y_pred / np.sum(y_pred)
            samples.append(y_pred)
        return np.array(samples)

    def set_data(self, yb: NDArray, yp: NDArray, mu0: NDArray, labels_post: NDArray = None,
                 prec_prior: float = 20) -> None:
        """
        Args:
            yb (NDArray): A shape (Nb, K) array representing the baseline data, where Nb
                is the number of observations and K represents the number of habitats.
            yp (NDArray): A shape (Np, K) array representing the post-treatment data with Np
                observations.
            mu0 (NDArray): A shape (K,) array representing the known ground truth. Default is None
                in which case it is assumed to be a simplex with equal entries (1/K).
            labels_post (NDArray): A shape (Np,) array representing the post-treatment labels with
                each element being a string. Default is None, in which case labels are numbered.
            prec_prior (float): The scale for the precision/concentration priors. Default is 20.
        """

        # Store the data
        self.yb = np.array(yb).astype("int")
        self.yp = np.array(yp).astype("int")
        if labels_post is None:
            labels_post = [i + 1 for i in range(self.yp.shape[0])]
        self.labels_post = np.array(labels_post)
        self.mu0 = np.array(mu0)
        self.prec_prior = prec_prior

        # Perform checks
        if self.yb.ndim != 2 or self.yp.ndim != 2:
            raise ValueError("Inputs must be 2-dimensional")
        dims = self.yb.shape[1]
        if self.yp.shape[1] != dims:
            raise ValueError("y_post has incompatible shape")
        if self.mu0.shape != (dims,):
            raise ValueError("mu0 has incompatible shape")

    def post_processing(self):
        """ Compute the IMS. """
        prec = np.array(self.post_treatment_az_data.posterior["prec"])  # (n_chains, n_samples)
        conc = np.array(self.post_treatment_az_data.posterior["conc"])  # (n_chains, n_samples)
        ims = prec / (prec + conc)

        # Add to the az posterior as instances of xarray (don't forget to add an empty axis for single parameters)
        self.post_treatment_az_data.posterior["ims"] = xarray.DataArray(ims, dims=("chain", "draw"))
