""" Functionality for fitting a Dirichlet mixture model """

import math

import numpy as np
from numpy.typing import NDArray
import scipy
from scipy.stats import dirichlet
from scipy.special import digamma, polygamma
from scipy.optimize import minimize

from coordinate_conversions import polar2cartesian, cartesian2polar


class DirichletMixture:
    """ Calculate a mixture model for a number of Dirichlet distributions.

    This uses the Expectation Maximization algorithm for computation.

    """
    def __init__(self, m_components: int = 2, max_iters: int = 100, atol: float = 1e-6):
        self.max_iters = max_iters
        self.m_components = m_components
        self.atol = atol
        self.alphas = None
        self.weights = None
        self._x = None

    @property
    def m_components(self) -> int:
        """ The number of components in the mixture model. """
        return self._m_components

    @m_components.setter
    def m_components(self, m_components: int):
        if m_components < 1:
            raise ValueError("Value cannot be < 1")
        self._m_components = m_components

    @property
    def max_iters(self) -> int:
        """ The maximum number of iterations for the EM algorithm """
        return self._max_iters

    @max_iters.setter
    def max_iters(self, max_iters: int):
        if max_iters < 1:
            raise ValueError("Value cannot be < 1")
        self._max_iters = max_iters

    @property
    def atol(self) -> float:
        """ The absolute tolerance to test for convergence (minimum across all parameters) """
        return self._atol

    @atol.setter
    def atol(self, atol: float):
        if atol < 0:
            raise ValueError("Value cannot be negative")
        self._atol = atol

    @property
    def alphas(self) -> NDArray:
        """ The component alphas derived after fitting. None if not yet fit. """
        return self._alphas

    @alphas.setter
    def alphas(self, alphas: NDArray):
        if alphas is not None:
            if np.any(alphas <= 0):
                raise ValueError("All alpha values must be > 0")
        self._alphas = alphas

    @property
    def weights(self) -> NDArray:
        """ The component weights derived after fitting. None if not yet fit. """
        return self._weights

    @weights.setter
    def weights(self, weights: NDArray):
        if weights is not None:
            if np.any(weights < 0):
                raise ValueError(f"All weights must be > 0. Current values = {weights}")
            if not np.isclose(np.sum(weights), 1.0, atol=1e-5):
                raise ValueError(f"All weights must sum to 1. Current sum = {np.sum(weights)}")
        self._weights = weights

    def __maximize_function__(self, p: NDArray) -> float:
        """ The objective function tal call during optimization.

        Args:
            p (NDArray): The polar coordinates at which to calculate the objective function.

        Returns:
            float: The inverse of the log-likelihood at position p.
        """

        # Convert to cartesian
        x = polar2cartesian(p[None, :])

        return 1 - np.clip(np.log(self.score_samples(x)), 1e-10, np.inf)

    def __check_fit__(self) -> bool:
        """ Check whether the model has been fit

        Returns:
            bool
        """
        fit = True
        if self.alphas is None:
            fit = False
        return fit

    @staticmethod
    def __valid_simplex__(x: NDArray) -> bool:
        """ Check that input samples are all valid simplexes (sum to 1)

        Args:
            x (NDArray): Shape (N, K) array of the samples to check.

        Returns:
            bool

        """
        bok = True
        if np.any(x < 0):
            bok = False
        if not np.allclose(np.sum(x, axis=1), 1):
            bok = False

        return bok

    @staticmethod
    def __moments_dirichlet_fit__(x: NDArray) -> NDArray:
        """ Fit parameters a dirichlet distribution using method of moments.

        Args:
            x (NDArray): Shape (N, K) array containing N observed simplex points (data).

        Returns:
            NDArray: Shape (K, ) alpha array for the dirichlet distribution describing x.

        """
        # Check data validity
        if not np.allclose(np.sum(x, axis=1), 1):
            raise ValueError("Input data must consist of simplexes.")

        e1 = np.mean(x, axis=0)
        e2 = np.mean(x ** 2, axis=0)
        alpha_0 = (e1 - e2) / (e2 - e1 ** 2)
        return alpha_0 * e1

    @staticmethod
    def __component_weights__(component_posteriors: NDArray) -> NDArray:
        """ Compute prior component probabilities (weights)

        Args:
            component_posteriors (NDArray) : Array of shape (N, M) where there are N data and
                M possible component labels.

        Returns:
            NDArray: Shape (M, ) array containing the prior probability for each component.

        """
        weights = np.mean(component_posteriors, axis=0)
        weights = weights / np.sum(weights)
        return weights

    @staticmethod
    def __component_posteriors__(alphas: NDArray, component_weights: NDArray,
                                 x: NDArray) -> NDArray:
        """ Compute the component (label) posterior probabilities

        Args:
            alphas (NDArray): Shape (M, K) array of Dirichlet distribution parameters, where there
                are M components and K data dimensions.
            component_weights (np.NDArray): Shape (M, ) array of prior component probabilities.
            x (np.NDArray): Shape (N, K) array of observed data.

        Returns:
            NDArray: Shape (N, M) array

        """
        m_components = component_weights.shape[0]
        posterior = np.array(
            [dirichlet.pdf(x.T, alphas[m]) * component_weights[m] for m in range(m_components)])
        posterior = posterior / np.sum(posterior, axis=0)
        return posterior.T

    def __component_alpha__(self, component_posteriors: NDArray, x: NDArray,
                            alpha_init: NDArray = None, step_size: float = 1.0) -> NDArray:
        """ Compute the MLE of alpha for a single component

        This is computed based on the data and the component posterior for each datum.

        Args:
            component_posteriors (NDArray): Shape (N, ) array containing the posterior probability
                of the component for all data.
            x (NDArray): Shape (N, K) array containing the observed data.
            alpha_init (NDArray): Shape (K, ) array representing the current estimate of alpha
            step_size (float): The step size of the Newton-Raphson optimization.

        Returns:
            np.NDArray: Shape (K) array of the

        """
        # Get problem dimensions
        n_samples, k_dims = x.shape

        # Check input values
        if not np.allclose(component_posteriors.shape, (n_samples,)):
            raise ValueError(f"Incompatible dimensions {x.shape} and {component_posteriors.shape}")

        if step_size < 0:
            raise ValueError("step_size must be > 0")

        if not DirichletMixture.__valid_simplex__(x):
            raise ValueError("Input data must consist of simplexes.")

        # Initialise values
        x_bar = np.mean(np.log(x).T * component_posteriors, axis=1)
        w = np.mean(component_posteriors)  # The component prior weight

        if alpha_init is None:
            alpha = np.ones(k_dims) * 10
        else:
            if np.any(alpha_init < 0):
                raise ValueError("All alpha values must be > 0")
            if not np.allclose(alpha_init.shape, (k_dims,)):
                raise ValueError(f"Incompatible dimensions {x.shape} and {alpha_init.shape}")
            alpha = alpha_init

        # Iterate
        for _ in range(self.max_iters):
            q = -n_samples * w * polygamma(1, alpha)
            z = n_samples * w * polygamma(1, np.sum(alpha))
            g = n_samples * w * (digamma(np.sum(alpha)) - digamma(alpha)) + n_samples * x_bar
            b = np.sum(g / q) / ((1.0 / z) + np.sum(1.0 / q))
            alpha_new = alpha - step_size * (g - b) / q
            if np.max(np.abs(alpha_new - alpha)) < self.atol:
                break
            alpha = alpha_new
        return alpha

    def __maximize_likelihood_grid_search__(self, samples=100000,
                                            bounds_offset: float = 1e-3) -> NDArray:
        """ Determine the simplex with maximum marginal likelihood using a grid search

        Args:
            samples (NDArray): The total number of samples for the grid search.
            bounds_offset (float): The offset for the bounds search in polar coordinates
                (bounds_offset, pi / 2 - bounds_offset)

        Returns:
            NDArray: The simplex defining the position of maximum marginal likelihood
        """

        if not self.__check_fit__():
            raise RuntimeError("Model not yet fit")

        # Determine the grid size along each simplex dimension
        k_dims = self.alphas.shape[1]
        grid_size = math.ceil(np.power(samples, 1.0 / (k_dims-1)))

        # Create grid of equidistant polar coordinates
        p = np.meshgrid(*[np.linspace(bounds_offset, np.pi / 2 - bounds_offset, grid_size)
                          for _ in range(k_dims-1)])
        p = np.c_[[p.ravel() for p in p]].T

        # Convert to cartesian and find optimum
        x = polar2cartesian(p)
        z = self.score_samples(x)
        x_opt = x[np.argmax(z), :]
        return x_opt

    def __maximize_likelihood__(self, init_samples=100000, bounds_offset: float = 1e-5) -> NDArray:
        """ Determine the simplex with maximum marginal likelihood

        Args:
            init_samples (NDArray): The total number of samples for initial grid search.
            bounds_offset (float): The offset for the bounds search in polar coordinates
                (bounds_offset, pi / 2 - bounds_offset)

        Returns:
            NDArray: The simplex defining the position of maximum marginal likelihood
        """

        if not self.__check_fit__():
            raise RuntimeError("Model not yet fit")

        k_dims = self.alphas.shape[1]

        # Perform initial grid search
        x_init = self.__maximize_likelihood_grid_search__(samples=init_samples,
                                                          bounds_offset=bounds_offset)
        p_init = cartesian2polar(x_init[None, :])[0]

        # Use Nelder-Mead minimization in polar coordinates for better bounds.
        result = minimize(self.__maximize_function__, p_init, method="Nelder-Mead",
                          bounds=[(bounds_offset, np.pi / 2 - bounds_offset)
                                  for _ in range(k_dims-1)])
        p_opt = result.x
        x_opt = polar2cartesian(p_opt[None, :])
        return x_opt

    def score_samples(self, x: NDArray):
        """ Compute the (marginal) likelihood of each sample

        Args:
            x (NDArray): An array of shape (n_samples, k_dims) of the samples.  Note that each
                sample should be a simplex (add to 1).

        Returns:
            NDArray: Shape (n_samples,) array of the log-likelihoods.

        """
        # Check data
        if not DirichletMixture.__valid_simplex__(x):
            raise ValueError("Input samples are not valid simplexes")

        if not self.__check_fit__():
            raise RuntimeError("Model not yet fit")

        z = np.array([dirichlet.pdf(x.T, self.alphas[m]) * self.weights[m]
                      for m in range(self.m_components)])
        z = np.sum(z, axis=0)
        return z

    def p_values(self, x_new: NDArray) -> NDArray:
        """ Calculate the p-value of new data using the 'kernel' approach

        Args:
            x_new (NDArray): The input data of shape (n_samples, k_dims) for which to calculate the
                p-values.

        Returns:
            NDArray: The p-values as shape (n_samples, ).

        """
        # Check data validity
        if not DirichletMixture.__valid_simplex__(x_new):
            raise ValueError("Input samples are not valid simplexes")

        if not self.__check_fit__():
            raise RuntimeError("Model not yet fit")

        # Determine the maximum marginal density
        x_max = self.__maximize_likelihood__(init_samples=10000)
        p_max = self.score_samples(x_max)

        # The (sorted) kernel distance of original data
        marginal = self.score_samples(self._x)
        k_dist = np.sort(np.clip(0, np.inf, p_max - marginal))

        # The normalised cumulative distribution
        k_dist_cum = np.cumsum(k_dist) / np.sum(k_dist)

        # The kernel distance of new data
        marginal = self.score_samples(x_new)
        k_dist_new = np.clip(0, np.inf, p_max - marginal)

        # Determine the percentiles of the new data compared to original
        interpolator = scipy.interpolate.interp1d(k_dist, k_dist_cum, fill_value=(0, 1),
                                                  assume_sorted=True, bounds_error=False,
                                                  kind="linear")
        p_values = 1 - interpolator(k_dist_new)

        return p_values

    def fit(self, x: NDArray):
        """ Fit the mixture model to some data

        Args:
            x (NDArray): The array of shape (N, K) to fit the model to. Note that each of the N
                data should be simplexes (sum to 1).

        """
        # Check data
        if not DirichletMixture.__valid_simplex__(x):
            raise ValueError("Input samples are not valid simplexes")

        # Initial estimate of alpha
        alpha_init = DirichletMixture.__moments_dirichlet_fit__(x)
        mean_init = dirichlet.rvs(alpha_init, size=self.m_components)
        alpha_0_init = np.sum(alpha_init) * self.m_components
        alphas = mean_init * alpha_0_init
        weights = np.ones(self.m_components) * self.m_components

        # Expectation maximization
        for _ in range(self.max_iters):
            posteriors = DirichletMixture.__component_posteriors__(alphas, weights, x)
            weights = DirichletMixture.__component_weights__(posteriors)
            for m in range(self.m_components):
                alpha = alphas[m]
                posterior = posteriors[:, m]
                alphas[m] = self.__component_alpha__(posterior, x, alpha_init=alpha, step_size=0.01)

        # Store the results
        self.alphas = alphas
        self.weights = weights
        self._x = x
