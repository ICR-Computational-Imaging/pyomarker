"""Tools for classical repeatability calculations"""

from typing import Tuple, Dict, Any

import numpy as np
from numpy.typing import NDArray
import scipy.stats


class BlandAltman:
    def __init__(self, ci = 0.9):
        self._ci = BlandAltman._validate_ci(ci)
        self._x1 = None
        self._x2 = None
        self._differences = None
        self._means = None

    @property
    def x1(self):
        if self._x1 is None:
            raise RuntimeError("Model not fit. Call fit(x1, x2) first.")
        return self._x1

    @property
    def x2(self):
        if self._x2 is None:
            raise RuntimeError("Model not fit. Call fit(x1, x2) first.")
        return self._x2

    @property
    def ci(self):
        return self._ci
    @ci.setter
    def ci(self, value):
        self._ci = BlandAltman._validate_ci(value)

    @staticmethod
    def _validate_paired_1d(x1: NDArray, x2: NDArray) -> tuple[np.ndarray, np.ndarray]:
        """Validate and normalize paired 1D inputs.

        Ensures:
        - both inputs are 1D
        - same length
        - finite
        - cast to float ndarray

        Parameters:
            x1 (NDArray): 1D array of baseline measurements.
            x2 (NDArray): 1D array of repeat baseline measurements.

        Returns:
            (x1, x2): validated NumPy arrays
        """
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        if x1.ndim != 1 or x2.ndim != 1:
            raise ValueError("x1 and x2 must be 1D arrays.")
        if x1.size != x2.size:
            raise ValueError("x1 and x2 must have the same size.")
        if np.any(~np.isfinite(x1)) or np.any(~np.isfinite(x2)):
            raise ValueError("x1 and x2 must contain only finite values.")

        return x1, x2

    @staticmethod
    def _validate_ci(ci: float) -> float:
        """ Validate that a confidence interval is valid.

        Ensures:
        - 0 < ci < 1
        - ci is a float

        Parameters:
            ci (float): confidence interval

        Returns:
            ci (float): confidence interval
        """
        try:
            ci = float(ci)
        except (TypeError, ValueError):
            raise ValueError("ci must be a float that satisfies 0 < ci < 1.")
        if not (0.0 < ci < 1.0):
            raise ValueError("ci must be a float that satisfies 0 < ci < 1.")
        return ci

    @staticmethod
    def within_subject_mean(x1: NDArray, x2: NDArray) -> NDArray:
        r""" Per subject mean.

        Definition
        ==========
        $$
        \bar{x}_{i} = \frac{1}{2}\,(x_{1i} + x_{2i})
        $$

        Parameters:
            x1 (NDArray): 1D array of baseline measurements.
            x2 (NDArray): 1D array of repeat baseline measurements.

        Returns:
            NDArray: 1D array within subject means.
        """
        x1, x2 = BlandAltman._validate_paired_1d(x1, x2)
        return 0.5 * (x1 + x2)

    @staticmethod
    def population_mean(x1: NDArray, x2: NDArray) -> float:
        r""" Population mean of paired measurements.

        Definition
        ==========
        $$
        \mu = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{2}\,(x_{1i} + x_{2i})
        $$

        Parameters:
            x1 (NDArray): 1D array of baseline measurements.
            x2 (NDArray): 1D array of repeat baseline measurements.

        Returns:
            float: The population mean.
        """
        x1, x2 = BlandAltman._validate_paired_1d(x1, x2)
        return float(np.mean(BlandAltman.within_subject_mean(x1, x2)))


    @staticmethod
    def between_subject_mean_squares(x1: NDArray, x2: NDArray) -> float:
        r""" Between subject, mean squares of paired measurements.

        Definition
        ==========
        $$
        bsms = \frac{2}{N} \sum_{i=1}^{N} (\bar{x}_{i} - \mu)^{2}
        $$

        Parameters:
            x1 (NDArray): 1D array of baseline measurements.
            x2 (NDArray): 1D array of repeat baseline measurements.

        Returns:
            float: bsms.
        """
        x1, x2 = BlandAltman._validate_paired_1d(x1, x2)
        n = x1.size
        w_mean = BlandAltman.within_subject_mean(x1, x2)
        p_mean = BlandAltman.population_mean(x1, x2)
        bsms = (2.0 / n) * np.sum((w_mean - p_mean) ** 2)
        return bsms

    @staticmethod
    def within_subject_mean_squares(x1: NDArray, x2: NDArray) -> float:
        r""" Within subject, mean squares of paired measurements.

        Definition
        ==========
        $$
        wsms = \frac{1}{N} \sum_{i=1}^{N} (x_{1i} - \bar{x}_{i})^{2} + (x_{2i} - \bar{x}_{i})^{2}
        $$

        Parameters:
            x1 (NDArray): 1D array of baseline measurements.
            x2 (NDArray): 1D array of repeat baseline measurements.

        Returns:
            float: wsms
        """
        x1, x2 = BlandAltman._validate_paired_1d(x1, x2)
        n = x1.size
        w_mean = BlandAltman.within_subject_mean(x1, x2)
        wsms = np.sum((x1 - w_mean) ** 2 + (x2 - w_mean) ** 2) / n
        return wsms

    @staticmethod
    def between_subject_standard_deviation(x1: NDArray, x2: NDArray) -> float:
        r""" Between subject standard deviation of paired measurements.

        Definition
        ==========
        $$
        \sigma_{b} = \sqrt{(bsms - wsms) / 2}
        $$

        Parameters:
            x1 (NDArray): 1D array of baseline measurements.
            x2 (NDArray): 1D array of repeat baseline measurements.

        Returns:
            float: The standard deviation.
        """
        x1, x2 = BlandAltman._validate_paired_1d(x1, x2)
        bsms = BlandAltman.between_subject_mean_squares(x1, x2)
        wsms = BlandAltman.within_subject_mean_squares(x1, x2)
        sb = np.sqrt(max((bsms - wsms) / 2, 0.0))
        return sb

    @staticmethod
    def within_subject_standard_deviation(x1: NDArray, x2: NDArray, ci: float = 0.9) -> Tuple[float, NDArray]:
        r""" Within subject standard deviation of paired measurements.

        Definition
        ==========
        $$
        \sigma_{w} = \sqrt{\frac{1}{2N}\sum_{i=1}^{N}(x_{2i} - x_{1i})^{2}}
        $$

        Parameters:
            x1 (NDArray): 1D array of baseline measurements.
            x2 (NDArray): 1D array of repeat baseline measurements.
            ci (float): The confidence interval.  Must be between 0 and 1. Defaults to 0.9.

        Returns:
            float: The within subject standard deviation.
            NDArray: The within subject standard deviation confidence interval.  Structure: (lower, upper).
        """
        x1, x2 = BlandAltman._validate_paired_1d(x1, x2)
        ci = BlandAltman._validate_ci(ci)

        d = x2 - x1
        n = x1.size
        sw = np.sqrt(np.sum(d ** 2) / (2.0 * n))

        alpha = (1.0 - ci) / 2.0
        chi2_lo = scipy.stats.chi2.ppf(alpha, n)
        chi2_hi = scipy.stats.chi2.ppf(1.0 - alpha, n)

        sw_ci = np.array([
            sw * np.sqrt(n / chi2_hi),  # lower
            sw * np.sqrt(n / chi2_lo),  # upper
        ])
        return sw, sw_ci

    @staticmethod
    def coefficient_of_variation(x1: NDArray, x2: NDArray) -> float:
        r""" The coefficient of variation on raw paired baseline measurements.

        Definition
        ==========
        $$
        CoV = \sigma_{w} / \mu \times 100\%
        $$

        Parameters:
            x1 (NDArray): 1D array of baseline measurements.
            x2 (NDArray): 1D array of repeat baseline measurements.

        Returns:
            float: The coefficient of variation.

        """
        x1, x2 = BlandAltman._validate_paired_1d(x1, x2)
        sw, _ = BlandAltman.within_subject_standard_deviation(x1, x2)
        p_mean = BlandAltman.population_mean(x1, x2)
        cov = sw / p_mean * 100.0
        return cov

    @staticmethod
    def ratio_coefficient_of_variation(x1: NDArray, x2: NDArray) -> float:
        r""" The coefficient of variation on log-transformed paired baseline measurements.

        Definition
        ==========
        $$
        CoVr = \sqrt{e^{\sigma_{w}^{2}} - 1} \times 100\%
        $$
        where $\sigma_{w}$ is the within subject standard deviation of the logarithm of paired baseline measurements.

        Parameters:
            x1 (NDArray): 1D array of baseline measurements.
            x2 (NDArray): 1D array of repeat baseline measurements.

        Raises:
            ValueError: If any elements of x1 or x2 are less than or equal to 0.

        Returns:
            float: The (ratio) coefficient of variation.

        """
        x1, x2 = BlandAltman._validate_paired_1d(x1, x2)
        if np.any(x1 <= 0) or np.any(x2 <= 0):
            raise ValueError("x1 and x2 must be > 0.")
        x1 = np.log(x1)
        x2 = np.log(x2)
        sw, _ = BlandAltman.within_subject_standard_deviation(x1, x2)
        cov = 100.0 * np.sqrt(np.exp(sw ** 2) - 1)
        return cov

    @staticmethod
    def coefficient_of_repeatability(x1: NDArray, x2: NDArray, ci: float = 0.9) -> Tuple[float, NDArray]:
        r""" The coefficient of repeatability of paired baseline measurements.

        Definition
        ==========
        $$
        r = 1.96 \times \sqrt{2} \times s_{w}
        $$

        Parameters:
            x1 (NDArray): 1D array of baseline measurements.
            x2 (NDArray): 1D array of repeat baseline measurements.
            ci (float): The confidence interval.  Must be between 0 and 1. Defaults to 0.9.

        Returns:
            float: The coefficient of repeatability.
            NDArray: The coefficient of repeatability confidence interval. Structure: (lower, upper).
        """
        x1, x2 = BlandAltman._validate_paired_1d(x1, x2)
        ci = BlandAltman._validate_ci(ci)

        sw, sw_ci = BlandAltman.within_subject_standard_deviation(x1, x2, ci)
        r = 1.96 * np.sqrt(2) * sw
        r_ci = 1.96 * np.sqrt(2) * sw_ci
        return r, r_ci

    @staticmethod
    def intraclass_correlation_coefficient(x1: NDArray, x2: NDArray, ci: float = 0.9) -> Tuple[float, NDArray]:
        r""" The intra-class correlation coefficient of paired baseline measurements.

        Definition
        ==========
        $$
        ICC = \frac{\sigma_{b}^{2}}{\sigma_{b}^{2} + \sigma_{w}^{2}}
        $$

        Parameters:
            x1 (NDArray): 1D array of baseline measurements.
            x2 (NDArray): 1D array of repeat baseline measurements.
            ci (float): The confidence interval.  Must be between 0 and 1. Defaults to 0.9.

        Returns:
            float: The coefficient of repeatability.
            NDArray: The coefficient of repeatability confidence interval. Structure: (lower, upper).
        """
        x1, x2 = BlandAltman._validate_paired_1d(x1, x2)
        ci = BlandAltman._validate_ci(ci)

        sb = BlandAltman.between_subject_standard_deviation(x1, x2)
        sw, _ = BlandAltman.within_subject_standard_deviation(x1, x2)
        icc = sb ** 2 / (sb ** 2 + sw ** 2)

        wsms = BlandAltman.within_subject_mean_squares(x1, x2)
        bsms = BlandAltman.between_subject_mean_squares(x1, x2)

        f0 = bsms / wsms
        n = x1.size
        q = 1 - (1 - ci) / 2
        f_lower = f0 / scipy.stats.f.ppf(q, n - 1, n)
        f_upper = f0 * scipy.stats.f.ppf(q, n, n - 1)
        icc_ci = np.array([(f_lower - 1) / (f_lower + 1), (f_upper - 1) / (f_upper + 1)])

        return icc, icc_ci

    @staticmethod
    def ratio_limits_of_agreement(x1: NDArray, x2: NDArray, ci: float = 0.9) -> Tuple[NDArray, NDArray]:
        r""" The limits of agreement derive from log-transformed paired baseline measurements.

        Definition
        ==========
        $$
        LoA = [\exp(±1.96\times \sqrt{2} \times s_{w}) - 1] \times 100\%
        $$
        where $\sigma_{w}$ is the within subject standard deviation of the logarithm of paired baseline measurements.

        Parameters:
            x1 (NDArray): 1D array of baseline measurements.
            x2 (NDArray): 1D array of repeat baseline measurements.
            ci (float): The confidence interval.  Must be between 0 and 1. Defaults to 0.9.

        Raises:
            ValueError: If any elements of x1 or x2 are less than or equal to 0.

        Returns:
            NDArray: The (ratio) limits of agreement. Structure: (negative, positive).
            NDArray: The (ratio) limits of agreement confidence interval.
                Structure: (negative/lower, negative/upper, positive/lower, positive/upper).

        """
        x1, x2 = BlandAltman._validate_paired_1d(x1, x2)
        ci = BlandAltman._validate_ci(ci)
        if np.any(x1 <= 0) or np.any(x2 <= 0):
            raise ValueError("x1 and x2 must be greater than 0")

        x1_log = np.log(x1)
        x2_log = np.log(x2)
        sw_log, sw_log_ci = BlandAltman.within_subject_standard_deviation(x1_log, x2_log, ci)

        loa = np.r_[np.exp(-1.96 * sw_log * np.sqrt(2)) - 1,
                    np.exp(+1.96 * sw_log * np.sqrt(2)) - 1] * 100.
        loa_ci = np.r_[np.exp(-1.96 * sw_log_ci[::-1] * np.sqrt(2)) - 1,
                       np.exp(+1.96 * sw_log_ci * np.sqrt(2)) - 1] * 100.

        return loa, loa_ci

    def fit(self, x1: NDArray, x2: NDArray) -> "BlandAltman":
        """ Fit the Bland-Altman model to data

        Note: This class assumes that the data are from two timepoints only.

        Parameters:
            x1 (NDArray): 1D array of baseline measurements.
            x2 (NDArray): 1D array of repeat baseline measurements.

        Returns:
            BlandAltman: A reference to self.
        """
        x1, x2 = BlandAltman._validate_paired_1d(x1, x2)
        self._x1 = x1
        self._x2 = x2
        return self

    def metrics(self) -> Dict[str, Any]:
        """ Calculate summary of repeatability metrics.

        Returns:
            Dict[str, Any]: The metrics dictionary.
        """
        metrics = {}
        sw, sw_ci = BlandAltman.within_subject_standard_deviation(self.x1, self.x2, ci=self.ci)
        metrics["sw"] = sw
        metrics["sw_ci"] = sw_ci

        icc, icc_ci = BlandAltman.intraclass_correlation_coefficient(self.x1, self.x2, ci=self.ci)
        metrics["icc"] = icc
        metrics["icc_ci"] = icc_ci

        r, r_ci = BlandAltman.coefficient_of_repeatability(self.x1, self.x2, ci=self.ci)
        metrics["r"] = r
        metrics["r_ci"] = r_ci

        metrics["cov"] = BlandAltman.coefficient_of_variation(self.x1, self.x2)

        if np.all(self.x1 > 0) and np.all(self.x2 > 0):
            loa, loa_ci = BlandAltman.ratio_limits_of_agreement(self.x1, self.x2, ci=self.ci)
            metrics["loa"] = loa
            metrics["loa_ci"] = loa_ci

            metrics["ratio_cov"] = BlandAltman.ratio_coefficient_of_variation(self.x1, self.x2)

        return metrics
