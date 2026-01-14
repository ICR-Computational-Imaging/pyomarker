"""Tools for classical repeatability calculations"""

from typing import Tuple, Dict, Any

import numpy as np
from numpy.typing import NDArray
import scipy.stats


class BlandAltman:
    def __init__(self, ddof = 0, confidence_interval_percentile = 0.9):
        self._ddof = ddof
        self._confidence_interval_percentile = confidence_interval_percentile
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
    def ddof(self):
        return self._ddof
    @ddof.setter
    def ddof(self, value):
        self._ddof = value

    @property
    def confidence_interval_percentile(self):
        return self._confidence_interval_percentile
    @confidence_interval_percentile.setter
    def confidence_interval_percentile(self, value):
        self._confidence_interval_percentile = value

    @staticmethod
    def _inv_chi_square(percentile: float, n: int) -> float:
        """ The inverse of the chi-square distribution

        :param percentile: The percentile of the chi-square distribution. 0.0 <= percentile <= 1.0
        :param n: The number of observations.
        :return: The inverse of the chi-square distribution.
        """
        return scipy.stats.chi2.ppf(percentile, n)

    @staticmethod
    def _difference(x1: NDArray, x2: NDArray) -> NDArray:
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        if x1.shape != x2.shape:
            raise ValueError("x1 and x2 must have the same shape.")
        return x2 - x1

    @staticmethod
    def _within_subject_mean(x: NDArray) -> NDArray:
        """ Subjects are assumed to be along axis 0. """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must have 2 dimensions.")
        return np.mean(x, axis=1)

    @staticmethod
    def _between_subject_mean_squares(x1: NDArray, x2: NDArray, ddof: int = 0) -> float:
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        n = x1.size
        within_subject_mean = BlandAltman._within_subject_mean(np.c_[x1, x2])
        population_mean = np.mean(within_subject_mean)
        bsms = (2.0 / (n - ddof)) * np.sum((within_subject_mean - population_mean) ** 2)
        return bsms

    @staticmethod
    def _within_subject_mean_squares(x1: NDArray, x2: NDArray, ddof: int = 0) -> float:
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        n = x1.size
        within_subject_mean = BlandAltman._within_subject_mean(np.c_[x1, x2])
        wsms = np.sum((x1 - within_subject_mean) ** 2 + (x2 - within_subject_mean) ** 2) / (n - ddof)
        return wsms

    @staticmethod
    def _within_subject_standard_deviation(x1: NDArray, x2: NDArray, ddof: int = 0, percentile: float = 0.9) \
            -> Tuple[float, NDArray]:
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        difference = BlandAltman._difference(x1, x2)
        n = x1.shape[0]

        sw = np.sqrt(np.sum(difference ** 2) / (2.0 * (n - ddof)))

        ichi_low = BlandAltman._inv_chi_square((1 - percentile) / 2, n)
        ichi_high = BlandAltman._inv_chi_square(1 - percentile / 2, n)
        sw_ci = np.array([sw * np.sqrt(n / ichi_low), sw * np.sqrt(n / ichi_high)])

        return sw, sw_ci

    @staticmethod
    def _within_subject_coefficient_of_variation(x1: NDArray, x2: NDArray, ddof: int = 0) -> float:
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        sw, _ = BlandAltman._within_subject_standard_deviation(x1, x2, ddof)
        within_subject_mean = BlandAltman._within_subject_mean(np.c_[x1, x2])
        population_mean = np.mean(within_subject_mean)
        cov = sw / population_mean * 100.0
        return cov

    @staticmethod
    def _between_subject_standard_deviation(x1: NDArray, x2: NDArray, ddof: int = 0) -> float:
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        bsms = BlandAltman._between_subject_mean_squares(x1, x2, ddof)
        wsms = BlandAltman._within_subject_mean_squares(x1, x2, ddof)
        sb = np.sqrt((bsms - wsms) / 2)
        return sb

    @staticmethod
    def _between_subject_coefficient_of_variation(x1: NDArray, x2: NDArray, ddof: int = 0) -> float:
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        sb = BlandAltman._between_subject_standard_deviation(x1, x2, ddof)
        within_subject_mean = BlandAltman._within_subject_mean(np.c_[x1, x2])
        population_mean = np.mean(within_subject_mean)
        cov = sb / population_mean * 100.0
        return cov

    @staticmethod
    def _coefficient_of_repeatability(x1: NDArray, x2: NDArray, ddof: int = 0, percentile: float = 0.9)\
            -> Tuple[float, NDArray]:
        """ Compute the coefficient of repeatability of input data. """
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        sw, sw_ci = BlandAltman._within_subject_standard_deviation(x1, x2, ddof, percentile)
        r = 1.96 * np.sqrt(2) * sw
        r_ci = 1.96 * np.sqrt(2) * sw_ci
        return r, r_ci

    @staticmethod
    def _intraclass_correlation_coefficient(x1: NDArray, x2: NDArray, ddof: int = 0, percentile: float = 0.9)\
            -> Tuple[float, NDArray]:
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        sb = BlandAltman._between_subject_standard_deviation(x1, x2, ddof)
        sw, _ = BlandAltman._within_subject_standard_deviation(x1, x2, ddof)
        icc = sb ** 2 / (sb ** 2 + sw ** 2)

        wsms = BlandAltman._within_subject_mean_squares(x1, x2, ddof)
        bsms = BlandAltman._between_subject_mean_squares(x1, x2, ddof)

        f0 = bsms / wsms
        n = x1.size
        f_upper = f0 * scipy.stats.f.ppf((1 - percentile / 2), n, n - 1)
        f_lower = f0 / scipy.stats.f.ppf((1 - percentile / 2), n - 1, n)
        icc_ci = np.array([(f_lower - 1) / (f_lower + 1), (f_upper - 1) / (f_upper + 1)])

        return icc, icc_ci

    @staticmethod
    def _limits_of_agreement(x1: NDArray, x2: NDArray, ddof: int = 0, percentile: float = 0.9)\
            -> Tuple[NDArray, NDArray]:
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        if np.any(x1 <= 0):
            raise ValueError("x1 must be greater than 0")
        if np.any(x2 <= 0):
            raise ValueError("x2 must be greater than 0")

        x1_log = np.log1p(x1)
        x2_log = np.log1p(x2)
        sw_log, sw_log_ci = BlandAltman._within_subject_standard_deviation(x1_log, x2_log, ddof, percentile)

        loa = np.r_[np.exp(-1.96 * sw_log * np.sqrt(2)) - 1,
                    np.exp(+1.96 * sw_log * np.sqrt(2)) - 1] * 100.
        loa_ci = np.r_[np.exp(-1.96 * sw_log_ci[::-1] * np.sqrt(2)) - 1,
                       np.exp(+1.96 * sw_log_ci * np.sqrt(2)) - 1] * 100.

        return loa, loa_ci

    def fit(self, x1: NDArray, x2: NDArray) -> "BlandAltman":
        """ Fit the Bland-Altman model to data

        Note: This class assumes that the data are from two timepoints only.

        :param x1: Baseline data set 1
        :param x2: Baseline data set 2
        :return: The fit model.
        """
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        if x1.shape != x2.shape:
            raise ValueError("x1 and x2 must have the same shape.")

        if np.any(~np.isfinite(x1)) or np.any(~np.isfinite(x2)):
            raise ValueError("x1 and x2 must not contain NaN or inf.")

        self._x1 = x1
        self._x2 = x2
        return self

    def metrics(self, log_transform: bool = False) -> Dict[str, Any]:
        """ Calculate summary of repeatability metrics.

        :param log_transform: If true, log transform is applied.
        :return: The metrics dictionary.
        """
        x1 = np.asarray(self.x1)
        x2 = np.asarray(self.x2)
        if log_transform:
            if np.any(x1 <= 0):
                raise ValueError("x1 must be greater than 0")
            if np.any(x2 <= 0):
                raise ValueError("x2 must be greater than 0")
            x1 = np.log1p(x1)
            x2 = np.log1p(x2)

        metrics = {}
        sw, sw_ci = BlandAltman._within_subject_standard_deviation(x1, x2,
                                                                   ddof=self.ddof,
                                                                   percentile=self.confidence_interval_percentile)
        metrics["sw"] = sw
        metrics["sw_ci"] = sw_ci

        icc, icc_ci = BlandAltman._intraclass_correlation_coefficient(x1, x2,
                                                                      ddof=self.ddof,
                                                                      percentile=self.confidence_interval_percentile)
        metrics["icc"] = icc
        metrics["icc_ci"] = icc_ci

        r, r_ci = BlandAltman._coefficient_of_repeatability(x1, x2,
                                                            ddof=self.ddof,
                                                            percentile=self.confidence_interval_percentile)
        metrics["r"] = r
        metrics["r_ci"] = r_ci

        metrics["cov"] = BlandAltman._within_subject_coefficient_of_variation(x1, x2, ddof=self.ddof)

        if not log_transform:
            if np.all(x1 > 0) and np.all(x2 > 0):
                loa, loa_ci = BlandAltman._limits_of_agreement(x1, x2,
                                                               ddof=self.ddof,
                                                               percentile=self.confidence_interval_percentile)
                metrics["loa"] = loa
                metrics["loa_ci"] = loa_ci

        return metrics


def repeatability_stats(x1, x2, calc_log=False):
    # Takes in two vectors of data and provides repeatability statistics

    # Make sure they are numpy vectors and have the same shape
    x1 = np.array(x1)
    x2 = np.array(x2)
    if np.size(x1.shape) > 1 or np.size(x2.shape) > 1:
        raise Exception("x1 and x2 must be vectors")
    if not np.array_equal(x1.shape, x2.shape):
        raise Exception("x1 and x2 must have same shape!")

    # The size of the data
    N = float(x1.size)

    # First calcaulte the repeatability using the un-logged data
    # The difference between data points
    d = x2 - x1

    # The overall mean and within subject means (M and m respectively)
    m = (x1 + x2) / 2
    M = np.mean(m)

    ddof = 0 # For now the degrees of freedom is 0 (assume x1 and x2 are not biased)

    # Inverse chi-square values for CI calcaultion
    ichi_025 = scipy.stats.chi2.ppf(0.025, N)
    ichi_975 = scipy.stats.chi2.ppf(0.975, N)

    sw = np.sqrt(np.sum(d**2)/(2.0 * (N-ddof)))
    sw_CI = np.array([sw * np.sqrt(N / ichi_975), sw * np.sqrt(N / ichi_025)])
    sw_CoV = sw/M * 100.0
    # Should we have a sw_COV_CI here?  Need to determine what this would be given error in M!

    r = 1.96 * np.sqrt(2) * sw
    r_CI = 1.96 * np.sqrt(2) * sw_CI
    r_CoV = r/M * 100.0

    # Between-subject mean squares and within-subject mean squares respectively
    BMS = (2.0 / (N-ddof)) * np.sum((m - M) ** 2)
    WMS = np.sum((x1 - m) ** 2 + (x2 - m) ** 2) / (N-ddof)

    sb = np.sqrt((BMS-WMS)/2)
    sb_CoV = sb/M * 100.0

    ICC = sb**2 / (sb**2 + sw**2)
    F0 = BMS / WMS
    FU = F0 * scipy.stats.f.ppf(0.975, N, N - 1)
    FL = F0 / scipy.stats.f.ppf(0.975, N - 1, N)
    ICC_CI = [(FL - 1) / (FL + 1), (FU - 1) / (FU + 1)]

    stats = {}
    stats["original"] = {'sw':sw, 'sw_CI':sw_CI, 'sw_CoV':sw_CoV,
                         'r':r, 'r_CI':r_CI, 'r_CoV':r_CoV,
                         'sb':sb, 'sb_CoV':sb_CoV,
                         'BMS':BMS, 'WMS':WMS, 'ICC':ICC, 'ICC_CI':ICC_CI}

    if calc_log:
        # Now perform for the logarithm of the data
        x1 = np.log(x1)
        x2 = np.log(x2)

        d = x2 - x1

        # The overall mean and within subject means (M and m respectively)
        m = (x1 + x2) / 2
        M = np.mean(m)

        sw = np.sqrt(np.sum(d ** 2) / (2.0 * (N - ddof)))
        sw_CI = np.array([sw * np.sqrt(N / ichi_975), sw * np.sqrt(N / ichi_025)])
        sw_CoV = np.sqrt(np.exp(sw ** 2) - 1) * 100.0
        sw_CoV_CI = np.sqrt(np.exp(sw_CI ** 2) - 1) * 100.0

        r = 1.96 * np.sqrt(2) * sw
        r_CI = 1.96 * np.sqrt(2) * sw_CI

        LoA = np.r_[np.exp(-1.96 * sw * np.sqrt(2)) - 1, np.exp(+1.96 * sw * np.sqrt(2)) - 1] * 100.
        LoA_CI = np.r_[np.exp(-1.96 * sw_CI[::-1] * np.sqrt(2)) - 1, np.exp(+1.96 * sw_CI * np.sqrt(2)) - 1] * 100.
        # Reverse the sign to ensure LoA CI in increasgin order.

        # Between-subject mean squares and within-subject mean squares respectively
        BMS = (2.0 / (N - ddof)) * np.sum((m - M) ** 2)
        WMS = np.sum((x1 - m) ** 2 + (x2 - m) ** 2) / (N - ddof)

        sb = np.sqrt((BMS - WMS) / 2)
        sb_CoV = np.sqrt(np.exp(sb ** 2) - 1) * 100.0

        ICC = sb ** 2 / (sb ** 2 + sw ** 2)
        F0 = BMS / WMS
        FU = F0 * scipy.stats.f.ppf(0.975, N, N - 1)
        FL = F0 / scipy.stats.f.ppf(0.975, N - 1, N)
        ICC_CI = [(FL - 1) / (FL + 1), (FU - 1) / (FU + 1)]

        stats['log'] = {'sw': sw, 'sw_CI': sw_CI, 'sw_CoV': sw_CoV, 'sw_CoV_CI':sw_CoV_CI,
                        'r': r, 'r_CI': r_CI,
                        'LoA':LoA, 'LoA_CI':LoA_CI,
                        'sb': sb, 'sb_CoV': sb_CoV,
                        'BMS': BMS, 'WMS': WMS, 'ICC': ICC, 'ICC_CI': ICC_CI}

    return stats

