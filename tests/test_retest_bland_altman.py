import numpy as np
import pytest

from pyomarker.models.test_retest.real.bland_altman import BlandAltman

@pytest.fixture
def simple_data():
    # Perfect repeatability
    x1 = np.array([1.0, 2.0, 3.0, 4.0])
    x2 = x1.copy()
    return x1, x2


@pytest.fixture
def noisy_data():
    rng = np.random.default_rng(123)
    x1 = rng.normal(loc=10.0, scale=2.0, size=30)
    x2 = x1 + rng.normal(scale=0.5, size=30)
    return x1, x2


def test_validate_paired_1d_shape_mismatch():
    with pytest.raises(ValueError):
        BlandAltman._validate_paired_1d([1, 2, 3], [1, 2])


def test_validate_paired_1d_nonfinite():
    with pytest.raises(ValueError):
        BlandAltman._validate_paired_1d([1, 2, np.nan], [1, 2, 3])


@pytest.mark.parametrize("ci", [-0.1, 0.0, 1.0, 1.1, "foo"])
def test_validate_ci_invalid(ci):
    with pytest.raises(ValueError):
        BlandAltman._validate_ci(ci)


def test_validate_ci_valid():
    assert BlandAltman._validate_ci(0.9) == 0.9


def test_within_subject_sd_zero_when_identical(simple_data):
    x1, x2 = simple_data
    sw, sw_ci = BlandAltman.within_subject_standard_deviation(x1, x2)

    assert sw == 0.0
    assert sw_ci[0] == 0.0
    assert sw_ci[1] == 0.0


def test_within_subject_mean(simple_data):
    x1, x2 = simple_data
    wsm = BlandAltman.within_subject_mean(x1, x2)

    np.testing.assert_allclose(wsm, x1)


def test_population_mean(simple_data):
    x1, x2 = simple_data
    mu = BlandAltman.population_mean(x1, x2)

    assert mu == pytest.approx(np.mean(x1))


def test_bsms_wsms_nonnegative(noisy_data):
    x1, x2 = noisy_data

    bsms = BlandAltman.between_subject_mean_squares(x1, x2)
    wsms = BlandAltman.within_subject_mean_squares(x1, x2)

    assert bsms >= 0.0
    assert wsms >= 0.0


def test_icc_bounds(noisy_data):
    x1, x2 = noisy_data

    icc, icc_ci = BlandAltman.intraclass_correlation_coefficient(x1, x2)

    assert 0.0 <= icc <= 1.0
    assert icc_ci[0] <= icc <= icc_ci[1]
    assert 0.0 <= icc_ci[0] <= icc_ci[1] <= 1.0


def test_icc_perfect_repeatability(simple_data):
    x1, x2 = simple_data

    icc, _ = BlandAltman.intraclass_correlation_coefficient(x1, x2)

    assert icc == pytest.approx(1.0)


def test_cov_zero_when_identical(simple_data):
    x1, x2 = simple_data
    cov = BlandAltman.coefficient_of_variation(x1, x2)

    assert cov == 0.0


def test_ratio_cov_requires_positive():
    with pytest.raises(ValueError):
        BlandAltman.ratio_coefficient_of_variation(
            np.array([1.0, -2.0]),
            np.array([1.1, 2.1])
        )


def test_ratio_cov_finite(noisy_data):
    x1, x2 = noisy_data
    x1 = np.abs(x1) + 1.0
    x2 = np.abs(x2) + 1.0

    cov = BlandAltman.ratio_coefficient_of_variation(x1, x2)
    assert np.isfinite(cov)
    assert cov >= 0.0


def test_ratio_loa_structure(noisy_data):
    x1, x2 = noisy_data
    x1 = np.abs(x1) + 1.0
    x2 = np.abs(x2) + 1.0

    loa, loa_ci = BlandAltman.ratio_limits_of_agreement(x1, x2)

    assert loa.shape == (2,)
    assert loa_ci.shape == (4,)
    assert loa[0] < loa[1]


def test_fit_and_metrics(noisy_data):
    x1, x2 = noisy_data

    model = BlandAltman(ci=0.9).fit(x1, x2)
    metrics = model.metrics()

    required_keys = {
        "sw", "sw_ci",
        "icc", "icc_ci",
        "r", "r_ci",
        "cov",
    }

    assert required_keys.issubset(metrics.keys())
    assert np.isfinite(metrics["sw"])
