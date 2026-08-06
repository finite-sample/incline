"""Tests for the smoothers that are probability models.

A Gaussian process and a state-space model carry their own posterior variance,
so they bypass both the operator probe and the bootstrap. That makes them the
two places where a variance bug cannot be caught by the shared machinery, which
is why the numbers here are checked against closed forms rather than shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from incline.axis import TimeAxis
from incline.process import (
    MATERN_ORDERS,
    GaussianProcess,
    StateSpace,
    _cross_derivative,
    _KernelParts,
    _prior_derivative_variance,
)


KERNELS = ["rbf", "matern32", "matern52"]


def ramp(n: int = 80, slope: float = 2.0, step: float = 1.0):
    """A noiseless straight line on an axis with the given spacing."""
    axis = TimeAxis._build(np.arange(n) * step, "index")
    return axis, slope * axis.x


@pytest.mark.parametrize("kernel", KERNELS)
def test_analytic_kernel_derivative_matches_numerical(kernel):
    """The differentiated covariance must be the covariance's derivative."""
    parts = _KernelParts(amplitude=1.7, length_scale=2.3, noise=0.0, family=kernel)
    r = np.array([0.4, 1.1, -2.0])
    step = 1e-5
    analytic = _cross_derivative(parts, r, 1)
    numerical = np.array(
        [
            (
                _cross_derivative(parts, np.array([v + step]), 0)[0]
                - _cross_derivative(parts, np.array([v - step]), 0)[0]
            )
            / (2 * step)
            for v in r
        ]
    )
    np.testing.assert_allclose(analytic, numerical, rtol=1e-5)


@pytest.mark.parametrize("kernel", KERNELS)
def test_prior_derivative_variance_matches_numerical(kernel):
    """The derivative's prior variance is -d2k/dr2 at zero separation."""
    parts = _KernelParts(amplitude=1.7, length_scale=2.3, noise=0.0, family=kernel)
    step = 1e-3

    def k(v):
        return float(_cross_derivative(parts, np.array([float(v)]), 0)[0])

    numerical = -(k(step) - 2 * k(0.0) + k(-step)) / step**2
    assert _prior_derivative_variance(parts, 1) == pytest.approx(numerical, rel=1e-3)


def test_rbf_second_derivative_prior_variance():
    """RBF is infinitely differentiable, so order 2 has 3 * amplitude / l**4."""
    parts = _KernelParts(amplitude=1.7, length_scale=2.3, noise=0.0, family="rbf")
    assert _prior_derivative_variance(parts, 2) == pytest.approx(3 * 1.7 / 2.3**4)


@pytest.mark.parametrize("kernel", KERNELS)
def test_gp_recovers_a_known_slope(kernel):
    """The posterior mean derivative must track the truth."""
    axis = TimeAxis.positional(120)
    x = axis.x / 12.0
    y = 0.5 * x + np.sin(x)
    truth = (0.5 + np.cos(x)) / 12.0

    estimate = GaussianProcess(kernel=kernel, n_restarts=0).fit(axis, y, order=1)
    assert np.corrcoef(estimate.derivative, truth)[0, 1] > 0.9


def test_gp_standard_error_is_the_right_order_of_magnitude():
    """Regression: the reported SE was once 3,614x too large.

    The old implementation differenced the posterior mean at a hundredth of the
    sample spacing and added the endpoint variances as if independent, so the
    error scaled with 1/dx and the interval swallowed every plausible value.
    """
    x = np.linspace(0, 10, 120)
    axis = TimeAxis._build(x, "index")
    y = 0.5 * x + np.sin(x) + np.random.default_rng(0).normal(0, 0.3, 120)

    estimate = GaussianProcess(kernel="rbf").fit(axis, y, order=1, se=True)
    assert 0.01 < float(np.median(estimate.se)) < 1.0
    assert estimate.provenance.se_method == "native"
    # The whole point of a usable interval is that it can exclude zero.
    assert estimate.significant.mean() > 0.5


def test_gp_uncertainty_does_not_depend_on_an_arbitrary_step():
    """An exact derivative posterior has no step size to be sensitive to."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.default_rng(1).normal(0, 0.2, 100)
    coarse = GaussianProcess(length_scale=2.0, n_restarts=0).fit(
        TimeAxis._build(x, "index"), y, order=1, se=True
    )
    # Re-expressing the same series on a finer grid rescales the slope but must
    # not change how many points are distinguishable from zero.
    fine = GaussianProcess(length_scale=4.0, n_restarts=0).fit(
        TimeAxis._build(2 * x, "index"), y, order=1, se=True
    )
    np.testing.assert_allclose(
        coarse.derivative, 2 * fine.derivative, rtol=1e-3, atol=1e-6
    )
    np.testing.assert_allclose(coarse.se, 2 * fine.se, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("kernel", KERNELS)
def test_matern_refuses_derivatives_it_does_not_have(kernel):
    """A nu-smooth process has floor(nu) derivatives; asking past that must raise."""
    axis = TimeAxis.positional(60)
    y = np.sin(axis.x / 8)
    too_far = MATERN_ORDERS[kernel] + 1
    if too_far > max(GaussianProcess.supported_orders):
        pytest.skip(f"{kernel} supports every order the smoother offers")
    with pytest.raises(ValueError, match="derivative orders up to"):
        GaussianProcess(kernel=kernel, n_restarts=0).fit(
            axis, y, order=too_far, se=True
        )


def test_unknown_kernel_is_refused():
    """A typo must not silently fall back to a default kernel."""
    axis = TimeAxis.positional(40)
    with pytest.raises(ValueError, match="Unknown kernel"):
        GaussianProcess(kernel="rbg", n_restarts=0).fit(axis, np.sin(axis.x))


def test_requested_length_scale_is_honoured_when_optimisation_is_off():
    """Regression: with_scale was a no-op because sklearn always re-optimized.

    ``n_restarts_optimizer`` controls how many restarts an optimization gets,
    not whether one happens, so a supplied length scale was only ever a
    starting point and every scale in a sweep produced the same fit.
    """
    axis = TimeAxis.positional(80)
    y = np.sin(axis.x / 10) + np.random.default_rng(0).normal(0, 0.3, 80)
    for wanted in (1.5, 8.0, 32.0):
        fitted = GaussianProcess(length_scale=wanted, optimize=False)._fitted(axis, y)
        assert fitted.kernel_.k1.k2.length_scale == pytest.approx(wanted)


def test_with_scale_actually_changes_the_gp_fit():
    """A narrower scale must give a wigglier derivative, or sweeps are useless."""
    axis = TimeAxis.positional(80)
    y = np.sin(axis.x / 10) + np.random.default_rng(0).normal(0, 0.3, 80)
    smoother = GaussianProcess()
    narrow = smoother.with_scale(0.05, axis).fit(axis, y, order=1).derivative
    wide = smoother.with_scale(0.5, axis).fit(axis, y, order=1).derivative
    assert np.std(narrow) > np.std(wide)


def test_supplied_amplitude_and_noise_are_honored_in_data_units():
    """Regression: a stated prior must be the prior that gets used.

    With standardization on, the response is rescaled by its own spread, so a
    supplied amplitude and noise level mean something other than what was
    asked for. Turning it off puts them in the data's own units.
    """
    axis = TimeAxis.positional(80)
    y = 5.0 * np.sin(axis.x / 10) + np.random.default_rng(0).normal(0, 1.5, 80)
    fitted = GaussianProcess(
        amplitude=25.0,
        length_scale=9.0,
        noise_level=2.25,
        optimize=False,
        standardize=False,
    )._fitted(axis, y)

    assert fitted.kernel_.k1.k1.constant_value == pytest.approx(25.0)
    assert fitted.kernel_.k2.noise_level == pytest.approx(2.25)


def test_standardizing_puts_the_prior_in_units_of_the_spread():
    """The default rescales, so the same numbers mean something different."""
    axis = TimeAxis.positional(80)
    y = 5.0 * np.sin(axis.x / 10) + np.random.default_rng(0).normal(0, 1.5, 80)
    standardized = GaussianProcess(
        amplitude=25.0, length_scale=9.0, noise_level=2.25, optimize=False
    ).fit(axis, y, order=1, se=True)
    exact = GaussianProcess(
        amplitude=25.0,
        length_scale=9.0,
        noise_level=2.25,
        optimize=False,
        standardize=False,
    ).fit(axis, y, order=1, se=True)
    assert not np.allclose(standardized.se, exact.se)


def test_level_is_recovered_whether_or_not_the_response_is_scaled():
    """Centering happens either way, so a series far from zero is not shrunk."""
    axis = TimeAxis.positional(80)
    y = 100.0 + np.sin(axis.x / 8)
    for standardize in (True, False):
        values = GaussianProcess(standardize=standardize).fit(axis, y, order=1).values
        assert np.mean(values) == pytest.approx(100.0, abs=1.0)


def test_optimisation_is_still_on_by_default():
    """Leaving the kernel to be learned is the sensible default."""
    axis = TimeAxis.positional(80)
    y = np.sin(axis.x / 10) + np.random.default_rng(0).normal(0, 0.3, 80)
    fitted = GaussianProcess(length_scale=1.5)._fitted(axis, y)
    assert fitted.kernel_.k1.k2.length_scale != pytest.approx(1.5)


def test_gp_fit_is_reused_between_estimate_and_posterior():
    """fit() asks for the mean and the variance; it should not refit for each."""
    axis = TimeAxis.positional(60)
    y = np.sin(axis.x / 8)
    smoother = GaussianProcess(n_restarts=0)
    assert smoother._fitted(axis, y) is smoother._fitted(axis, y)


@pytest.mark.parametrize("step", [1.0, 0.5, 0.25, 2.0])
def test_state_space_slope_is_spacing_invariant(step):
    """Regression: on y = 2t the slope is 2.0 per unit t at any sampling step.

    The state advances per observation, so it has to be divided by the step
    exactly once. Dividing twice, or not at all, shows up here.
    """
    axis, y = ramp(n=80, slope=2.0, step=step)
    estimate = StateSpace().fit(axis, y, order=1)
    interior = estimate.derivative[20:-5]
    assert np.median(interior) == pytest.approx(2.0, rel=0.1)


def test_state_space_derivative_is_the_slope_not_its_gradient():
    """Regression: the structural model once reported d(slope)/dt instead.

    On a straight line the slope is a nonzero constant and its gradient is
    zero, so the two are trivially distinguishable.
    """
    axis, y = ramp(n=80, slope=2.0)
    estimate = StateSpace().fit(axis, y, order=1)
    assert abs(float(np.median(estimate.derivative[20:-5])) - 2.0) < 0.3


def test_state_space_reports_a_native_standard_error():
    """The slope is a state, so its variance is a smoother-covariance entry."""
    axis = TimeAxis.positional(90)
    y = 0.05 * axis.x + np.random.default_rng(2).normal(0, 0.3, 90)
    estimate = StateSpace().fit(axis, y, order=1, se=True)
    assert estimate.provenance.se_method == "native"
    assert estimate.se is not None
    assert np.all(estimate.se[np.isfinite(estimate.se)] >= 0)


def test_state_space_seasonal_component_is_accepted():
    """A seasonal term is a real statsmodels option, unlike damped_trend."""
    axis = TimeAxis.positional(96)
    y = 0.02 * axis.x + 2 * np.sin(2 * np.pi * axis.x / 12)
    estimate = StateSpace(seasonal_periods=12).fit(axis, y, order=1)
    assert np.isfinite(estimate.derivative).sum() > 80
    assert estimate.provenance.params["seasonal_periods"] == 12


def test_state_space_has_no_damped_trend_option():
    """Regression: it used to accept one and forward it to a parameter that
    does not exist, so the setting silently did nothing."""
    assert "damped" not in StateSpace.__dataclass_fields__
    assert "damped_trend" not in StateSpace.__dataclass_fields__
    with pytest.raises(TypeError):
        StateSpace(damped_trend=True)  # type: ignore[call-arg]


def test_state_space_has_no_hyperparameter_inflation_option():
    """Regression: an inflation heuristic that could be 1e8 too large.

    It scaled the interval by ``bse / |param|``, which is undefined when a
    variance is estimated at exactly zero -- routine whenever a component is
    not needed. Median inflation over 40 fits was 1e5, maximum 3e8.
    """
    assert "hyperparameter_uncertainty" not in StateSpace.__dataclass_fields__
    with pytest.raises(TypeError):
        StateSpace(hyperparameter_uncertainty=True)  # type: ignore[call-arg]


def test_state_space_standard_error_is_the_right_order_of_magnitude():
    """The reported error must be commensurate with the actual estimation error."""
    axis = TimeAxis.positional(120)
    rng = np.random.default_rng(4)
    level, slope = np.empty(120), np.empty(120)
    level[0], slope[0] = 0.0, 0.05
    for t in range(1, 120):
        slope[t] = slope[t - 1] + rng.normal(0, 0.01)
        level[t] = level[t - 1] + slope[t - 1] + rng.normal(0, 0.05)

    estimate = StateSpace().fit(axis, level + rng.normal(0, 0.5, 120), order=1, se=True)
    error = abs(float(estimate.derivative[60]) - float(slope[60]))
    assert float(estimate.se[60]) < 100 * max(error, 1e-3)


def test_matern52_delivers_the_second_derivative_it_advertises():
    """Regression: MATERN_ORDERS said 2 but the kernel branch raised.

    A nu=5/2 process is twice mean-square differentiable, so the estimate and
    its posterior variance both exist; the advertised capability just was not
    implemented.
    """
    assert MATERN_ORDERS["matern52"] == 2
    axis = TimeAxis.positional(120)
    x = axis.x / 12.0
    y = np.sin(x)

    estimate = GaussianProcess(kernel="matern52", n_restarts=0).fit(
        axis, y, order=2, se=True
    )
    assert np.all(np.isfinite(estimate.derivative))
    assert estimate.se is not None
    assert np.all(estimate.se >= 0)
    # The second derivative of sin is -sin, up to the axis rescaling.
    assert (
        np.corrcoef(estimate.derivative[20:-20], (-np.sin(x) / 144)[20:-20])[0, 1] > 0.8
    )


def test_matern52_second_derivative_kernel_matches_numerical():
    """The analytic forms must be the derivatives they claim to be."""
    parts = _KernelParts(amplitude=1.7, length_scale=2.3, noise=0.0, family="matern52")
    r = np.array([0.4, 1.1, -2.0])
    step = 1e-4

    def base(v):
        return float(_cross_derivative(parts, np.array([float(v)]), 0)[0])

    numerical = np.array(
        [(base(v + step) - 2 * base(v) + base(v - step)) / step**2 for v in r]
    )
    np.testing.assert_allclose(_cross_derivative(parts, r, 2), numerical, rtol=1e-3)

    # d4k/dr4 at the origin converges slowly because of the |r|^5 term, so
    # compare a Richardson extrapolation rather than a single difference.
    def fourth(h):
        return (
            base(2 * h) - 4 * base(h) + 6 * base(0) - 4 * base(-h) + base(-2 * h)
        ) / h**4

    extrapolated = 2 * fourth(0.0125) - fourth(0.025)
    assert _prior_derivative_variance(parts, 2) == pytest.approx(extrapolated, rel=0.02)
