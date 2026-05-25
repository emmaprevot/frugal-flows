"""Rational quadratic spline bijections (https://arxiv.org/abs/1906.04032)."""

from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
from flowjax import wrappers
from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.rational_quadratic_spline import _real_to_increasing_on_interval
from jaxtyping import Array


class RationalQuadraticSplineAdditiveCond(AbstractBijection):
    """Scalar rational-quadratic spline with an additive conditional shift.

    The spline itself is the standard RQS of Durkan et al. 2019
    (https://arxiv.org/abs/1906.04032), eqs. 4–5, with linear (identity) tails
    outside ``[-interval, interval]``. On top of the spline this variant adds a
    condition-driven location shift:

        forward:  y = RQS(x) + ate * condition[0]
        inverse:  x = RQS⁻¹(y - ate * condition[0])

    The "Additive" in the class name is load-bearing: the condition term is a
    constant offset in ``y``, so ``dy/dx = RQS'(x)`` — the Jacobian depends
    only on the spline, not on the condition. The log-det implementations rely
    on this invariant.

    When ``condition is None`` (the default call from flowjax, since
    ``cond_shape`` is a ``ClassVar = None``) the shift term is dropped and
    this is plain RQS. Conditioning is therefore "soft": it only applies when
    a caller explicitly forwards a ``condition``.

    Note:
        Not used anywhere inside ``frugal_flows`` — the flows use flowjax's own
        ``RationalQuadraticSpline`` as the MAF transformer. Kept for the
        notebooks / completeness.

    Args:
        knots: Number of knots.
        interval: interval to transform, [-interval, interval].
        min_derivative: Minimum dervivative. Defaults to 1e-3.
        softmax_adjust: Controls minimum bin width and height by rescaling softmax
            output, e.g. 0=no adjustment, 1=average softmax output with evenly spaced
            widths, >1 promotes more evenly spaced widths. See
            ``real_to_increasing_on_interval``. Defaults to 1e-2.
        ate: Average treatment effect; per-unit shift applied as
            ``ate * condition[0]`` (only when a ``condition`` is supplied).
            Defaults to 0.0.
    """

    knots: int
    interval: float | int
    softmax_adjust: float | int
    min_derivative: float
    x_pos: Array | wrappers.AbstractUnwrappable[Array]
    y_pos: Array | wrappers.AbstractUnwrappable[Array]
    derivatives: Array | wrappers.AbstractUnwrappable[Array]
    shape: ClassVar[tuple] = ()
    cond_shape: ClassVar[None] = None
    ate: float | int = 0.0

    def __init__(
        self,
        *,
        knots: int,
        interval: float | int,
        min_derivative: float = 1e-3,
        softmax_adjust: float | int = 1e-2,
        ate: float | int = 0.0,
    ):
        self.knots = knots
        self.interval = interval
        self.softmax_adjust = softmax_adjust
        self.min_derivative = min_derivative
        self.ate = ate

        # Inexact arrays
        pos_parameterization = partial(
            _real_to_increasing_on_interval,
            interval=interval,
            softmax_adjust=softmax_adjust,
        )

        self.x_pos = wrappers.Lambda(pos_parameterization, jnp.zeros(knots))
        self.y_pos = wrappers.Lambda(pos_parameterization, jnp.zeros(knots))
        self.derivatives = wrappers.Lambda(
            lambda arr: jax.nn.softplus(arr) + self.min_derivative,
            jnp.full(knots + 2, jnp.log(jnp.exp(1 - min_derivative) - 1)),
        )

    def transform(self, x, condition=None):
        # Following notation from the paper
        x_pos, y_pos, derivatives = self.x_pos, self.y_pos, self.derivatives
        in_bounds = jnp.logical_and(x >= -self.interval, x <= self.interval)
        x_robust = jnp.where(in_bounds, x, 0)  # To avoid nans
        k = jnp.searchsorted(x_pos, x_robust) - 1  # k is bin number
        xi = (x_robust - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
        sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
        dk, dk1, yk, yk1 = derivatives[k], derivatives[k + 1], y_pos[k], y_pos[k + 1]
        num = (yk1 - yk) * (sk * xi**2 + dk * xi * (1 - xi))
        den = sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)
        y = yk + num / den  # eq. 4

        # avoid numerical precision issues transforming from in -> out of bounds
        y = jnp.clip(y, -self.interval, self.interval)
        if condition is not None:
            return jnp.where(in_bounds, y, x) + self.ate * condition[0]
        else:
            return jnp.where(in_bounds, y, x)

    def transform_and_log_det(self, x, condition=None):
        y = self.transform(x, condition=condition)
        derivative = self.derivative(x)
        return y, jnp.log(derivative).sum()

    def inverse(self, y, condition=None):
        # Following notation from the paper
        if condition is not None:
            y = y - self.ate * condition[0]
        x_pos, y_pos, derivatives = self.x_pos, self.y_pos, self.derivatives
        in_bounds = jnp.logical_and(y >= -self.interval, y <= self.interval)
        y_robust = jnp.where(in_bounds, y, 0)  # To avoid nans
        k = jnp.searchsorted(y_pos, y_robust) - 1
        xk, xk1, yk, yk1 = x_pos[k], x_pos[k + 1], y_pos[k], y_pos[k + 1]
        sk = (yk1 - yk) / (xk1 - xk)
        y_delta_s_term = (y_robust - yk) * (
            derivatives[k + 1] + derivatives[k] - 2 * sk
        )
        a = (yk1 - yk) * (sk - derivatives[k]) + y_delta_s_term
        b = (yk1 - yk) * derivatives[k] - y_delta_s_term
        c = -sk * (y_robust - yk)
        sqrt_term = jnp.sqrt(b**2 - 4 * a * c)
        xi = (2 * c) / (-b - sqrt_term)
        x = xi * (xk1 - xk) + xk

        # avoid numerical precision issues transforming from in -> out of bounds
        x = jnp.clip(x, -self.interval, self.interval)
        return jnp.where(in_bounds, x, y)

    def inverse_and_log_det(self, y, condition=None):
        x = self.inverse(y, condition=condition)
        derivative = self.derivative(x)
        return x, -jnp.log(derivative).sum()

    def derivative(self, x) -> Array:
        """RQS'(x) at the actual spline input.

        The forward is ``y = RQS(x) + ate * condition[0]``: the condition term
        is additive in ``y``, so ``dy/dx = RQS'(x)`` and does not depend on the
        condition. Both log-det methods therefore evaluate ``derivative`` at
        the RQS-input-domain point and pass no condition.
        """
        # Following notation from the paper (eq. 5)
        x_pos, y_pos, derivatives = self.x_pos, self.y_pos, self.derivatives
        in_bounds = jnp.logical_and(x >= -self.interval, x <= self.interval)
        x_robust = jnp.where(in_bounds, x, 0)  # To avoid nans
        k = jnp.searchsorted(x_pos, x_robust) - 1
        xi = (x_robust - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
        sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
        dk, dk1 = derivatives[k], derivatives[k + 1]
        num = sk**2 * (dk1 * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
        den = (sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)) ** 2
        derivative = num / den
        return jnp.where(in_bounds, derivative, 1.0)
