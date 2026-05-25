"""Math test for ``sample_outcome.logistic_outcome``.

``logistic_outcome`` is inverse-CDF sampling of a Bernoulli: given uniform
``u_y`` and ``p = sigmoid(ate*x + const)``, return ``1`` iff ``u_y >= 1 - p``.
So ``P(Y=1) = P(u_y >= 1-p) = p``. We check both the exact threshold and the
empirical rate. (The flow-coupled samplers — causal_cdf / location_translation —
need trained flow objects and are integration-level, out of scope here.)
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from frugal_flows.sample_outcome import logistic_outcome


def test_logistic_outcome_exact_threshold():
    """p=0.5 (ate*x+const=0): u_y>=0.5 -> 1, else 0."""
    u_y = jnp.array([0.1, 0.49, 0.5, 0.9])
    x = jnp.zeros(4)
    out = logistic_outcome(u_y=u_y, ate=1.0, causal_condition=x, const=0.0)
    assert jnp.array_equal(out, jnp.array([0, 0, 1, 1]))


def test_logistic_outcome_empirical_rate_matches_sigmoid():
    """Empirical P(Y=1) ≈ sigmoid(ate*x + const)."""
    n = 40_000
    ate, const, x_val = 2.0, -0.3, 1.0
    u_y = jr.uniform(jr.PRNGKey(0), (n,))
    x = jnp.full((n,), x_val)
    out = logistic_outcome(u_y=u_y, ate=ate, causal_condition=x, const=const)

    p = 1.0 / (1.0 + jnp.exp(-(ate * x_val + const)))  # sigmoid(1.7) ≈ 0.8455
    assert out.mean() == pytest.approx(float(p), abs=0.01)
