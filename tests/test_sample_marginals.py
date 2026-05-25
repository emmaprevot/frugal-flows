"""Tests for ``sample_marginals`` (inverse PIT: quantiles -> marginal samples).

Continuous path is tightly coupled to flowjax flow internals (hardcoded
bijection indices) and needs a trained flow; that is a Phase-2 fragility, not
covered here. These tests cover the discrete path: correctness of the
empirical-CDF inverse, and a regression marker for bug #7 (the vmapped
fast-path in ``from_quantiles_to_marginal_discr`` is broken and dead).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from frugal_flows.sample_marginals import (
    from_quantiles_to_marginal_discr,
    univariate_from_quantiles_to_marginal_discr,
)


def test_univariate_discrete_inverse_cdf_is_correct():
    """u > cdf_levels, summed, indexes the category. CDF [0.5,0.8,1.0],
    labels {10,20,30}: u=0.3->10, 0.6->20, 0.9->30, 0.99->30."""
    cdf_levels = jnp.array([0.5, 0.8, 1.0])
    key_mapping = jnp.array([10, 20, 30])
    u = jnp.array([0.3, 0.6, 0.9, 0.99])
    out = univariate_from_quantiles_to_marginal_discr(cdf_levels, key_mapping, u)
    assert jnp.array_equal(out, jnp.array([10, 20, 30, 30]))


def test_from_quantiles_to_marginal_discr_correct_via_loop_fallback():
    """End-to-end discrete inverse PIT. Output is correct because the (broken)
    vmapped path is caught and the Python-loop fallback runs."""
    mappings = {0: {10: 0, 20: 1, 30: 2}}
    empirical_cdfs = jnp.array([[0.5, 0.8, 1.0]])
    u_z = jnp.array([[0.3], [0.6], [0.9]])
    out = from_quantiles_to_marginal_discr(
        key=jr.PRNGKey(0),
        mappings=mappings,
        nvars=1,
        empirical_cdfs=empirical_cdfs,
        n_samples=3,
        u_z=u_z,
    )
    assert out.shape == (3, 1)
    assert jnp.array_equal(out.ravel(), jnp.array([10, 20, 30]))


def test_bug7_vmapped_fastpath_is_broken_and_dead():
    """REGRESSION MARKER (bug #7): the vmapped fast-path as written in
    from_quantiles_to_marginal_discr passes 5 args to a 3-arg function, so it
    always raises and is silently swallowed by a bare `except Exception`.
    Pin the signature mismatch so a future fix is deliberate."""
    keys = jr.split(jr.PRNGKey(0), 1)
    empirical_cdfs = jnp.array([[0.5, 0.8, 1.0]])
    z_map = jnp.array([[10, 20, 30]])
    unis_T = jnp.array([[0.3, 0.6, 0.9]])
    vmapped = jax.vmap(
        univariate_from_quantiles_to_marginal_discr, in_axes=(0, 0, None, 0, 0)
    )
    with pytest.raises(TypeError):
        vmapped(keys, empirical_cdfs, 3, z_map, unis_T)
