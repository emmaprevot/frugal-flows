"""Tests for ``benchmarking.FrugalFlowModel`` construction.

Covers the fix for bug #8: the constructor previously used ``Z_disc != None`` /
``Z_cont != None``, which broadcast elementwise over array confounders and
raised ``ValueError: ambiguous truth value`` — making the class unusable with
its primary input (real multi-element array confounders). The idiom is now
``is None`` / ``is not None``, which is identity-check-on-the-Python-object
and never broadcasts.

``benchmarking`` imports ``wandb`` at module top (an unused hard dependency);
skip cleanly if it is not installed rather than error the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

benchmarking = pytest.importorskip(
    "frugal_flows.benchmarking",
    reason="frugal_flows.benchmarking imports wandb (unused hard dep) — skip if absent",
)


def test_construct_with_array_z_disc():
    """Bug #8 fix: array-valued Z_disc must not crash construction."""
    Y = np.zeros((10, 1))
    X = np.zeros((10, 1))
    Z_disc = np.zeros((10, 2))
    model = benchmarking.FrugalFlowModel(Y=Y, X=X, Z_disc=Z_disc)
    assert model.conf_shape == 2
    assert model.Z_cont is None


def test_construct_with_array_z_cont():
    """Bug #8 fix: array-valued Z_cont must not crash construction."""
    Y = np.zeros((10, 1))
    X = np.zeros((10, 1))
    Z_cont = np.zeros((10, 3))
    model = benchmarking.FrugalFlowModel(Y=Y, X=X, Z_cont=Z_cont)
    assert model.conf_shape == 3
    assert model.Z_disc is None


def test_construct_with_both_z_disc_and_z_cont():
    """Bug #8 fix: both confounder blocks together must not crash."""
    Y = np.zeros((10, 1))
    X = np.zeros((10, 1))
    Z_disc = np.zeros((10, 2))
    Z_cont = np.zeros((10, 3))
    model = benchmarking.FrugalFlowModel(Y=Y, X=X, Z_disc=Z_disc, Z_cont=Z_cont)
    assert model.conf_shape == 5


def test_construct_with_no_confounders():
    """Degenerate path: both Z blocks None still works (was the only path
    that worked pre-fix; check it still works post-fix)."""
    Y = np.zeros((10, 1))
    X = np.zeros((10, 1))
    model = benchmarking.FrugalFlowModel(Y=Y, X=X)
    assert model.conf_shape == 0
    assert model.Z_disc is None
    assert model.Z_cont is None
