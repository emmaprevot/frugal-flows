"""Regression marker for bug #8 in ``benchmarking.FrugalFlowModel``.

The class uses ``if Z_disc != None:`` / ``== None`` (~10 sites). For an
array-valued confounder, ``array != None`` is elementwise → ``if array:`` →
``ValueError: ambiguous truth value``. So the class cannot be constructed with
real (multi-element) array confounders — its main use case. This pins the
known defect so a future fix (``is not None``) is deliberate.

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


def test_bug8_array_confounder_breaks_construction():
    """REGRESSION MARKER (bug #8): constructing with a multi-element array
    Z_disc hits `if Z_disc != None:` -> ambiguous truth value."""
    Y = np.zeros((10, 1))
    X = np.zeros((10, 1))
    Z_disc = np.zeros((10, 2))  # real array confounder -> elementwise != None
    with pytest.raises(ValueError, match="ambiguous"):
        benchmarking.FrugalFlowModel(Y=Y, X=X, Z_disc=Z_disc)
