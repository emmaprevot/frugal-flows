"""Shared pytest fixtures.

Math-grounded tests need synthetic data with a known causal effect. We don't
want every test run to call into R, so the causal-fixture data is generated
once by ``tests/fixtures/_generate.py`` and cached on disk as .npz. Tests load
the cached arrays via the ``causl_*`` fixtures here.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def key():
    """Deterministic top-level PRNG key for tests."""
    return jr.PRNGKey(0)


def _load_npz(name: str) -> dict:
    path = FIXTURES_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(
            f"fixture {path.name} missing — run `python tests/fixtures/_generate.py` "
            "to regenerate (requires R + causl, see install_rpy2_libraries.py)."
        )
    raw = np.load(path, allow_pickle=False)
    missing = set(raw["_missing"].tolist()) if "_missing" in raw.files else set()
    out: dict = {}
    for k in ("Z_cont", "Z_disc", "X", "Y"):
        out[k] = None if k in missing else jnp.asarray(raw[k])
    out["meta"] = {k[len("meta_"):]: raw[k].item() for k in raw.files if k.startswith("meta_")}
    return out


@pytest.fixture(scope="session")
def causl_gaussian_known_ate():
    """Causl-simulated dataset with Gaussian Z, continuous X, continuous Y.

    Returns a dict with keys Z_cont, Z_disc (None), X, Y, and meta (ate, const,
    scale, n, seed). Known ATE = meta['ate'] = 1.0.
    """
    return _load_npz("gaussian_known_ate")
