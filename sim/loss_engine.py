"""
Monte Carlo aggregate loss simulation engine.

Pure numpy/pandas math, independent of any UI framework so it can be
unit-tested directly. Two building blocks:

- FrequencySpec / SeveritySpec: parameterize and sample the frequency and
  severity distributions that drive the collective risk model.
- CoverageLayer: a single (attachment, limit) transform applied either
  per-occurrence or to a year's aggregate loss. Deductibles, limits, XOL
  reinsurance layers, aggregate stop-loss, and quota share are all
  expressed with the same abstraction and can be stacked.

simulate_aggregate_losses() draws N years of claim experience and returns
the full per-occurrence and per-year arrays (not just summary stats) so
downstream calcs (ILFs, LER, layering) reuse the same draws instead of
re-simulating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

# =============================================================================
# Frequency
# =============================================================================

FrequencyDist = Literal["poisson", "negative_binomial"]


@dataclass(frozen=True)
class FrequencySpec:
    """Claim-count distribution. NB is parameterized by mean & variance
    (variance must exceed mean — that's what makes it overdispersed
    relative to Poisson, where variance == mean)."""

    dist: FrequencyDist
    mean: float
    variance: Optional[float] = None  # required, and must be > mean, for negative_binomial

    def __post_init__(self):
        if self.mean <= 0:
            raise ValueError("Frequency mean must be positive.")
        if self.dist == "negative_binomial":
            if self.variance is None or self.variance <= self.mean:
                raise ValueError(
                    "Negative binomial requires variance > mean (overdispersion). "
                    f"Got mean={self.mean}, variance={self.variance}."
                )

    def _nb_r_p(self) -> tuple[float, float]:
        p = self.mean / self.variance
        r = self.mean ** 2 / (self.variance - self.mean)
        return r, p

    def analytic_mean(self) -> float:
        return self.mean

    def analytic_variance(self) -> float:
        if self.dist == "poisson":
            return self.mean
        return self.variance

    def sample(self, n_years: int, rng: np.random.Generator) -> np.ndarray:
        if self.dist == "poisson":
            return rng.poisson(self.mean, size=n_years)
        elif self.dist == "negative_binomial":
            r, p = self._nb_r_p()
            return rng.negative_binomial(r, p, size=n_years)
        raise ValueError(f"Unknown frequency distribution: {self.dist}")


# =============================================================================
# Severity
# =============================================================================

SeverityDist = Literal["lognormal", "pareto", "gamma", "weibull"]


def _weibull_shape_from_cv(cv: float, lo: float = 0.02, hi: float = 100.0, tol: float = 1e-9, max_iter: int = 200) -> float:
    """Solve for Weibull shape k such that CV(k) matches the target CV.
    CV^2(k) = Gamma(1+2/k)/Gamma(1+1/k)^2 - 1 is strictly decreasing in k,
    so bisection on [lo, hi] is safe."""

    def cv2(k: float) -> float:
        g1 = math.gamma(1 + 1 / k)
        g2 = math.gamma(1 + 2 / k)
        return g2 / g1 ** 2 - 1

    target = cv ** 2
    lo_val, hi_val = cv2(lo), cv2(hi)
    if target > lo_val or target < hi_val:
        raise ValueError(f"CV={cv} out of solvable range for Weibull shape in [{lo}, {hi}].")

    a, b = lo, hi
    for _ in range(max_iter):
        mid = (a + b) / 2
        m = cv2(mid)
        if abs(m - target) < tol:
            return mid
        if m > target:
            a = mid
        else:
            b = mid
    return (a + b) / 2


@dataclass(frozen=True)
class SeveritySpec:
    """Per-claim severity distribution. Provide `mean` + `cv` for the
    intuitive parameterization, or raw distribution parameters to override
    (raw params take precedence when supplied)."""

    dist: SeverityDist
    mean: Optional[float] = None
    cv: Optional[float] = None
    # raw overrides
    mu: Optional[float] = None       # lognormal
    sigma: Optional[float] = None    # lognormal
    alpha: Optional[float] = None    # pareto shape / gamma shape
    x_m: Optional[float] = None      # pareto scale (minimum value)
    beta: Optional[float] = None     # gamma scale
    k: Optional[float] = None        # weibull shape
    lam: Optional[float] = None      # weibull scale

    def resolved(self) -> dict:
        """Return the raw distribution parameters, deriving them from
        mean/cv when raw params weren't supplied directly."""
        if self.dist == "lognormal":
            if self.mu is not None and self.sigma is not None:
                return {"mu": self.mu, "sigma": self.sigma}
            sigma2 = math.log(1 + self.cv ** 2)
            sigma = math.sqrt(sigma2)
            mu = math.log(self.mean) - sigma2 / 2
            return {"mu": mu, "sigma": sigma}

        if self.dist == "pareto":
            if self.alpha is not None and self.x_m is not None:
                return {"alpha": self.alpha, "x_m": self.x_m}
            alpha = 1 + math.sqrt(1 + 1 / self.cv ** 2)
            x_m = self.mean * (alpha - 1) / alpha
            return {"alpha": alpha, "x_m": x_m}

        if self.dist == "gamma":
            if self.alpha is not None and self.beta is not None:
                return {"alpha": self.alpha, "beta": self.beta}
            alpha = 1 / self.cv ** 2
            beta = self.mean * self.cv ** 2
            return {"alpha": alpha, "beta": beta}

        if self.dist == "weibull":
            if self.k is not None and self.lam is not None:
                return {"k": self.k, "lam": self.lam}
            k = _weibull_shape_from_cv(self.cv)
            lam = self.mean / math.gamma(1 + 1 / k)
            return {"k": k, "lam": lam}

        raise ValueError(f"Unknown severity distribution: {self.dist}")

    def analytic_mean(self) -> float:
        p = self.resolved()
        if self.dist == "lognormal":
            return math.exp(p["mu"] + p["sigma"] ** 2 / 2)
        if self.dist == "pareto":
            return p["alpha"] * p["x_m"] / (p["alpha"] - 1)
        if self.dist == "gamma":
            return p["alpha"] * p["beta"]
        if self.dist == "weibull":
            return p["lam"] * math.gamma(1 + 1 / p["k"])
        raise ValueError(f"Unknown severity distribution: {self.dist}")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n == 0:
            return np.array([])
        p = self.resolved()
        if self.dist == "lognormal":
            return rng.lognormal(p["mu"], p["sigma"], size=n)
        if self.dist == "pareto":
            # np's pareto() draws Lomax(alpha); x_m*(1+Lomax) is standard
            # single-parameter Pareto Type I with minimum x_m.
            return p["x_m"] * (1 + rng.pareto(p["alpha"], size=n))
        if self.dist == "gamma":
            return rng.gamma(p["alpha"], p["beta"], size=n)
        if self.dist == "weibull":
            return rng.weibull(p["k"], size=n) * p["lam"]
        raise ValueError(f"Unknown severity distribution: {self.dist}")


# =============================================================================
# Simulation
# =============================================================================

@dataclass
class SimulationResult:
    seed: int
    n_years: int
    freq_spec: FrequencySpec
    sev_spec: SeveritySpec
    claim_counts: np.ndarray        # (n_years,)
    occurrence_losses: np.ndarray   # (total_claims,) flattened across all years
    occurrence_year: np.ndarray     # (total_claims,) year index of each occurrence
    aggregate_losses: np.ndarray    # (n_years,) sum of severities per year (0 if no claims)


def simulate_aggregate_losses(
    freq_spec: FrequencySpec,
    sev_spec: SeveritySpec,
    n_years: int,
    seed: int,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    claim_counts = freq_spec.sample(n_years, rng)
    total_claims = int(claim_counts.sum())

    if total_claims == 0:
        occurrence_losses = np.array([])
        occurrence_year = np.array([], dtype=int)
        aggregate_losses = np.zeros(n_years)
    else:
        occurrence_losses = sev_spec.sample(total_claims, rng)
        occurrence_year = np.repeat(np.arange(n_years), claim_counts)
        aggregate_losses = np.bincount(occurrence_year, weights=occurrence_losses, minlength=n_years)

    return SimulationResult(
        seed=seed,
        n_years=n_years,
        freq_spec=freq_spec,
        sev_spec=sev_spec,
        claim_counts=claim_counts,
        occurrence_losses=occurrence_losses,
        occurrence_year=occurrence_year,
        aggregate_losses=aggregate_losses,
    )


# =============================================================================
# Coverage layers
# =============================================================================

LayerKind = Literal["occurrence", "aggregate", "quota_share"]


@dataclass(frozen=True)
class CoverageLayer:
    """A single (attachment, limit) transform on the loss distribution.

    - kind="occurrence": applied to each simulated claim independently.
      Covers both a primary per-occurrence deductible/limit and a
      per-occurrence XOL reinsurance layer (e.g. $500K xs $500K) — they're
      the same operation: ceded = min(max(loss - attachment, 0), limit).
    - kind="aggregate": applied once to each year's total retained loss
      (an aggregate stop-loss), not to individual occurrences.
    - kind="quota_share": a flat % cession, no attachment/limit.
    """

    name: str
    kind: LayerKind
    attachment: float = 0.0
    limit: float = float("inf")
    quota_share_pct: float = 0.0


@dataclass
class LayerStage:
    layer: CoverageLayer
    ceded_total: float          # total ceded loss dollars, summed across all simulated years
    retained_total: float       # total retained loss dollars after this layer
    n_years: int

    @property
    def ceded_loss_cost(self) -> float:
        return self.ceded_total / self.n_years

    @property
    def retained_loss_cost(self) -> float:
        return self.retained_total / self.n_years


def apply_coverage_layers(
    sim: SimulationResult,
    layers: list[CoverageLayer],
) -> tuple[list[LayerStage], np.ndarray, np.ndarray]:
    """Apply a stack of coverage layers in order. Each layer transforms the
    *retained* loss from the previous stage. Returns per-layer stage
    results plus the final retained occurrence-level and year-level arrays.
    """
    occ = sim.occurrence_losses.copy()
    year = sim.occurrence_year
    n_years = sim.n_years
    stages: list[LayerStage] = []

    for layer in layers:
        if layer.kind == "occurrence":
            ceded = np.clip(occ - layer.attachment, 0, layer.limit)
            retained = occ - ceded
        elif layer.kind == "quota_share":
            pct = layer.quota_share_pct
            ceded = occ * pct
            retained = occ * (1 - pct)
        elif layer.kind == "aggregate":
            annual = np.bincount(year, weights=occ, minlength=n_years) if len(occ) else np.zeros(n_years)
            ceded_annual = np.clip(annual - layer.attachment, 0, layer.limit)
            retained_annual = annual - ceded_annual
            scale = np.ones(n_years)
            nonzero = annual > 0
            scale[nonzero] = retained_annual[nonzero] / annual[nonzero]
            retained = occ * scale[year] if len(occ) else occ
            ceded = occ - retained
        else:
            raise ValueError(f"Unknown layer kind: {layer.kind}")

        stages.append(LayerStage(
            layer=layer,
            ceded_total=float(ceded.sum()),
            retained_total=float(retained.sum()),
            n_years=n_years,
        ))
        occ = retained

    final_retained_occ = occ
    final_retained_annual = np.bincount(year, weights=occ, minlength=n_years) if len(occ) else np.zeros(n_years)
    return stages, final_retained_occ, final_retained_annual


# =============================================================================
# Increased Limits Factors / Loss Elimination Ratio
# =============================================================================

def limited_expected_value(severity_sample: np.ndarray, limit: float) -> float:
    """LEV(x) = E[min(Severity, x)], estimated from the simulated draws."""
    if len(severity_sample) == 0:
        return 0.0
    return float(np.mean(np.minimum(severity_sample, limit)))


def ilf_curve(severity_sample: np.ndarray, limits: np.ndarray, base_limit: float) -> pd.DataFrame:
    """ILF(x) = LEV(x) / LEV(base_limit) across a range of limits."""
    base_lev = limited_expected_value(severity_sample, base_limit)
    rows = []
    for lim in limits:
        lev = limited_expected_value(severity_sample, lim)
        rows.append({"limit": lim, "LEV": lev, "ILF": lev / base_lev if base_lev > 0 else np.nan})
    return pd.DataFrame(rows)


def ler_curve(severity_sample: np.ndarray, deductibles: np.ndarray) -> pd.DataFrame:
    """LER(d) = LEV(d) / E[Severity] across a range of deductibles."""
    mean_sev = float(np.mean(severity_sample)) if len(severity_sample) else np.nan
    rows = []
    for d in deductibles:
        lev = limited_expected_value(severity_sample, d)
        rows.append({"deductible": d, "LEV": lev, "LER": lev / mean_sev if mean_sev else np.nan})
    return pd.DataFrame(rows)


# =============================================================================
# Aggregate distribution statistics
# =============================================================================

def aggregate_statistics(aggregate_losses: np.ndarray) -> dict:
    mean = float(np.mean(aggregate_losses))
    std = float(np.std(aggregate_losses, ddof=1))
    cv = std / mean if mean else np.nan
    skew = float(np.mean((aggregate_losses - mean) ** 3) / std ** 3) if std > 0 else np.nan
    return {"mean": mean, "std": std, "cv": cv, "skewness": skew}


def var_tvar_table(aggregate_losses: np.ndarray, confidence_levels: list[float]) -> pd.DataFrame:
    """VaR_q = q-th quantile of the aggregate loss distribution.
    TVaR_q (a.k.a. CTE_q) = E[Aggregate | Aggregate >= VaR_q]."""
    rows = []
    for q in confidence_levels:
        var = float(np.quantile(aggregate_losses, q))
        tail = aggregate_losses[aggregate_losses >= var]
        tvar = float(tail.mean()) if len(tail) > 0 else var
        rows.append({"confidence": q, "VaR": var, "TVaR": tvar})
    return pd.DataFrame(rows)


# =============================================================================
# ALAE / ULAE loading
# =============================================================================

def apply_alae_ulae(aggregate_losses: np.ndarray, alae_pct: float = 0.0, ulae_pct: float = 0.0) -> np.ndarray:
    """Simple proportional loads: ALAE loads losses, ULAE loads the
    resulting (loss + ALAE) ultimate. Both default to 0 (no-op)."""
    with_alae = aggregate_losses * (1 + alae_pct)
    ultimate = with_alae * (1 + ulae_pct)
    return ultimate


# =============================================================================
# Credibility tie-in (Bühlmann-Straub style blending)
# =============================================================================

def credibility_weighted_loss_cost(
    class_mean: float,
    class_exposure: float,
    prior_mean: float,
    k: float,
) -> tuple[float, float]:
    """Blend this simulated scenario's mean loss cost (treated as observed
    class experience with exposure = class_exposure) against a broader
    prior mean, using the same Z = m/(m+k) credibility pattern as the
    Bühlmann-Straub calculator. Returns (Z, blended_estimate)."""
    z = class_exposure / (class_exposure + k)
    blended = z * class_mean + (1 - z) * prior_mean
    return z, blended
