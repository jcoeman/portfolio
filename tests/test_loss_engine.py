"""
Sanity checks for sim/loss_engine.py against known analytic properties.
These are deviations-mean-a-bug checks, not just "does it run" checks:

- E[Aggregate] ~= E[Frequency] * E[Severity]  (compound distribution mean)
- ILF(x) is monotone non-decreasing and concave in x
- LER(d) is monotone non-decreasing in d
- Coverage layer ceded + retained reconciles to the pre-layer total
- Weibull/Pareto/Gamma mean-CV parameterization round-trips to the target mean

Run directly with `python tests/test_loss_engine.py` (no pytest dependency).
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.loss_engine import (
    CoverageLayer,
    FrequencySpec,
    SeveritySpec,
    aggregate_statistics,
    apply_coverage_layers,
    ilf_curve,
    ler_curve,
    limited_expected_value,
    simulate_aggregate_losses,
    var_tvar_table,
)

N_YEARS = 200_000
SEED = 12345
RTOL = 0.02  # 2% simulation-error tolerance at 200k years


def test_poisson_lognormal_aggregate_mean():
    freq = FrequencySpec(dist="poisson", mean=5.0)
    sev = SeveritySpec(dist="lognormal", mean=10_000.0, cv=1.5)
    sim = simulate_aggregate_losses(freq, sev, N_YEARS, SEED)

    expected = freq.analytic_mean() * sev.analytic_mean()
    actual = sim.aggregate_losses.mean()
    assert math.isclose(actual, expected, rel_tol=RTOL), f"E[Agg]={actual:,.1f} vs expected {expected:,.1f}"


def test_negative_binomial_pareto_aggregate_mean():
    freq = FrequencySpec(dist="negative_binomial", mean=8.0, variance=24.0)
    sev = SeveritySpec(dist="pareto", mean=25_000.0, cv=2.0)
    sim = simulate_aggregate_losses(freq, sev, N_YEARS, SEED)

    expected = freq.analytic_mean() * sev.analytic_mean()
    actual = sim.aggregate_losses.mean()
    assert math.isclose(actual, expected, rel_tol=RTOL), f"E[Agg]={actual:,.1f} vs expected {expected:,.1f}"


def test_gamma_weibull_severity_means():
    for dist, mean, cv in [("gamma", 5_000.0, 0.8), ("weibull", 5_000.0, 0.8), ("weibull", 5_000.0, 2.0)]:
        spec = SeveritySpec(dist=dist, mean=mean, cv=cv)
        rng = np.random.default_rng(SEED)
        sample = spec.sample(500_000, rng)
        assert math.isclose(sample.mean(), mean, rel_tol=RTOL), (
            f"{dist} cv={cv}: sample mean {sample.mean():,.1f} vs target {mean:,.1f}"
        )
        assert math.isclose(spec.analytic_mean(), mean, rel_tol=1e-6), (
            f"{dist} cv={cv}: analytic mean {spec.analytic_mean():,.1f} vs target {mean:,.1f}"
        )


def test_ilf_monotone_and_concave():
    sev = SeveritySpec(dist="lognormal", mean=10_000.0, cv=1.5)
    rng = np.random.default_rng(SEED)
    sample = sev.sample(500_000, rng)

    limits = np.array([25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000])
    df = ilf_curve(sample, limits, base_limit=100_000)

    ilf = df["ILF"].to_numpy()
    assert np.all(np.diff(ilf) >= -1e-9), f"ILF not monotone non-decreasing: {ilf}"

    # concavity: successive differences (per unit limit) should not increase
    diffs = np.diff(ilf) / np.diff(limits)
    assert np.all(np.diff(diffs) <= 1e-12), f"ILF not concave, marginal ILF/$ increased: {diffs}"


def test_ler_monotone():
    sev = SeveritySpec(dist="gamma", mean=8_000.0, cv=1.2)
    rng = np.random.default_rng(SEED)
    sample = sev.sample(500_000, rng)

    deductibles = np.array([0, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000])
    df = ler_curve(sample, deductibles)

    ler = df["LER"].to_numpy()
    assert np.all(np.diff(ler) >= -1e-9), f"LER not monotone non-decreasing: {ler}"
    assert math.isclose(ler[0], 0.0, abs_tol=1e-9), "LER at deductible=0 should be 0"
    assert ler[-1] <= 1.0 + 1e-9, "LER should not exceed 1"


def test_coverage_layers_reconcile():
    freq = FrequencySpec(dist="poisson", mean=6.0)
    sev = SeveritySpec(dist="lognormal", mean=20_000.0, cv=1.8)
    sim = simulate_aggregate_losses(freq, sev, 50_000, SEED)

    layers = [
        CoverageLayer(name="Primary ded/limit", kind="occurrence", attachment=5_000, limit=95_000),
        CoverageLayer(name="XOL 500K xs 500K", kind="occurrence", attachment=500_000, limit=500_000),
        CoverageLayer(name="Agg stop-loss", kind="aggregate", attachment=1_000_000, limit=2_000_000),
    ]
    stages, final_occ, final_annual = apply_coverage_layers(sim, layers)

    pre_layer_total = sim.occurrence_losses.sum()
    ceded_sum = sum(s.ceded_total for s in stages)
    retained_final = final_occ.sum()
    assert math.isclose(ceded_sum + retained_final, pre_layer_total, rel_tol=1e-9), (
        "ceded + final retained should reconcile to the pre-layer total"
    )

    for s in stages:
        assert s.ceded_total >= -1e-6
        assert s.retained_total >= -1e-6


def test_quota_share_layer():
    freq = FrequencySpec(dist="poisson", mean=4.0)
    sev = SeveritySpec(dist="gamma", mean=15_000.0, cv=1.0)
    sim = simulate_aggregate_losses(freq, sev, 20_000, SEED)

    layers = [CoverageLayer(name="30% QS", kind="quota_share", quota_share_pct=0.3)]
    stages, final_occ, _ = apply_coverage_layers(sim, layers)

    total = sim.occurrence_losses.sum()
    assert math.isclose(stages[0].ceded_total, 0.3 * total, rel_tol=1e-9)
    assert math.isclose(final_occ.sum(), 0.7 * total, rel_tol=1e-9)


def test_var_tvar_ordering():
    freq = FrequencySpec(dist="poisson", mean=5.0)
    sev = SeveritySpec(dist="lognormal", mean=10_000.0, cv=1.5)
    sim = simulate_aggregate_losses(freq, sev, N_YEARS, SEED)

    df = var_tvar_table(sim.aggregate_losses, [0.95, 0.99, 0.995])
    assert np.all(np.diff(df["VaR"].to_numpy()) >= 0), "VaR should increase with confidence level"
    assert np.all(df["TVaR"].to_numpy() >= df["VaR"].to_numpy() - 1e-9), "TVaR should be >= VaR at each level"


def test_aggregate_statistics_reasonable():
    freq = FrequencySpec(dist="poisson", mean=5.0)
    sev = SeveritySpec(dist="lognormal", mean=10_000.0, cv=1.5)
    sim = simulate_aggregate_losses(freq, sev, N_YEARS, SEED)
    stats = aggregate_statistics(sim.aggregate_losses)
    assert stats["mean"] > 0
    assert stats["std"] > 0
    assert stats["cv"] > 0
    # compound Poisson with heavy-tailed severity should be right-skewed
    assert stats["skewness"] > 0


TESTS = [
    test_poisson_lognormal_aggregate_mean,
    test_negative_binomial_pareto_aggregate_mean,
    test_gamma_weibull_severity_means,
    test_ilf_monotone_and_concave,
    test_ler_monotone,
    test_coverage_layers_reconcile,
    test_quota_share_layer,
    test_var_tvar_ordering,
    test_aggregate_statistics_reasonable,
]


if __name__ == "__main__":
    failures = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    if failures:
        sys.exit(1)
