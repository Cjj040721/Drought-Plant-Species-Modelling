from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class SpeciesPool:
    growth: np.ndarray
    carrying_capacity: np.ndarray
    drought_tolerance: np.ndarray


def make_species_pool(count: int = 16, seed: int = 19) -> SpeciesPool:
    rng = np.random.default_rng(seed)
    return SpeciesPool(
        growth=rng.uniform(0.035, 0.075, count),
        carrying_capacity=rng.uniform(35, 75, count),
        drought_tolerance=rng.uniform(0.15, 0.9, count),
    )


def simulate_community(
    pool: SpeciesPool,
    selected_species: np.ndarray,
    drought_index: np.ndarray,
    seed: int,
    interaction_strength: float = 0.22,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    growth = pool.growth[selected_species]
    capacity = pool.carrying_capacity[selected_species]
    tolerance = pool.drought_tolerance[selected_species]
    count = len(selected_species)
    interaction = rng.uniform(0.05, interaction_strength, (count, count))
    np.fill_diagonal(interaction, 1.0)
    competition = growth[:, None] / capacity[:, None] * interaction
    initial = rng.uniform(5, 12, count)
    days = np.arange(len(drought_index), dtype=float)

    def dynamics(_: float, biomass: np.ndarray) -> np.ndarray:
        day = min(int(_), len(drought_index) - 1)
        stress_multiplier = 1 - (1 - tolerance) * 0.9 * drought_index[day]
        biomass = np.maximum(biomass, 0)
        return biomass * (growth * stress_multiplier - competition @ biomass)

    solution = solve_ivp(
        dynamics,
        (0, len(drought_index) - 1),
        initial,
        t_eval=days,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    biomass = np.maximum(solution.y.T, 0)
    return pd.DataFrame({
        "day": days,
        "total_biomass": biomass.sum(axis=1),
        "persistent_species": (biomass > 0.1).sum(axis=1),
    })


def run_richness_experiments(
    drought_index: np.ndarray,
    richness_levels: tuple[int, ...] = (1, 2, 4, 8, 16),
    replicates: int = 12,
) -> pd.DataFrame:
    pool = make_species_pool()
    rows = []
    for richness in richness_levels:
        for replicate in range(replicates):
            seed = 1000 + richness * 100 + replicate
            rng = np.random.default_rng(seed)
            selected = np.sort(rng.choice(len(pool.growth), richness, replace=False))
            drought = simulate_community(pool, selected, drought_index, seed)
            control = simulate_community(pool, selected, np.zeros_like(drought_index), seed)
            ratio = drought["total_biomass"].to_numpy() / np.maximum(control["total_biomass"].to_numpy(), 1e-9)
            rows.append({
                "richness": richness,
                "replicate": replicate,
                "seed": seed,
                "biomass_retention": ratio[-1],
                "minimum_biomass_ratio": ratio[30:].min(),
                "final_persistence": drought["persistent_species"].iloc[-1] / richness,
                "final_biomass": drought["total_biomass"].iloc[-1],
            })
    return pd.DataFrame(rows)


def summarise_experiments(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby("richness").agg(
        biomass_retention_mean=("biomass_retention", "mean"),
        biomass_retention_std=("biomass_retention", "std"),
        minimum_biomass_ratio_mean=("minimum_biomass_ratio", "mean"),
        final_persistence_mean=("final_persistence", "mean"),
        replicates=("replicate", "count"),
    ).reset_index()
    summary["retention_ci95"] = 1.96 * summary["biomass_retention_std"] / np.sqrt(summary["replicates"])
    return summary
