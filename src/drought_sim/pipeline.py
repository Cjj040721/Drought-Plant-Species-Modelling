import argparse
import json
from pathlib import Path

from .climate import compute_stress_index, fetch_daily_weather, synthetic_climate
from .community import run_richness_experiments, summarise_experiments
from .reporting import save_report_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate plant communities under observed or synthetic climate stress")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--latitude", type=float, default=39.9042)
    parser.add_argument("--longitude", type=float, default=116.4074)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.offline:
        climate_raw = synthetic_climate()
        source = "Deterministic synthetic climate"
    else:
        climate_raw = fetch_daily_weather(args.latitude, args.longitude, args.start, args.end)
        source = "Open-Meteo Historical Weather API"
    climate = compute_stress_index(climate_raw)
    results = run_richness_experiments(climate["drought_index"].to_numpy(), replicates=args.replicates)
    summary = summarise_experiments(results)
    args.output.mkdir(parents=True, exist_ok=True)
    climate.to_csv(args.output / "climate_stress.csv", index=False)
    results.to_csv(args.output / "simulation_results.csv", index=False)
    summary.to_csv(args.output / "richness_summary.csv", index=False)
    save_report_figure(climate, summary, args.output / "simulation_report.png")
    metadata = {
        "climate_source": source,
        "latitude": None if args.offline else args.latitude,
        "longitude": None if args.offline else args.longitude,
        "simulation_start": climate["date"].min().date().isoformat(),
        "simulation_end": climate["date"].max().date().isoformat(),
        "simulated_communities": int(len(results)),
        "replicates_per_richness": args.replicates,
        "stress_definition": "0.75 rainfall deficit + 0.25 positive temperature anomaly; both clipped to [0, 1]",
        "interpretation": "Exploratory simulation, not an ecological forecast or management recommendation",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
