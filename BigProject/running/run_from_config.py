"""
run_from_config.py

Run simulations from a YAML configuration file.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from metrics import plot_sentiment_evolution
from run_multiple_simulations import run_multiple_simulations


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML config file into a dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file was not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("The YAML config must load as a dictionary.")

    return config

def _resolve_seeds(config: Dict[str, Any]) -> list[int]:
    """
    Resolve seeds either from an explicit list or from a compact seed specification.
    """
    if "seeds" in config and config["seeds"] is not None:
        seeds = config["seeds"]
        if not isinstance(seeds, list):
            raise TypeError("config['seeds'] must be a list of integers.")
        return seeds

    if "seed_spec" in config and config["seed_spec"] is not None:
        seed_spec = config["seed_spec"]

        if not isinstance(seed_spec, dict):
            raise TypeError("config['seed_spec'] must be a dictionary.")

        start = seed_spec.get("start", 0)
        n_repetitions = seed_spec.get("n_repetitions")

        if n_repetitions is None:
            raise ValueError("seed_spec must include 'n_repetitions'.")

        if not isinstance(start, int):
            raise TypeError("seed_spec['start'] must be an integer.")

        if not isinstance(n_repetitions, int):
            raise TypeError("seed_spec['n_repetitions'] must be an integer.")

        if n_repetitions < 1:
            raise ValueError("seed_spec['n_repetitions'] must be at least 1.")

        return list(range(start, start + n_repetitions))

    raise ValueError("The config must contain either 'seeds' or 'seed_spec'.")


def _build_output_kwargs(output_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract output-related keyword arguments for run_multiple_simulations(...).

    If output_root is not provided in the YAML, default to PROJECT_ROOT so outputs
    are saved under BigProject/outputs rather than BigProject/running/outputs.
    """
    output_root = output_cfg.get("output_root")
    if output_root is None:
        output_root = str(PROJECT_ROOT)

    return {
        "output_root": output_root,
        "output_subdir": output_cfg.get("output_subdir", "outputs"),
        "create_subfolders": output_cfg.get("create_subfolders", True),
        "add_timestamp_to_output_dir": output_cfg.get("add_timestamp_to_output_dir", True),
        "separate_condition_folders": output_cfg.get("separate_condition_folders", True),
        "separate_run_folders": output_cfg.get("separate_run_folders", False),
        "save_full_results": output_cfg.get("save_full_results", True),
        "save_run_metadata": output_cfg.get("save_run_metadata", True),
        "save_summary": output_cfg.get("save_summary", True),
        "summary_filename": output_cfg.get("summary_filename", "summary.csv"),
        "results_filename_template": output_cfg.get("results_filename_template", "simulation_result_run_{run_id}.pkl"),
        "metadata_filename_template": output_cfg.get("metadata_filename_template", "simulation_result_run_{run_id}_metadata.json"),
    }


def _make_sentiment_plots(batch: Dict[str, Any], output_cfg: Dict[str, Any]) -> None:
    """
    Save sentiment evolution plots for each run if requested.
    """
    if not output_cfg.get("make_sentiment_plots", True):
        return

    output_dir = batch.get("output_dir")
    if output_dir is None:
        return

    plots_dir = Path(output_dir) / "sentiment_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for results in batch["results"]:
        plot_sentiment_evolution(
            results,
            save_path=plots_dir / f"sentiment_run_{results.run_id}.png",
            show=False,
        )


def _combine_batch_outputs(batches: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine multiple batch dictionaries into one.
    """
    all_results = []
    summary_dfs = []
    output_dir = None

    for batch in batches:
        all_results.extend(batch.get("results", []))

        summary_df = batch.get("summary_df")
        if summary_df is not None:
            summary_dfs.append(summary_df)

        if output_dir is None:
            output_dir = batch.get("output_dir")

    combined_summary = pd.concat(summary_dfs, ignore_index=True) if summary_dfs else None
    summary_path = None

    return {
        "results": all_results,
        "summary_df": combined_summary,
        "output_dir": output_dir,
        "summary_path": summary_path,
    }


def run_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run simulations according to the YAML config.

    Supported modes:
    - single
    - manual
    - grid
    """
    mode = config.get("mode", "single")
    seeds = _resolve_seeds(config)

    output_cfg = config.get("output", {})
    output_kwargs = _build_output_kwargs(output_cfg)

    if mode == "single":
        single = config.get("single_condition", {})
        batch = run_multiple_simulations(
            seeds=seeds,
            **single,
            **output_kwargs,
        )

    elif mode == "manual":
        conditions = config.get("manual_conditions", [])
        batch = run_multiple_simulations(
            seeds=seeds,
            conditions=conditions,
            **output_kwargs,
        )

    elif mode == "grid":
        grid_cfg = config.get("grid", {})
        condition_grid = grid_cfg.get("condition_grid", {})
        fixed_params = grid_cfg.get("fixed_params", {})
        condition_name_keys = grid_cfg.get("condition_name_keys")

        batch = run_multiple_simulations(
            seeds=seeds,
            condition_grid=condition_grid,
            fixed_params=fixed_params,
            condition_name_keys=condition_name_keys,
            **output_kwargs,
        )

    else:
        raise ValueError(f"Unsupported mode {mode!r}. Choose from 'single', 'manual', or 'grid'.")

    _make_sentiment_plots(batch, output_cfg)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the emotion contagion ABM from a YAML config file.")
    parser.add_argument("--config", default="default.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    batch = run_from_config(config)

    print(f"Completed {len(batch['results'])} runs.")
    print(f"Outputs saved under: {batch['output_dir']}")

    if batch.get("summary_df") is not None:
        print("Summary DataFrame created.")

    if batch.get("summary_path") is not None:
        print(f"Summary CSV: {batch['summary_path']}")


if __name__ == "__main__":
    main()