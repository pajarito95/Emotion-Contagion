"""
run_from_config.py

Run simulations from a YAML configuration file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from metrics import plot_sentiment_evolution
from run_multiple_simulations import run_multiple_simulations


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file was not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("The YAML config must load as a dictionary.")
    return config


def run_from_config(config: Dict[str, Any]):
    mode = config.get("mode", "single")
    seeds = config.get("seeds")
    output_cfg = config.get("output", {})

    common_kwargs = {
        "seeds": seeds,
        "output_root": output_cfg.get("output_root"),
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

    if mode == "single":
        single = config.get("single_condition", {})
        batch = run_multiple_simulations(**common_kwargs, **single)
    elif mode == "manual":
        conditions = config.get("manual_conditions", [])
        batch = run_multiple_simulations(**common_kwargs, conditions=conditions)
    elif mode == "grid":
        import itertools
        import pandas as pd
        
        grid_cfg = config.get("grid", {})
        fixed_params = grid_cfg.get("fixed_params", {})
        condition_grid = grid_cfg.get("condition_grid", {})
        condition_name_keys = grid_cfg.get("condition_name_keys", list(condition_grid.keyd()))

        keys = list(condition_grid.keys())
        values = [condition_grid[k] for k in keys]

        all_results = []
        summary_dfs = []
        output_dirs = []
        summary_paths = []

        for combo in itertools.product(*values):
            condition_params = dict(zip(keys, combo))
            condition_name = "_".join(f"{k}-{condition_params[k]}" for k in condition_name_keys if k in condition_params)

        merged = {**common_kwargs, **fixed_params, **condition_params,  "condition_name": condition_name}
        
        batch_i = run_multiple_simulations(**_filter_kwargs_for_callable(run_multiple_simulations, merged))
        
        all_results.extend(batch_i["results"])
        
        if batch_i.get("summary_df") is not None:
            summary_dfs.append(batch_i["summary_df"])

        output_dirs.append(batch_i["output_dir"])
        summary_paths.append(batch_i["summary_path"])

        summary_df = pd.concat(summary_dgs, ignore_index=True) if summary_dfs else None
        
        batch = run_multiple_simulations(
            "results": all_results,
            "summary_df": summary_df,
            "output_dir": output_dirs[0] if output_dirs else None,
            "summary_path": summary_paths[0] if summary_paths else None
            )
    else:
        raise ValueError(f"Unsupported mode {mode!r}. Choose from 'single', 'manual', or 'grid'.")

    if output_cfg.get("make_sentiment_plots", True):
        plots_dir = Path(batch["output_dir"]) / "sentiment_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for results in batch["results"]:
            plot_sentiment_evolution(results, save_path=plots_dir / f"sentiment_run_{results.run_id}.png", show=False)

    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the emotion contagion ABM from a YAML config file.")
    parser.add_argument("--config", default="default.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()
    batch = run_from_config(load_config(args.config))
    print(f"Completed {len(batch['results'])} runs.")
    print(f"Outputs saved under: {batch['output_dir']}")
    if batch.get("summary_path") is not None:
        print(f"Summary CSV: {batch['summary_path']}")


if __name__ == "__main__":
    main()
