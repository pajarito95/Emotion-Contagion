"""
Batch runner for the emotion contagion ABM.
- Run multiple simulations across one or many conditions.
- Build conditions manually or automatically from parameter grids.
- Save full SimulationResults objects as pickle files.
- Save readable JSON metadata files alongside pickles.
- Save a compact batch summary CSV.
- Allow user-defined parent output path.
- Fall back to the current working directory when no path is provided.
- Optionally timestamp output folders to prevent overwriting.
"""

from __future__ import annotations

from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import json
import pickle
import pandas as pd

from run_simulation import run_simulation


DEFAULT_OUTPUT_SUBDIR = "outputs"


def resolve_output_root(output_root: Optional[str | Path] = None) -> Path:
    if output_root is None:
        return Path.cwd().resolve()
    return Path(output_root).expanduser().resolve()


def make_timestamp_string() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def prepare_output_directory(
    output_root: Optional[str | Path] = None,
    output_subdir: Optional[str] = DEFAULT_OUTPUT_SUBDIR,
    create_subfolders: bool = True,
    add_timestamp_to_output_dir: bool = False,
) -> Path:
    root = resolve_output_root(output_root)

    if create_subfolders and output_subdir:
        final_subdir = output_subdir
        if add_timestamp_to_output_dir:
            final_subdir = f"{output_subdir}_{make_timestamp_string()}"
        outdir = root / final_subdir
    else:
        outdir = root

    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _validate_seeds(seeds: Sequence[int]) -> None:
    if not isinstance(seeds, Sequence) or len(seeds) == 0:
        raise ValueError("seeds must be a non-empty sequence of integers.")

    for seed in seeds:
        if not isinstance(seed, int):
            raise TypeError(f"Each seed must be an integer, but received {type(seed).__name__}.")


def _validate_conditions(conditions: Sequence[Dict[str, Any]]) -> None:
    if not isinstance(conditions, Sequence) or len(conditions) == 0:
        raise ValueError("conditions must be a non-empty sequence of dictionaries.")

    for i, condition in enumerate(conditions):
        if not isinstance(condition, dict):
            raise TypeError(f"conditions[{i}] must be a dictionary, but received {type(condition).__name__}.")

        required = {"population_size", "structure", "leader_style", "max_iterations"}
        missing = required - set(condition.keys())
        if missing:
            raise KeyError(f"conditions[{i}] is missing required key(s): {sorted(missing)}.")


def _make_run_output_dir(
    base_output_dir: Path,
    condition_name: Optional[str],
    run_id: Any,
    separate_condition_folders: bool,
    separate_run_folders: bool,
) -> Path:
    outdir = base_output_dir

    if separate_condition_folders:
        safe_condition_name = condition_name if condition_name else "unnamed_condition"
        outdir = outdir / safe_condition_name
        outdir.mkdir(parents=True, exist_ok=True)

    if separate_run_folders:
        outdir = outdir / f"run_{run_id}"
        outdir.mkdir(parents=True, exist_ok=True)

    return outdir


def _results_to_summary_row(results, condition_name: Optional[str] = None, condition_index: Optional[int] = None) -> Dict[str, Any]:
    final_avg_emotion = None
    if results.avg_emotion_history:
        final_avg_emotion = results.avg_emotion_history[-1]

    final_member_emotions = [agent["emotion"] for agent in results.state.agents if agent.get("role") == "member"]
    leader = results.state.agents[results.state.leader_index]

    return {
        "condition_index": condition_index,
        "condition_name": condition_name,
        "run_id": results.run_id,
        "seed": results.seed,
        "n_agents": len(results.state.agents),
        "leader_index": results.state.leader_index,
        "leader_style": results.metadata.get("leader_style"),
        "structure": results.metadata.get("structure"),
        "max_iterations": results.metadata.get("max_iterations"),
        "adaptive_intimacy": results.metadata.get("adaptive_intimacy"),
        "num_interventions": len(results.intervention_timesteps),
        "intervention_timesteps": results.intervention_timesteps,
        "total_interactions": sum(results.interactions_per_timestep),
        "mean_interactions_per_timestep": (
            sum(results.interactions_per_timestep) / len(results.interactions_per_timestep)
            if results.interactions_per_timestep
            else 0.0
        ),
        "final_avg_member_emotion": final_avg_emotion,
        "final_min_member_emotion": min(final_member_emotions) if final_member_emotions else None,
        "final_max_member_emotion": max(final_member_emotions) if final_member_emotions else None,
        "leader_emotion": leader.get("emotion"),
        "leader_threshold": leader.get("interventionThreshold"),
        "leader_emotion_management_ability": leader.get("emotionManagementAbility"),
    }


def save_simulation_result(results, filepath: Path) -> None:
    with filepath.open("wb") as f:
        pickle.dump(results, f)


def save_run_metadata_json(metadata: Dict[str, Any], filepath: Path) -> None:
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def save_summary_table(summary_rows: List[Dict[str, Any]], filepath: Path) -> pd.DataFrame:
    df = pd.DataFrame(summary_rows)
    df.to_csv(filepath, index=False)
    return df


def build_conditions_from_grid(
    grid: Dict[str, Sequence[Any]],
    fixed_params: Optional[Dict[str, Any]] = None,
    condition_name_keys: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(grid, dict) or len(grid) == 0:
        raise ValueError("grid must be a non-empty dictionary of parameter lists.")

    for key, values in grid.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) == 0:
            raise ValueError(f"grid[{key!r}] must be a non-empty sequence of values.")

    fixed_params = dict(fixed_params) if fixed_params is not None else {}
    keys = list(grid.keys())
    values_product = product(*(grid[key] for key in keys))

    if condition_name_keys is None:
        condition_name_keys = keys

    conditions = []
    for combo in values_product:
        condition = dict(fixed_params)
        condition.update(dict(zip(keys, combo)))

        name_parts = []
        for key in condition_name_keys:
            if key in condition:
                name_parts.append(f"{key}_{condition[key]}")
        condition["condition_name"] = "__".join(name_parts) if name_parts else "condition"

        conditions.append(condition)

    return conditions


def _build_single_condition_from_arguments(
    population_size: Optional[int],
    structure: Optional[str],
    leader_style: Optional[str],
    max_iterations: Optional[int],
    min_weight: float,
    include_leader_ties: bool,
    leader_to_member_cap: Optional[float],
    member_to_leader_cap: Optional[float],
    leader_to_member_value: Optional[float],
    member_to_leader_value: Optional[float],
    strength: Optional[float],
    n_communities: Optional[int],
    intra_strength: Optional[float],
    inter_strength: Optional[float],
    n_periphery: Optional[int],
    core_to_periph: Optional[float],
    periph_to_core: Optional[float],
    periph_to_periph: Optional[float],
    adaptive_intimacy: bool,
    kappa: Optional[float],
    decay: Optional[float],
    min_w: Optional[float],
    max_w: Optional[float],
    dampening: float,
    condition_name: Optional[str],
) -> Dict[str, Any]:
    required = {
        "population_size": population_size,
        "structure": structure,
        "leader_style": leader_style,
        "max_iterations": max_iterations,
    }

    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"When conditions is not provided, these arguments are required: {missing}.")

    return {
        "condition_name": condition_name,
        "population_size": population_size,
        "structure": structure,
        "leader_style": leader_style,
        "max_iterations": max_iterations,
        "min_weight": min_weight,
        "include_leader_ties": include_leader_ties,
        "leader_to_member_cap": leader_to_member_cap,
        "member_to_leader_cap": member_to_leader_cap,
        "leader_to_member_value": leader_to_member_value,
        "member_to_leader_value": member_to_leader_value,
        "strength": strength,
        "n_communities": n_communities,
        "intra_strength": intra_strength,
        "inter_strength": inter_strength,
        "n_periphery": n_periphery,
        "core_to_periph": core_to_periph,
        "periph_to_core": periph_to_core,
        "periph_to_periph": periph_to_periph,
        "adaptive_intimacy": adaptive_intimacy,
        "kappa": kappa,
        "decay": decay,
        "min_w": min_w,
        "max_w": max_w,
        "dampening": dampening,
    }


def _build_run_metadata(
    results,
    condition: Dict[str, Any],
    condition_index: int,
    condition_name: Optional[str],
) -> Dict[str, Any]:
    return {
        "run_id": results.run_id,
        "seed": results.seed,
        "condition_index": condition_index,
        "condition_name": condition_name,
        "condition": condition,
        "leader_index": results.state.leader_index,
        "n_agents": len(results.state.agents),
        "structure": results.metadata.get("structure"),
        "leader_style": results.metadata.get("leader_style"),
        "max_iterations": results.metadata.get("max_iterations"),
        "adaptive_intimacy": results.metadata.get("adaptive_intimacy"),
        "num_interventions": len(results.intervention_timesteps),
        "intervention_timesteps": results.intervention_timesteps,
    }


def run_multiple_simulations(
    seeds: Sequence[int],
    conditions: Optional[Sequence[Dict[str, Any]]] = None,
    condition_grid: Optional[Dict[str, Sequence[Any]]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    condition_name_keys: Optional[Sequence[str]] = None,
    population_size: Optional[int] = None,
    structure: Optional[str] = None,
    leader_style: Optional[str] = None,
    max_iterations: Optional[int] = None,
    min_weight: float = 0.01,
    include_leader_ties: bool = True,
    leader_to_member_cap: Optional[float] = None,
    member_to_leader_cap: Optional[float] = None,
    leader_to_member_value: Optional[float] = None,
    member_to_leader_value: Optional[float] = None,
    strength: Optional[float] = None,
    n_communities: Optional[int] = None,
    intra_strength: Optional[float] = None,
    inter_strength: Optional[float] = None,
    n_periphery: Optional[int] = None,
    core_to_periph: Optional[float] = None,
    periph_to_core: Optional[float] = None,
    periph_to_periph: Optional[float] = None,
    adaptive_intimacy: bool = False,
    kappa: Optional[float] = None,
    decay: Optional[float] = None,
    min_w: Optional[float] = None,
    max_w: Optional[float] = None,
    dampening: float = 0.08,
    condition_name: Optional[str] = None,
    output_root: Optional[str | Path] = None,
    output_subdir: Optional[str] = DEFAULT_OUTPUT_SUBDIR,
    create_subfolders: bool = True,
    add_timestamp_to_output_dir: bool = True,
    separate_condition_folders: bool = True,
    separate_run_folders: bool = False,
    save_full_results: bool = True,
    save_run_metadata: bool = True,
    save_summary: bool = True,
    summary_filename: str = "summary.csv",
    results_filename_template: str = "simulation_result_run_{run_id}.pkl",
    metadata_filename_template: str = "simulation_result_run_{run_id}_metadata.json",
) -> Dict[str, Any]:
    _validate_seeds(seeds)

    if "{run_id}" not in results_filename_template:
        raise ValueError("results_filename_template must contain the placeholder {run_id}.")
    if "{run_id}" not in metadata_filename_template:
        raise ValueError("metadata_filename_template must contain the placeholder {run_id}.")

    provided_modes = sum([
        conditions is not None,
        condition_grid is not None,
        any(value is not None for value in [population_size, structure, leader_style, max_iterations]),
    ])

    if provided_modes == 0:
        raise ValueError("Provide either direct single-condition arguments, conditions, or condition_grid.")
    if conditions is not None and condition_grid is not None:
        raise ValueError("Provide either conditions or condition_grid, not both.")

    if condition_grid is not None:
        conditions_to_run = build_conditions_from_grid(
            grid=condition_grid,
            fixed_params=fixed_params,
            condition_name_keys=condition_name_keys,
        )
    elif conditions is not None:
        _validate_conditions(conditions)
        conditions_to_run = [dict(condition) for condition in conditions]
    else:
        conditions_to_run = [
            _build_single_condition_from_arguments(
                population_size=population_size,
                structure=structure,
                leader_style=leader_style,
                max_iterations=max_iterations,
                min_weight=min_weight,
                include_leader_ties=include_leader_ties,
                leader_to_member_cap=leader_to_member_cap,
                member_to_leader_cap=member_to_leader_cap,
                leader_to_member_value=leader_to_member_value,
                member_to_leader_value=member_to_leader_value,
                strength=strength,
                n_communities=n_communities,
                intra_strength=intra_strength,
                inter_strength=inter_strength,
                n_periphery=n_periphery,
                core_to_periph=core_to_periph,
                periph_to_core=periph_to_core,
                periph_to_periph=periph_to_periph,
                adaptive_intimacy=adaptive_intimacy,
                kappa=kappa,
                decay=decay,
                min_w=min_w,
                max_w=max_w,
                dampening=dampening,
                condition_name=condition_name,
            )
        ]

    _validate_conditions(conditions_to_run)

    output_dir = prepare_output_directory(
        output_root=output_root,
        output_subdir=output_subdir,
        create_subfolders=create_subfolders,
        add_timestamp_to_output_dir=add_timestamp_to_output_dir,
    )

    all_results = []
    summary_rows = []
    global_run_counter = 0

    for condition_index, condition in enumerate(conditions_to_run, start=1):
        current_condition_name = condition.get("condition_name", f"condition_{condition_index}")

        for seed in seeds:
            global_run_counter += 1
            run_id = global_run_counter

            results = run_simulation(
                seed=seed,
                run_id=run_id,
                population_size=condition["population_size"],
                structure=condition["structure"],
                leader_style=condition["leader_style"],
                max_iterations=condition["max_iterations"],
                min_weight=condition.get("min_weight", 0.01),
                include_leader_ties=condition.get("include_leader_ties", True),
                leader_to_member_cap=condition.get("leader_to_member_cap"),
                member_to_leader_cap=condition.get("member_to_leader_cap"),
                leader_to_member_value=condition.get("leader_to_member_value"),
                member_to_leader_value=condition.get("member_to_leader_value"),
                strength=condition.get("strength"),
                n_communities=condition.get("n_communities"),
                intra_strength=condition.get("intra_strength"),
                inter_strength=condition.get("inter_strength"),
                n_periphery=condition.get("n_periphery"),
                core_to_periph=condition.get("core_to_periph"),
                periph_to_core=condition.get("periph_to_core"),
                periph_to_periph=condition.get("periph_to_periph"),
                adaptive_intimacy=condition.get("adaptive_intimacy", False),
                kappa=condition.get("kappa"),
                decay=condition.get("decay"),
                min_w=condition.get("min_w"),
                max_w=condition.get("max_w"),
                dampening=condition.get("dampening", 0.08),
                condition_name=current_condition_name,
            )

            all_results.append(results)
            summary_rows.append(_results_to_summary_row(results, condition_name=current_condition_name, condition_index=condition_index))

            if save_full_results or save_run_metadata:
                run_output_dir = _make_run_output_dir(
                    base_output_dir=output_dir,
                    condition_name=current_condition_name,
                    run_id=run_id,
                    separate_condition_folders=separate_condition_folders,
                    separate_run_folders=separate_run_folders,
                )

                if save_full_results:
                    results_path = run_output_dir / results_filename_template.format(run_id=run_id)
                    save_simulation_result(results, results_path)

                if save_run_metadata:
                    metadata_path = run_output_dir / metadata_filename_template.format(run_id=run_id)
                    run_metadata = _build_run_metadata(
                        results=results,
                        condition=condition,
                        condition_index=condition_index,
                        condition_name=current_condition_name,
                    )
                    save_run_metadata_json(run_metadata, metadata_path)

    summary_df = None
    summary_path = None

    if save_summary:
        summary_path = output_dir / summary_filename
        summary_df = save_summary_table(summary_rows, summary_path)

    return {
        "results": all_results,
        "summary_df": summary_df,
        "output_dir": output_dir,
        "summary_path": summary_path,
        "conditions_used": conditions_to_run,
    }