"""
Batch runner for the emotion contagion ABM.
- Run multiple simulations across seeds
- Organize outputs cleanly
- Allow users to specify an output parent path
- Fall back to the current working directory when no path is provided
- Optionally save full results and a compact summary table

Current design assumptions:
- Single-run logic lives in run_simulation.py
- This file handles repeated runs and output organization
- RL is not included here either (yet)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
import pickle
from datetime import datetime

import pandas as pd

from run_simulation import run_simulation

def resolve_output_root(output_root: Optional[str | Path] = None) -> Path:
    """
    Resolve the main parent output path. If output_root is None, the current working directory is used.

    Parameters
    ----------
    output_root : str | Path | None, optional
        User-provided parent path

    Returns
    -------
    Path
        Resolved absolute path
    """
    if output_root is None:
        return Path.cwd().resolve()

    return Path(output_root).expanduser().resolve()


def prepare_output_directory(
    output_root: Optional[str | Path] = None,
    output_subdir: Optional[str] = "outputs",
    create_subfolders: bool = True,
) -> Path:
    """
    Prepare the directory where batch outputs will be written.
    - If output_root is None, use the current working directory.
    - If output_subdir is provided and create_subfolders is True, create that subdirectory under output_root.
    - If output_subdir is None or create_subfolders is False, save directly under output_root.

    Parameters
    ----------
    output_root : str | Path | None, optional
        User-chosen parent path
    output_subdir : str | None, optional
        Name of subdirectory to create under the parent path
    create_subfolders : bool, optional
        Whether to create and use the subdirectory

    Returns
    -------
    Path
        Final output directory
    """
    root = resolve_output_root(output_root)

    if create_subfolders and output_subdir:
        outdir = root / output_subdir
    else:
        outdir = root

    outdir.mkdir(parents=True, exist_ok=True)
    
    return outdir


def _validate_seeds(seeds: Sequence[int]) -> None:
    """
    Validate the list of seeds.
    """
    if not isinstance(seeds, Sequence) or len(seeds) == 0:
        raise ValueError("seeds must be a non-empty sequence of integers.")

    for seed in seeds:
        if not isinstance(seed, int):
            raise TypeError(f"Each seed must be an integer, but received {type(seed).__name__}.")


def _make_run_output_dir(base_output_dir: Path, run_id: Any, separate_run_folders: bool) -> Path:
    """
    Create a per-run output folder when requested.
    """
    if separate_run_folders:
        run_dir = base_output_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    return base_output_dir


def _results_to_summary_row(results, condition_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert one SimulationResults object into a compact summary row.
    """
    final_avg_emotion = None
    if results.avg_emotion_history:
        final_avg_emotion = results.avg_emotion_history[-1]

    final_member_emotions = [
        agent["emotion"]
        for agent in results.state.agents
        if agent.get("role") == "member"
    ]

    leader = results.state.agents[results.state.leader_index]

    row = {
        "run_id": results.run_id,
        "seed": results.seed,
        "condition_name": condition_name,
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
            if results.interactions_per_timestep else 0.0
        ),
        "final_avg_member_emotion": final_avg_emotion,
        "final_min_member_emotion": min(final_member_emotions) if final_member_emotions else None,
        "final_max_member_emotion": max(final_member_emotions) if final_member_emotions else None,
        "leader_emotion": leader.get("emotion"),
        "leader_threshold": leader.get("interventionThreshold"),
        "leader_emotion_management_ability": leader.get("emotionManagementAbility"),
    }

    return row


def save_simulation_result(results, filepath: Path) -> None:
    """
    Save one full SimulationResults object to disk using pickle.
    """
    with filepath.open("wb") as f:
        pickle.dump(results, f)


def save_summary_table(summary_rows: List[Dict[str, Any]], filepath: Path) -> pd.DataFrame:
    """
    Save the summary rows as a CSV file and return the DataFrame.
    """
    df = pd.DataFrame(summary_rows)
    df.to_csv(filepath, index=False)
    return df

def save_run_metadata_file(results, filepath: Path) -> None:
    """
    Save compact metadata for on simulation run as json
    """
    metadata = {
        "run_id": results.run_id,
        "seed": results.seed,
        "metadata": results.metadata,
        "leader_summary": results.leader_summary,
        "num_interventions": len(results.interventions_timesteps),
        "intervention_timesteps": results.intervention_timesteps,
        "total_interactions": sum(results.interactions_per_timestep),
    }

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

def run_multiple_simulations(
    seeds: Sequence[int],
    population_size: int,
    structure: str,
    leader_style: str,
    max_iterations: int,
    intra_strength: float = 0.5,
    inter_strength: float = 0.2,
    community_size: Optional[tuple[int, int]] = None,
    min_weight: float = 0.01,
    core_to_core: float = 0.65,
    core_to_periph: float = 0.5,
    periph_to_core: float = 0.2,
    periph_to_periph: float = 0.2,
    core_proportion: float = 0.25,
    include_leader_ties: bool = True,
    leader_to_member_cap: Optional[float] = None,
    member_to_leader_cap: Optional[float] = None,
    leader_to_member_value: Optional[float] = None,
    member_to_leader_value: Optional[float] = None,
    adaptive_intimacy: bool = False,
    kappa: Optional[float] = None,
    decay: Optional[float] = None,
    min_w: Optional[float] = None,
    max_w: Optional[float] = None,
    dampening: float = 0.08,
    condition_name: Optional[str] = None,
    output_root: Optional[str | Path] = None,
    output_subdir: Optional[str] = "outputs",
    create_subfolders: bool = True,
    add_timestamps_to_output_dir: bool = False,
    separate_condition_folders: bool = False,
    separate_run_folders: bool = False,
    save_full_results: bool = True,
    save_summary: bool = True,
    save_run_metadata: bool = True,
    summary_filename: str = "summary.csv",
    results_filename_template: str = "simulation_result_run_{run_id}.pkl",
    metadata_filename_template: str = "simulation_result_run_{run_id}_metadata.json",
) -> Dict[str, Any]:
    """
    Run multiple simulations over a sequence of seeds.

    Parameters
    ----------
    seeds : sequence[int]
        Random seeds to use for repeated runs
    population_size : int
        Total number of agents including the leader
    structure : str
        Network structure
    leader_style : str
        Leader style
    max_iterations : int
        Number of timesteps per run

    Remaining simulation parameters match run_simulation(...)

    condition_name : str | None, optional
        Optional label for the batch condition
    output_root : str | Path | None, optional
        Parent output path chosen by the user. If None, the current working directory is used
    output_subdir : str | None, optional
        Subdirectory name under the parent output path
    create_subfolders : bool, optional
        Whether to create and use output_subdir
    separate_run_folders : bool, optional
        Whether to create a separate folder for each run
    save_full_results : bool, optional
        Whether to save each full SimulationResults object as a pickle file
    save_summary : bool, optional
        Whether to save a compact summary CSV
    summary_filename : str, optional
        Name of the summary CSV file
    results_filename_template : str, optional
        Filename template for pickled run outputs. Must contain {run_id}

    Returns
    -------
    dict
        Dictionary containing:
        - "results": list of full SimulationResults objects
        - "summary_df": summary DataFrame or None
        - "output_dir": resolved output directory
        - "summary_path": summary file path or None
    """
    _validate_seeds(seeds)

    if "{run_id}" not in results_filename_template:
        raise ValueError("results_filename_template must contain the placeholder {run_id}.")

    if save_run_metadata and "{run_id}" not in metadata_filename_template:
        raise ValueError("metadata_filename_template must contain the placeholder {run_id}.")

    output_dir = prepare_output_directory(
        output_root=output_root,
        output_subdir=output_subdir,
        create_subfolders=create_subfolders,
    )

    if add_timestamps_to_output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

    if separate_condition_folders and condition_name:
        output_dir = output_dir / condition_name
        output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary_rows = []

    for run_number, seed in enumerate(seeds, start=1):
        run_id = run_number

        results = run_simulation(
            seed=seed,
            run_id=run_id,
            population_size=population_size,
            structure=structure,
            leader_style=leader_style,
            max_iterations=max_iterations,
            intra_strength=intra_strength,
            inter_strength=inter_strength,
            community_size=community_size,
            min_weight=min_weight,
            core_to_core=core_to_core,
            core_to_periph=core_to_periph,
            periph_to_core=periph_to_core,
            periph_to_periph=periph_to_periph,
            core_proportion=core_proportion,
            include_leader_ties=include_leader_ties,
            leader_to_member_cap=leader_to_member_cap,
            member_to_leader_cap=member_to_leader_cap,
            leader_to_member_value=leader_to_member_value,
            member_to_leader_value=member_to_leader_value,
            adaptive_intimacy=adaptive_intimacy,
            kappa=kappa,
            decay=decay,
            min_w=min_w,
            max_w=max_w,
            dampening=dampening,
        )

        all_results.append(results)
        summary_rows.append(_results_to_summary_row(results, condition_name=condition_name))

        run_output_dir = _make_run_output_dir(
            base_output_dir=output_dir,
            run_id=run_id,
            separate_run_folders=separate_run_folders,
            )
        
        if save_full_results:
            results_filename = results_filename_template.format(run_id=run_id)
            results_path = run_output_dir / results_filename
            save_simulation_result(results, results_path)

        if save_run_metadata:
            metadata_filename = metadata_filename_template.format(run_id=run_id)
            metadata_path = run_output_dir / metadata_filename
            save_run_metadata_file(results, metadata_path)

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
    }
