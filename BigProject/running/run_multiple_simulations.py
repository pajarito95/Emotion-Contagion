"""
Batch runner for the emotion contagion ABM.

Features:
- Run multiple simulations across one or many conditions
- Build conditions manually or automatically from parameter grids
- Save full SimulationResults objects as pickle files
- Save readable JSON metadata files alongside pickles
- Save a compact batch summary CSV
- Save learned RL Q-tables
- Save human-readable note.txt describing the batch
- Optional timestamped batch folders
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Sequence

import json
import pickle
from matplotlib import lines
import pandas as pd
import numpy as np

from run_simulation import run_simulation


DEFAULT_OUTPUT_SUBDIR = "outputs"

# OUTPUT DIRECTORY HELPERS
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
    add_timestamp_to_output_dir: bool = True,
) -> Path:

    root = resolve_output_root(output_root)

    if create_subfolders and output_subdir:
        base_dir = root / output_subdir
    else:
        base_dir = root

    if add_timestamp_to_output_dir:
        outdir = base_dir / make_timestamp_string()
    else:
        outdir = base_dir

    outdir.mkdir(parents=True, exist_ok=True)

    return outdir

# VALIDATION
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
            raise TypeError(f"conditions[{i}] must be a dictionary, but received {type(condition).__name__}." )

        required = {"population_size", "structure", "leader_style", "max_iterations"}
        missing = required - set(condition.keys())
        if missing:
            raise KeyError(f"conditions[{i}] is missing required key(s): {sorted(missing)}.")

# CONDITION BUILDERS
def build_conditions_from_grid(
    grid: Dict[str, Sequence[Any]],
    fixed_params: Optional[Dict[str, Any]] = None,
    condition_name_keys: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:

    if not isinstance(grid, dict) or len(grid) == 0:
        raise ValueError("grid must be a non-empty dictionary.")

    fixed_params = dict(fixed_params) if fixed_params else {}

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

        condition["condition_name"] = "__".join(name_parts)
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
    use_rl_leader: bool,
    rl_alpha: float,
    rl_gamma: float,
    rl_epsilon_start: float,
    rl_epsilon_end: float,
    rl_epsilon_decay_steps: int,
    rl_actions: Sequence[int],
    condition_name: Optional[str],
) -> Dict[str, Any]:

    required = {"population_size": population_size, "structure": structure, "leader_style": leader_style, "max_iterations": max_iterations}
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"When conditions is not provided, these are required: {missing}")

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

        "use_rl_leader": use_rl_leader,
        "rl_alpha": rl_alpha,
        "rl_gamma": rl_gamma,
        "rl_epsilon_start": rl_epsilon_start,
        "rl_epsilon_end": rl_epsilon_end,
        "rl_epsilon_decay_steps": rl_epsilon_decay_steps,
        "rl_actions": list(rl_actions),
    }

# SAVE HELPERS
def save_simulation_result(results, filepath: Path) -> None:
    with filepath.open("wb") as f:
        pickle.dump(results, f)

def save_run_metadata_json(metadata: Dict[str, Any], filepath: Path) -> None:
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

def save_summary_table(summary_rows: List[Dict[str, Any]], filepath: Path,) -> pd.DataFrame:
    df = pd.DataFrame(summary_rows)
    df.to_csv(filepath, index=False)
    return df

def save_q_table(q_table: np.ndarray, filepath: Path) -> None:
    with filepath.open("wb") as f:
        pickle.dump(q_table, f)

# SUMMARY HELPERS
def _results_to_summary_row(results, condition_name: Optional[str] = None, condition_index: Optional[int] = None) -> Dict[str, Any]:
    final_avg_emotion = None
    if results.avg_emotion_history:
        final_avg_emotion = results.avg_emotion_history[-1]

    final_member_emotions = [agent["emotion"] for agent in results.state.agents if agent.get("role") == "member" ]

    return {
        "condition_index": condition_index,
        "condition_name": condition_name,

        "run_id": results.run_id,
        "seed": results.seed,

        "structure": results.metadata.get("structure"),
        "leader_style": results.metadata.get("leader_style"),

        "population_size": results.metadata.get("population_size"),
        "max_iterations": results.metadata.get("max_iterations"),

        "adaptive_intimacy": results.metadata.get("adaptive_intimacy"),
        "use_rl_leader": results.metadata.get("use_rl_leader"),

        "num_interventions": len(results.intervention_timesteps),

        "final_avg_member_emotion": final_avg_emotion,
        "final_min_member_emotion": min(final_member_emotions) if final_member_emotions else None,
        "final_max_member_emotion": max(final_member_emotions) if final_member_emotions else None,

        "mean_rl_reward": float(np.mean(results.rl_rewards)) if results.rl_rewards else None,
        "mean_rl_quality": float(np.mean(results.rl_quality)) if results.rl_quality else None,
    }

def _build_run_metadata(
    results,
    condition: Dict[str, Any],
    condition_index: int,
    condition_name: Optional[str]
) -> Dict[str, Any]:

    return {
        "run_id": results.run_id,
        "seed": results.seed,

        "condition_index": condition_index,
        "condition_name": condition_name,

        "condition": condition,

        "structure": results.metadata.get("structure"),
        "leader_style": results.metadata.get("leader_style"),

        "population_size": results.metadata.get("population_size"),
        "max_iterations": results.metadata.get("max_iterations"),

        "adaptive_intimacy": results.metadata.get("adaptive_intimacy"),

        "use_rl_leader": results.metadata.get("use_rl_leader"),

        "num_interventions": len(results.intervention_timesteps),
        "intervention_timesteps": results.intervention_timesteps,
    }

# NOTE.TXT
def _format_condition_note(condition: Dict[str, Any], index: int) -> List[str]:
    lines: List[str] = []

    
    population_size = condition.get("population_size")
    n_members = population_size - 1 if isinstance(population_size, int) else "unknown"
    lines.append(f"Simulation length: {condition.get('max_iterations')}")
    
    lines.append(f"Condition {index}")
    lines.append("-" * 40)
    lines.append(f"Condition name: {condition.get('condition_name')}")
    lines.append(f"Leader style: {condition.get('leader_style')}")
    lines.append(f"Network structure: {condition.get('structure')}")

    lines.append(f"Population size: {population_size}")
    lines.append("Number of leaders: 1")
    lines.append(f"Number of members: {n_members}")

    lines.append(f"Adaptive intimacy: {condition.get('adaptive_intimacy')}")
    lines.append(f"RL enabled: {condition.get('use_rl_leader')}")

    structure = condition.get("structure")

    if structure == "random":
        lines.append("Structure details:")
        lines.append(f"  Random tie strength: {condition.get('strength')}")

    elif structure == "community":
        lines.append("Structure details:")
        lines.append(f"  Number of communities: {condition.get('n_communities')}")
        lines.append(f"  Intra-community strength: {condition.get('intra_strength')}")
        lines.append(f"  Inter-community strength: {condition.get('inter_strength')}")

    elif structure == "core_periphery":
        lines.append("Structure details:")
        lines.append(f"  Number of peripheral members: {condition.get('n_periphery')}")
        lines.append(f"  Core-to-periphery strength: {condition.get('core_to_periph')}")
        lines.append(f"  Periphery-to-core strength: {condition.get('periph_to_core')}")
        lines.append(f"  Periphery-to-periphery strength: {condition.get('periph_to_periph')}")

    if condition.get("use_rl_leader", False):
        lines.append("RL details:")
        lines.append(f"  RL actions: {condition.get('rl_actions')}")
        lines.append(f"  alpha: {condition.get('rl_alpha')}")
        lines.append(f"  gamma: {condition.get('rl_gamma')}")
        lines.append(f"  epsilon_start: {condition.get('rl_epsilon_start')}")
        lines.append(f"  epsilon_end: {condition.get('rl_epsilon_end')}")
        lines.append(f"  epsilon_decay_steps: {condition.get('rl_epsilon_decay_steps')}")

    lines.append("")
    return lines

def write_batch_note(output_dir: Path, conditions_to_run: Sequence[Dict[str, Any]], seeds: Sequence[int]) -> None:
    lines: List[str] = []

    lines.append("Emotion Contagion ABM Batch Note")
    lines.append("=" * 60)
    lines.append(f"Batch folder: {output_dir}")
    lines.append(f"Number of conditions: {len(conditions_to_run)}")
    lines.append(f"Number of repetitions: {len(seeds)}")
    lines.append(f"Seeds: {list(seeds)}")
    lines.append("")

    if conditions_to_run:
        population_sizes = sorted({c.get("population_size") for c in conditions_to_run})
        structures = sorted({c.get("structure") for c in conditions_to_run})
        leader_styles = sorted({c.get("leader_style") for c in conditions_to_run})
        max_iterations = sorted({c.get("max_iterations") for c in conditions_to_run})

        lines.append("Batch Summary")
        lines.append("-" * 60)
        lines.append(f"Population sizes tested: {population_sizes}")
        lines.append(f"Network structures tested: {structures}")
        lines.append(f"Leader styles tested: {leader_styles}")
        lines.append(f"Simulation lengths tested: {max_iterations}")

        lines.append("")

        lines.append("Condition Details")
        lines.append("-" * 60)

        for i, condition in enumerate(conditions_to_run, start=1):
            lines.extend(_format_condition_note(condition, i))

    note_path = output_dir / "note.txt"

    note_path.write_text("\n".join(lines), encoding="utf-8")

# RUN OUTPUT FOLDERS
def _make_run_output_dir(
    base_output_dir: Path,
    condition_name: Optional[str],
    run_id: Any,
    separate_condition_folders: bool,
    separate_run_folders: bool,
) -> Path:

    outdir = base_output_dir
    if separate_condition_folders:
        safe_condition_name = (condition_name if condition_name else "unnamed_condition")
        outdir = outdir / safe_condition_name
        outdir.mkdir(parents=True, exist_ok=True)

    if separate_run_folders:
        outdir = outdir / f"run_{run_id}"
        outdir.mkdir(parents=True, exist_ok=True)

    return outdir

# MAIN BATCH RUNNER
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

    use_rl_leader: bool = True,
    rl_alpha: float = 0.1,
    rl_gamma: float = 0.95,
    rl_epsilon_start: float = 0.3,
    rl_epsilon_end: float = 0.05,
    rl_epsilon_decay_steps: int = 2000,
    rl_actions: Sequence[int] = (0, 1, 2, 3),

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
    save_q_tables: bool = True,
    summary_filename: str = "summary.csv",
    results_filename_template: str = "simulation_result_run_{run_id}.pkl",
    metadata_filename_template: str = "simulation_result_run_{run_id}_metadata.json",
    qtable_filename_template:  str = "simulation_result_run_{run_id}_qtable.pkl"
) -> Dict[str, Any]:

    _validate_seeds(seeds)

    provided_modes = sum([conditions is not None, condition_grid is not None, any(v is not None for v in [population_size, structure, leader_style, max_iterations])])
    if provided_modes == 0:
        raise ValueError("Provide conditions, condition_grid, or direct single-condition arguments.")

    if conditions is not None and condition_grid is not None:
        raise ValueError("Provide either conditions or condition_grid, not both.")

    # BUILD CONDITIONS
    if condition_grid is not None:
        conditions_to_run = build_conditions_from_grid(grid=condition_grid, fixed_params=fixed_params, condition_name_keys=condition_name_keys)

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

                use_rl_leader=use_rl_leader,
                rl_alpha=rl_alpha,
                rl_gamma=rl_gamma,
                rl_epsilon_start=rl_epsilon_start,
                rl_epsilon_end=rl_epsilon_end,
                rl_epsilon_decay_steps=rl_epsilon_decay_steps,
                rl_actions=rl_actions,

                condition_name=condition_name,
            )
        ]

    _validate_conditions(conditions_to_run)

    # OUTPUT DIRECTORY
    output_dir = prepare_output_directory( output_root=output_root, output_subdir=output_subdir, create_subfolders=create_subfolders, add_timestamp_to_output_dir=add_timestamp_to_output_dir)
    write_batch_note(output_dir=output_dir, conditions_to_run=conditions_to_run, seeds=seeds)

    # RUN
    all_results = []
    summary_rows = []

    global_run_counter = 0
    print(conditions_to_run[0])
    for c in conditions_to_run:
        print(c["leader_style"], c.get("use_rl_leader"))
    for condition_index, condition in enumerate(conditions_to_run, start=1):
        current_condition_name = condition.get(f"condition_name: condition_{condition_index}")

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
                member_to_leader_cap=condition.get( "member_to_leader_cap"),
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

                use_rl_leader=condition.get("use_rl_leader", False),
                rl_alpha=condition.get("rl_alpha", 0.1),
                rl_gamma=condition.get("rl_gamma", 0.95),
                rl_epsilon_start=condition.get("rl_epsilon_start", 0.3),
                rl_epsilon_end=condition.get("rl_epsilon_end", 0.05),
                rl_epsilon_decay_steps=condition.get("rl_epsilon_decay_steps", 2000),
                rl_actions=condition.get("rl_actions", [0, 1, 2, 3]),

                condition_name=current_condition_name
            )

            all_results.append(results)
            summary_rows.append(_results_to_summary_row(results=results, condition_name=current_condition_name, condition_index=condition_index))

            # SAVE
            if save_full_results or save_run_metadata or save_q_tables:
                run_output_dir = _make_run_output_dir(base_output_dir=output_dir, condition_name=current_condition_name, run_id=run_id, separate_condition_folders=separate_condition_folders, separate_run_folders=separate_run_folders)

                if save_full_results:
                    results_path = (run_output_dir / results_filename_template.format(run_id=run_id))
                    save_simulation_result(results, results_path)

                if save_run_metadata:
                    metadata_path = (run_output_dir / metadata_filename_template.format(run_id=run_id))
                    run_metadata = _build_run_metadata(results=results, condition=condition, condition_index=condition_index, condition_name=current_condition_name)
                    save_run_metadata_json(run_metadata, metadata_path)

                if (save_q_tables and results.q_table is not None):
                    qtable_path = (run_output_dir / qtable_filename_template.format(run_id=run_id))
                    save_q_table(results.q_table, qtable_path)

    # SUMMARY CSV
    summary_df = None
    summary_path = None
    if save_summary:
        summary_path = output_dir / summary_filename
        summary_df = save_summary_table(summary_rows=summary_rows, filepath=summary_path)

    # RETURN
    return {
        "results": all_results,
        "summary_df": summary_df,
        "summary_path": summary_path,
        "output_dir": output_dir,
        "conditions_used": conditions_to_run,
    }