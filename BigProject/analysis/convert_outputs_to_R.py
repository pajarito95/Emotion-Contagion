"""
Convert saved simulation outputs into flat files that are easy to read in R.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pickle

import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
RUNNING_DIR = PROJECT_ROOT / "running"
SRC_DIR = PROJECT_ROOT / "src"

for path in [str(PROJECT_ROOT), str(RUNNING_DIR), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def export_emotion_history(results_list, output_path: Path) -> pd.DataFrame:
    rows = []

    for results in results_list:
        condition_name = results.metadata.get("condition_name")
        leader_index = results.state.leader_index

        for t, emotions_at_t in enumerate(results.emotion_history):
            for agent_index, emotion in enumerate(emotions_at_t):
                agent = results.state.agents[agent_index]

                rows.append(
                    {
                        "run_id": results.run_id,
                        "seed": results.seed,
                        "condition_name": condition_name,
                        "leader_style": results.metadata.get("leader_style"),
                        "structure": results.metadata.get("structure"),
                        "max_iterations": results.metadata.get("max_iterations"),
                        "leader_index": leader_index,
                        "time": t,
                        "agent_index": agent_index,
                        "role": agent.get("role"),
                        "emotion": emotion,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    return df


def export_avg_emotion_history(results_list, output_path: Path) -> pd.DataFrame:
    rows = []

    for results in results_list:
        condition_name = results.metadata.get("condition_name")

        for t, avg_emotion in enumerate(results.avg_emotion_history):
            rows.append(
                {
                    "run_id": results.run_id,
                    "seed": results.seed,
                    "condition_name": condition_name,
                    "leader_style": results.metadata.get("leader_style"),
                    "structure": results.metadata.get("structure"),
                    "time": t,
                    "avg_member_emotion": avg_emotion,
                }
            )

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    return df


def export_interventions(results_list, output_path: Path) -> pd.DataFrame:
    rows = []

    for results in results_list:
        condition_name = results.metadata.get("condition_name")

        for t in results.intervention_timesteps:
            rows.append(
                {
                    "run_id": results.run_id,
                    "seed": results.seed,
                    "condition_name": condition_name,
                    "leader_style": results.metadata.get("leader_style"),
                    "structure": results.metadata.get("structure"),
                    "intervention_time": t,
                }
            )

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    return df


def export_initial_conditions(results_list, output_path: Path) -> pd.DataFrame:
    frames = []

    for results in results_list:
        df = results.initial_conditions.copy()
        df["run_id"] = results.run_id
        df["seed"] = results.seed
        df["condition_name"] = results.metadata.get("condition_name")
        df["leader_style"] = results.metadata.get("leader_style")
        df["structure"] = results.metadata.get("structure")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_parquet(output_path, index=False)
    return out


def main(results_dir: str | Path, output_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pickle_paths = sorted(results_dir.rglob("simulation_result_run_*.pkl"))

    if not pickle_paths:
        raise FileNotFoundError(f"No result pickle files were found under: {results_dir}")

    results_list = [load_pickle(path) for path in pickle_paths]

    export_emotion_history(results_list, output_dir / "emotion_history.parquet")
    export_avg_emotion_history(results_list, output_dir / "avg_emotion_history.parquet")
    export_interventions(results_list, output_dir / "interventions.parquet")
    export_initial_conditions(results_list, output_dir / "initial_conditions.parquet")

    print(f"Loaded {len(results_list)} result files.")
    print(f"Exported files to: {output_dir}")


if __name__ == "__main__":
    main(
        results_dir="outputs",
        output_dir="outputs/for_r",
    )