"""
Lightweight plotting and summary helpers
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd


def plot_sentiment_evolution(results, save_path: Optional[str | Path] = None, show: bool = False):
    """
    Plot sentiment evolution for one run.

    Intended behavior:
    - plot each agent's emotion trajectory over time as grey lines
    - plot the average team/member emotion over time as a red line
    - plot leader intervention timesteps as blue vertical dashed lines
    - include leader style, network structure, seed, and run id in the title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    emotion_history = results.emotion_history
    n_timesteps = len(emotion_history)
    timesteps = list(range(n_timesteps))

    if n_timesteps == 0:
        raise ValueError("results.emotion_history is empty, so there is nothing to plot.")

    n_agents = len(emotion_history[0])

    # Plot individual member-only trajectories in grey
    member_indices = [i for i, agent in enumerate(results.state.agents) if agent.get("role") == "member"]
    for agent_index in member_indices:
        series = [emotion_history[t][agent_index] for t in timesteps]
        ax.plot(timesteps, series, color="grey", alpha=0.35, linewidth=1)

    # Plot average member emotion in red
    avg_series = results.avg_emotion_history
    avg_timesteps = list(range(len(avg_series)))
    ax.plot(avg_timesteps, avg_series, color="red", linewidth=2.5, label="Average member emotion")

    # Plot intervention timesteps as blue vertical dashed lines
    intervention_timesteps = results.intervention_timesteps or []
    first_line = True
    for t in intervention_timesteps:
        if first_line:
            ax.axvline(t, color="blue", linestyle="--", alpha=0.7, linewidth=1.5, label="Leader intervention")
            first_line = False
        else:
            ax.axvline(t, color="blue", linestyle="--", alpha=0.7, linewidth=1.5)

    leader_style = results.metadata.get("leader_style", "UnknownStyle")
    structure = results.metadata.get("structure", "UnknownStructure")
    seed = results.seed
    run_id = results.run_id

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Emotion")
    ax.set_title(f"Sentiment Evolution | Style: {leader_style} | Structure: {structure} | Seed: {seed} | Run: {run_id}")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def summary_dataframe_from_batch(batch) -> pd.DataFrame:
    if batch.get("summary_df") is not None:
        return batch["summary_df"]
    raise ValueError("No summary DataFrame was found in the batch output.")

# from __future__ import annotations
# import pickle
# from pathlib import Path
# from typing import Iterable
# import numpy as np
# import pandas as pd
# from simulation_state import AllSimulationResults

# STYLE_PRETTY = {
#     "High_Fully_Constrained": "High Fully",
#     "High_Initially_Constrained": "High Initial",
#     "Low_Fully_Constrained": "Low Fully",
#     "Low_Initially_Constrained": "Low Initial",
#     "Free": "Free",
#     "No_Intervention": "None",
# }

# def load_condition(base_dir: str | Path, date_str: str, team_size: int, network_structure: str, style: str) -> AllSimulationResults | None:
#     path = Path(base_dir) / date_str / f"team_size_{team_size}" / network_structure / style / "all_results.pkl"
#     if not path.exists():
#         print(f"Missing results file: {path}")
#         return None
    
#     with open(path, "rb") as f:
#         return pickle.load(f)

# def load_many_conditions(
#     base_dir: str | Path,
#     date_str: str,
#     team_size: int,
#     network_structures: Iterable[str],
#     styles: Iterable[str],
# ) -> dict[tuple[str, str], AllSimulationResults | None]:
#     results = {}
#     for network_structure in network_structures:
#         for style in styles:
#             results[(network_structure, style)] = load_condition(base_dir=base_dir, date_str=date_str, team_size=team_size, network_structure=network_structure, style=style)
#     return results

# def pad_runs(list_of_lists: list[list[float]]) -> np.ndarray:
#     if len(list_of_lists) == 0:
#         return np.empty((0, 0))
    
#     max_t = max(len(x) for x in list_of_lists if len(x) > 0)
#     arr = np.full((len(list_of_lists), max_t), np.nan)

#     for i, seq in enumerate(list_of_lists):
#         arr[i, : len(seq)] = seq

#     return arr

# def leader_score_table(results_dict: dict[tuple[str, str], AllSimulationResults | None]) -> pd.DataFrame:
#     rows = []
#     for (network, style), results in results_dict.items():
#         if results is None:
#             continue

#         for run in results.runs:
#             final_emotions = np.array(run.emotion_history[-1], dtype=float)
#             rows.append(
#                 {
#                     "Leader": style,
#                     "Network": network,
#                     "Score": float(np.mean(final_emotions) - np.std(final_emotions)),
#                     "Raw_Mean": float(np.mean(final_emotions)),
#                     "Raw_SD": float(np.std(final_emotions)),
#                     "Interventions": len(run.intervention_log),
#                     "Pretty_Leader": STYLE_PRETTY.get(style, style),
#                 }
#             )
#     return pd.DataFrame(rows)

# def temporal_variance_table(results_dict: dict[tuple[str, str], AllSimulationResults | None]) -> pd.DataFrame:
#     rows = []
#     for (network, style), results in results_dict.items():
#         if results is None:
#             continue

#         for run in results.runs:
#             for time_idx, agent_emotions in enumerate(run.emotion_history):
#                 rows.append(
#                     {
#                         "Leader": style,
#                         "Network": network,
#                         "Run": run.run_id,
#                         "Time": time_idx,
#                         "Variance": float(np.var(agent_emotions)),
#                     }
#                 )
#     return pd.DataFrame(rows)

# def summary_table(results_dict: dict[tuple[str, str], AllSimulationResults | None]) -> pd.DataFrame:
#     rows = []
#     for (network, style), results in results_dict.items():
#         if results is None:
#             continue

#         for run in results.runs:
#             final_emotions = np.array(run.emotion_history[-1], dtype=float)
#             rows.append(
#                 {
#                     "Leader": style,
#                     "Network": network,
#                     "Run": run.run_id,
#                     "Final_Avg_Emotion": float(np.mean(final_emotions)),
#                     "Final_Emotion_SD": float(np.std(final_emotions)),
#                     "Score": float(np.mean(final_emotions) - np.std(final_emotions)),
#                     "Interventions": len(run.intervention_log),
#                     "Mean_Homophily": float(np.mean(run.homophily_index)) if run.homophily_index else np.nan,
#                     "Mean_Quality": float(np.mean(run.rl_quality)) if run.rl_quality else np.nan,
#                 }
#             )
#     return pd.DataFrame(rows)