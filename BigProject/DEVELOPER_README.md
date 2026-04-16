# Developer README

This README is for people who want to modify, extend, or maintain the codebase.

The user-facing `README.md` explains how to run the project. This file explains how the project is organized internally, where to make changes, and what still needs attention.

## Architecture at a glance

The project is currently organized around these responsibilities:

- `agents.py`: Creates the initial agent pool and configures the chosen leader in place.
- `simulation_state.py`: Defines the simulation state container.
- `network.py`: Builds the base directed intimacy matrix over all agents.
- `build_simulation.py`: Assembles the initial state at time 0.
- `member_dynamics.py`: Handles member-to-member contagion and adaptive intimacy updates.
- `leader_intervention.py`: Handles leader intervention logic only.
- `run_simulation.py`: Runs one simulation.
- `run_multiple_simulations.py`: Runs many simulations across seeds and/or conditions.
- `metrics.py`: Plotting and lightweight output helpers, currently including sentiment evolution plots.
- `run_from_config.py`: Entry script that reads YAML and dispatches runs.
- `run_from_config.ipynb`: Notebook front-end for config-driven execution.
- `default.yaml`: Example config file for single/manual/grid execution modes.

## Current modeling assumptions

These are the assumptions the current modular structure is built around:

1. All agents, including the leader, live in one shared `agents` list.
2. The leader is identified by `leader_index`.
3. There is one unified `intimacy_matrix` over all agents.
4. Member contagion is currently member-only.
5. Leader intervention is separate from member contagion.
6. RL is intentionally excluded for now and should later plug into the leader intervention layer rather than into the core member dynamics.

## Where to edit what

### If you want to change how agents are initialized
Check:
- `agents.py`
- `build_simulation.py`

Typical examples:
- change initial emotion distribution
- change member parameter ranges
- change how leader style is encoded
- add new agent fields

### If you want to change how the leader is represented
Check:
- `agents.py`
- `simulation_state.py`
- `build_simulation.py`
- `leader_intervention.py`

Typical examples:
- keep more member-like fields on the leader
- change style definitions
- add leader-specific parameters
- change how the leader is identified

### If you want to change the intimacy matrix or network structure
Check:
- `network.py`
- `build_simulation.py`
- `member_dynamics.py`
- `leader_intervention.py`

Typical examples:
- add a new network structure
- change community/random/core-periphery assignment logic
- change normalization rules
- change leader-to-member tie initialization
- decide whether leader-related ties adapt over time

### If you want to change follower/member contagion
Check:
- `member_dynamics.py`

Typical examples:
- change the Bosse-style update
- change how interactions are sampled
- change whether contagion includes the leader
- change how absorption is tracked
- change adaptive intimacy behavior

### If you want to change leader intervention
Check:
- `leader_intervention.py`
- `run_simulation.py`

Typical examples:
- change when intervention triggers
- change what intervention does
- add target selection
- add different intervention mechanisms by style
- later connect RL output to leader actions

### If you want to change what gets saved
Check:
- `run_simulation.py`
- `run_multiple_simulations.py`
- `metrics.py`

Typical examples:
- add new histories or summaries
- save more metadata
- change folder structure
- add new plots

### If you want to change batch execution behavior
Check:
- `run_multiple_simulations.py`
- `run_from_config.py`
- `default.yaml`

Typical examples:
- add new config modes
- change grid behavior
- change filename templates
- add extra output options

## Known TODOs and notes

### 1. Leader handling inside non-random network structures needs more intentional treatment
Right now the leader is included in the same full matrix as everyone else, and the base structure generator may assign the leader to a community or to the core/periphery randomly.

This is acceptable for now, but it is a modeling choice that should likely be revisited.

Questions to decide later:
- In `community`, should the leader belong to one community, bridge communities, or be treated separately?
- In `core_periphery`, should the leader always be forced into the core?
- Should leader-related ties be generated separately from the follower-follower structure instead of being overwritten after base generation?
- Should leader-to-member and member-to-leader ties remain asymmetric by default?

Relevant files:
- `network.py`
- `build_simulation.py`

### 2. Decide whether leader-related ties should adapt over time
Current adaptive intimacy updates in `member_dynamics.py` only modify member-member ties.

This preserves the current assumption that leader-member dynamics are different from member-member contagion, but it should be made explicit whether:
- leader-to-member ties should remain fixed,
- member-to-leader ties should remain fixed,
- or either/both should adapt over time.

Relevant files:
- `member_dynamics.py`
- `leader_intervention.py`

### 3. Clarify the exact meaning of the reduced leader styles
The current non-RL code supports:
- `No_Intervention`
- `High_Initially_Constrained`
- `Low_Initially_Constrained`

At the moment, these mainly differ by threshold behavior and leader configuration. If the “initially constrained” aspect is supposed to carry time-based or phase-based meaning beyond the threshold itself, that still needs to be implemented explicitly.

Relevant files:
- `agents.py`
- `leader_intervention.py`

### 4. RL should become its own module and plug into the intervention layer
The intended future design is:
- core simulation remains usable without RL
- RL becomes optional
- RL selects or shapes leader intervention rather than replacing the whole engine

Suggested future structure:
- `rl/`
  - `policy.py`
  - `state.py`
  - `training.py`
  - `integration.py`

Integration target:
- `leader_intervention.py`

### 5. Some notebook-era plotting and analysis functions are not yet modularized
The sentiment evolution plot has been moved into `metrics.py`, but not every old notebook-specific graph or statistical analysis has been ported.

Potential future additions:
- social network graph creation
- richer post-run summary plots
- convenience loaders for saved batch outputs
- analysis-side utilities for comparing conditions

Relevant files:
- `metrics.py`
- analysis notebook(s)

### 6. `requirements.txt` should be cleaned up
The uploaded requirements file had at least one obvious typo and may not perfectly match the current modularized codebase.

This should be checked before sharing or packaging the project.

### 7. Consider whether a dedicated `SimulationResults` module is worth adding
Right now the results container is defined inside `run_simulation.py`. That is fine for now, but if the results object grows much more, it may be worth moving it into its own file such as:
- `simulation_results.py`

## Extension advice

### Adding a new network structure
1. Add the structure logic in `network.py`.
2. Update validation in `build_simulation.py`.
3. Decide how leader-related ties should behave for that structure.
4. Test that row normalization still behaves as expected.

### Adding a new leader style
1. Add or update the style in `agents.py`.
2. Update validation and logic in `leader_intervention.py`.
3. Decide whether the style changes only thresholds or also intervention behavior.
4. Add YAML examples if users should be able to run it directly.

### Adding a new saved output
1. Add tracking in `run_simulation.py`.
2. Add summary extraction in `run_multiple_simulations.py`.
3. Add optional plotting or loading support in `metrics.py` if useful.

### Adding RL later
1. Keep member contagion untouched.
2. Keep build/init untouched unless RL needs extra state.
3. Add RL state/action logic in a separate module.
4. Connect RL output to leader intervention decisions.
5. Make sure config-driven execution still works with RL disabled.

## Recommended maintenance approach

When making changes:
1. Change one modeling assumption at a time.
2. Keep the single-run path working before expanding batch logic.
3. Keep user-facing config names simple, even if internal code becomes more detailed.
4. Avoid mixing plotting or analysis logic back into the engine files.
5. Prefer explicit inputs/returns over hidden shared state.

## Suggested future cleanup priorities

A reasonable future order would be:
1. Confirm leader treatment in community/core-periphery structures.
2. Decide whether leader-related intimacy should adapt.
3. Port any remaining important notebook analyses/graphs.
4. Add the RL module as an optional layer.
5. Clean requirements and package-level imports.
