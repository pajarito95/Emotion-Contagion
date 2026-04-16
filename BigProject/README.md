# Emotion Contagion ABM

This project runs an emotion contagion agent-based model with:
- member-to-member emotional contagion
- leader intervention as a separate layer
- optional adaptive intimacy among members
- single runs or multi-condition experiment batches
- configuration-driven execution through a YAML file

Note that this version does not have the RL extension. That might get added in later as an optional layer.

## What is `__init__.py` for?

`__init__.py` tells Python that a folder should be treated as a package/module directory. It helps Python recognize that the files in that folder belong together and can be imported cleanly.
- makes imports more consistent
- helps keep the code organized as one project rather than as unrelated scripts
- gives you a place to expose package-level imports later if you want to

If a file is empty, that might still be okay as it may just be there for structure.

---

## Project layout

### Core simulation files

#### `agents.py`
Creates the initial population of agents and configures one chosen agent as the leader.

Go here if you want to change:
- how member agents are initialized
- the distributions used for initial agent attributes
- which parameters belong to members versus the leader
- how leader styles are configured at initialization

#### `simulation_state.py`
Defines the state container for the simulation.

Go here if you want to change:
- what is stored as part of the live simulation state
- how the leader is identified
- what metadata is carried alongside a run

#### `network.py`
Builds the initial intimacy matrix.

Go here if you want to change:
- how the community, random, or core-periphery structure is generated
- follower-follower tie generation
- structural assignments
- row-normalization behavior

#### `build_simulation.py`
Constructs the full initial simulation setup at time 0.

Go here if you want to change:
- how the initial state is assembled
- how the leader is selected
- whether leader-member ties are included
- how leader-member ties differ from member-member ties
- how initialization options are wired together

#### `member_dynamics.py`
Contains member-to-member contagion and adaptive member-member intimacy updates.

Go here if you want to change:
- how member emotions update during interactions
- how interacting member pairs are chosen
- how average member emotion is computed
- how adaptive intimacy is updated over time among members

#### `leader_intervention.py`
Contains the leader-specific intervention logic.

Go here if you want to change:
- when the leader intervenes
- how strong the intervention is
- how leader influence is applied to members
- style-dependent leader behavior

#### `run_simulation.py`
Runs one full simulation from initialization through the final timestep.

Go here if you want to change:
- the overall timestep order of events
- what gets logged each timestep
- what one run returns as output
- how single-run results are packaged

#### `run_multiple_simulations.py`
Runs multiple simulations across seeds and conditions.

Go here if you want to change:
- batch execution behavior
- condition handling
- parameter-grid expansion
- output folder creation
- saving of pickles, JSON metadata, and summary CSV files

#### `metrics.py`
Contains plotting and summary helpers for saved outputs.

Go here if you want to change:
- sentiment evolution plots
- plotting defaults
- additional summary or metric helpers

But you can also just do things in an analysis file/notebook using outputs from the simulation.

---

## Entry-point files

#### `default.yaml`
User-facing configuration file.

Go here if you want to change:
- number of agents
- structure type
- leader style
- number of timesteps
- adaptive intimacy settings
- output paths
- whether plots should be generated
- whether to run a single condition, manual list of conditions, or a grid of conditions

#### `run_from_config.py`
Runs the project from a YAML config file.

Use this if you want a script-based workflow from the command line.

#### `run_from_config.ipynb`
Notebook front end for running the project from the same YAML config file.

Use this if you prefer Jupyter and want a more interactive workflow.

#### `__init__.py`
Package marker file.

You usually do not need to edit this unless you later want to do things with imports at the package level.

---

## Output files

Depending on the config settings, runs may save:
- `.pkl` files containing the full run results
- `.json` files containing readable run metadata
- `.csv` summary tables for batches
- `.png` sentiment evolution plots

---

## Ways to run the project

You can run the project in three main ways:
- by editing the YAML file and using the Python script
- by editing the YAML file and using the notebook
- by importing the Python functions directly into your own code

### Option 1: Run from YAML using the Python script

1. Open `default.yaml`
2. Edit the settings you want
3. Run:

```bash
python run_from_config.py --config default.yaml
```

Perhaps the cleanest option for those who want to run experiments without editing the internal code files.

### Option 2: Run from YAML using the notebook

1. Open `run_from_config.ipynb`
2. Make sure `default.yaml` contains the settings you want
3. Run the notebook cells

Useful if you prefer working in Jupyter or want to inspect outputs interactively.

### Option 3: Run directly from Python

You can also call the code directly.

Single run:

```python
from run_simulation import run_simulation

results = run_simulation(
    seed=1,
    run_id=1,
    population_size=11,
    structure="community",
    leader_style="High_Initially_Constrained",
    max_iterations=100,
)
```

Batch run:

```python
from run_multiple_simulations import run_multiple_simulations

batch = run_multiple_simulations(
    seeds=[1, 2, 3],
    population_size=11,
    structure="community",
    leader_style="High_Initially_Constrained",
    max_iterations=100,
)
```

---

## YAML run modes

The YAML file supports three modes.

### 1. `single`
Runs one condition.

Use this when you want one setup and one set of seeds.

### 2. `manual`
Runs a user-written list of conditions.

Use this when you want to explicitly define each condition yourself.

### 3. `grid`
Builds conditions automatically from parameter lists using all combinations.

Use this when you want to sweep over combinations of settings.

---

## How to decide where to edit

### “I want to change how agents are initialized.”
Check:
- `agents.py`

### “I want to change how the leader is defined or styled.”
Check:
- `agents.py`
- `leader_intervention.py`
- `default.yaml`

### “I want to change who interacts with whom.”
Check:
- `member_dynamics.py`
- `network.py`
- `build_simulation.py`

### “I want to change the community/random/core-periphery setup.”
Check:
- `network.py`
- `default.yaml`

### “I want to change leader-member intimacy rules.”
Check:
- `build_simulation.py`
- `network.py`
- `default.yaml`

### “I want to change the emotional contagion update equation.”
Check:
- `member_dynamics.py`

### “I want to change adaptive intimacy behavior.”
Check:
- `member_dynamics.py`
- `default.yaml`

### “I want to change when the leader intervenes or how strong the intervention is.”
Check:
- `leader_intervention.py`
- `default.yaml`

### “I want to change what gets saved after a run.”
Check:
- `run_simulation.py`
- `run_multiple_simulations.py`
- `default.yaml`

### “I want to change how batches are organized or saved.”
Check:
- `run_multiple_simulations.py`
- `default.yaml`

### “I want to add or change plots.”
Check:
- `metrics.py`
- `run_from_config.py`
- `run_from_config.ipynb`

### “I want to add RL later.”
Most likely check:
- a future RL module
- `leader_intervention.py`
- `run_simulation.py`
- `default.yaml`

---

## Typical workflow for a user

### Simple run
1. Edit `default.yaml`
2. Choose `mode: single`
3. Run with either:
   - `python run_from_config.py --config default.yaml`
   - `run_from_config.ipynb`

### Several hand-written conditions
1. Edit `default.yaml`
2. Choose `mode: manual`
3. Add the condition dictionaries you want
4. Run using the script or notebook

### Parameter sweep
1. Edit `default.yaml`
2. Choose `mode: grid`
3. Define `condition_grid`
4. Define `fixed_params`
5. Run using the script or notebook

---

## Notes on saved outputs

### Pickle files
These contain the full Python results objects. They are best when you want to reload detailed run outputs later.

### JSON metadata files
These contain readable metadata for each saved run. They are useful for quickly seeing what a run was without loading Python objects.

### Summary CSV
This contains one row per run and is useful for quick comparisons across seeds and conditions.

### Sentiment evolution plots
These are optional plot outputs showing emotional trajectories over time.

---

## Suggested starting point

If you are new to the project:
1. Open `default.yaml`
2. Set up one small test run
3. Run it using either the script or notebook
4. Check the saved summary and plots
5. Then begin changing deeper files only if you want to modify the model itself
