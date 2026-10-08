import gc
import time
from pathlib import Path

from run_from_config import load_config, run_from_config


CONFIG_PATH = Path("default.yaml")

# ============================================================
# Population sizes to test
# Each value = 1 leader + (size - 1) members, where (size - 1) % 10 == 0
# ============================================================
POPULATION_SIZES = [11, 41, 81, 111]

# Optionally reduce seeds for larger sizes to manage computation time.
# Set to None to use the config's n_repetitions (100) for all sizes.
# Or provide a dict mapping population_size -> n_seeds.
# Example: {51: 50, 61: 50, 71: 30, 81: 30, 91: 20, 101: 20}
SEEDS_OVERRIDE = None

# Resume support: skip sizes you've already completed.
# Set to a population size (e.g. 31) to start from there.
START_FROM_SIZE = None

# Estimate reference: current N=10 adaptive pickle ~500KB, fixed ~200KB
# Unpickled in-memory objects are roughly 3x the pickle size.
# These are used only for printing warnings, not for any logic.
_BASE_N = 10
_BASE_ADAPTIVE_PKL_KB = 500
_BASE_FIXED_PKL_KB = 200
_MEM_MULTIPLIER = 3  # in-memory object ~= 3x pickle size


# The order of the combinations is important for the labels to match the correct combination.
# The order is: adaptive_intimacy, include_leader_ties
combinations = [
    (True,  True,  "TT"),
    (False, True,  "FT"),
]


def _estimate_disk_gb(pop_size, n_seeds):
    """Estimate total disk usage for one population size (both combinations)."""
    n_runs_per_combo = 18 * n_seeds  # 3 structures × 6 styles × n_seeds
    scale = (pop_size / _BASE_N) ** 2
    adaptive_gb = n_runs_per_combo * _BASE_ADAPTIVE_PKL_KB * scale / 1e6
    fixed_gb = n_runs_per_combo * _BASE_FIXED_PKL_KB * scale / 1e6
    return adaptive_gb + fixed_gb


def _estimate_peak_memory_gb(pop_size, n_seeds, keep_results):
    """Estimate peak memory if all results are held (keep_results=True) or just one (False)."""
    n_runs_per_combo = 18 * n_seeds
    scale = (pop_size / _BASE_N) ** 2
    obj_mb = _BASE_ADAPTIVE_PKL_KB * scale * _MEM_MULTIPLIER / 1e3
    if keep_results:
        return n_runs_per_combo * obj_mb / 1e3
    else:
        return obj_mb / 1e3  # just one object at a time


# ============================================================
# Pre-flight: print estimates and warnings
# ============================================================
print("=" * 70)
print("POPULATION SIZE SWEEP — PRE-FLIGHT ESTIMATES")
print("=" * 70)
print(f"{'Size':>6s} {'Members':>8s} {'Seeds':>6s} {'Runs/combo':>11s} "
      f"{'Total runs':>11s} {'Est. disk':>10s} {'Peak mem (keep=False)':>22s}")
print("-" * 80)

total_disk = 0
sizes_to_run = POPULATION_SIZES
if START_FROM_SIZE is not None:
    sizes_to_run = [s for s in POPULATION_SIZES if s >= START_FROM_SIZE]

for size in sizes_to_run:
    config = load_config(CONFIG_PATH)
    n_seeds = SEEDS_OVERRIDE.get(size, config["seed_spec"]["n_repetitions"]) if SEEDS_OVERRIDE else config["seed_spec"]["n_repetitions"]
    runs_per_combo = 18 * n_seeds
    total_runs = runs_per_combo * len(combinations)
    disk_gb = _estimate_disk_gb(size, n_seeds)
    total_disk += disk_gb
    mem_gb = _estimate_peak_memory_gb(size, n_seeds, keep_results=False)
    print(f"{size:>6d} {size - 1:>8d} {n_seeds:>6d} {runs_per_combo:>11,d} "
          f"{total_runs:>11,d} {disk_gb:>9.1f}G {mem_gb:>21.1f}G")

print("-" * 80)
print(f"{'TOTAL':>6s} {'':>8s} {'':>6s} {'':>11s} "
      f"{'':>11s} {total_disk:>9.1f}G")
print()

# ============================================================
# Main loop
# ============================================================
for pop_size in sizes_to_run:
    print(f"\n{'#' * 70}")
    print(f"# POPULATION SIZE = {pop_size}  ({pop_size - 1} members + 1 leader)")
    print(f"{'#' * 70}")

    config = load_config(CONFIG_PATH)

    # Determine number of seeds for this size
    default_n_seeds = config["seed_spec"]["n_repetitions"]
    if SEEDS_OVERRIDE and pop_size in SEEDS_OVERRIDE:
        n_seeds = SEEDS_OVERRIDE[pop_size]
        config["seed_spec"]["n_repetitions"] = n_seeds
        if n_seeds != default_n_seeds:
            print(f"  NOTE: Using {n_seeds} seeds (reduced from {default_n_seeds})")
    else:
        n_seeds = default_n_seeds

    config["seeds"] = list(range(config["seed_spec"]["start"],
                                  config["seed_spec"]["start"] + n_seeds))

    # Set population size
    fixed_params = config["grid"]["fixed_params"]
    fixed_params["population_size"] = pop_size

    for adaptive_intimacy, include_leader_ties, label in combinations:
        # Reload config to reset any changes from previous combination
        config = load_config(CONFIG_PATH)
        if SEEDS_OVERRIDE and pop_size in SEEDS_OVERRIDE:
            config["seed_spec"]["n_repetitions"] = SEEDS_OVERRIDE[pop_size]
        n_seeds = config["seed_spec"]["n_repetitions"]
        config["seeds"] = list(range(config["seed_spec"]["start"],
                                      config["seed_spec"]["start"] + n_seeds))

        fixed_params = config["grid"]["fixed_params"]
        fixed_params["population_size"] = pop_size
        fixed_params["adaptive_intimacy"] = adaptive_intimacy
        fixed_params["include_leader_ties"] = include_leader_ties

        n_runs = 18 * n_seeds
        print(f"\n  Combination {label} (adaptive={adaptive_intimacy}, "
              f"leader_ties={include_leader_ties}) — {n_runs} runs")

        t0 = time.time()

        # keep_results=False: save to disk but don't hold in memory
        batch = run_from_config(config, keep_results=False)

        elapsed = time.time() - t0
        print(f"  Completed {label} in {elapsed:.0f}s ({elapsed/60:.1f} min)")

        # Write combination.txt
        output_dir = Path(batch["output_dir"])
        (output_dir / "combination.txt").write_text(
            f"combination = {label}\n"
            f"adaptive_intimacy = {adaptive_intimacy}\n"
            f"include_leader_ties = {include_leader_ties}\n"
            f"population_size = {pop_size}\n",
            encoding="utf-8",
        )

        # Free memory
        del batch
        gc.collect()

print(f"\n{'=' * 70}")
print("ALL POPULATION SIZES COMPLETE")
print(f"{'=' * 70}")