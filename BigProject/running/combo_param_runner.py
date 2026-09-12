from pathlib import Path

from run_from_config import load_config, run_from_config


CONFIG_PATH = Path("default.yaml")

combinations = [
    (True,  True,  "TT"),
    (True,  False, "TF"),
    (False, True,  "FT"),
    (False, False, "FF")
]

for adaptive_intimacy, include_leader_ties, label in combinations:

    config = load_config(CONFIG_PATH)

    # Generate seeds from seed_spec
    start = config["seed_spec"]["start"]
    n_repetitions = config["seed_spec"]["n_repetitions"]
    config["seeds"] = list(range(start, start + n_repetitions))

    # Change only the three requested parameters
    fixed_params = config["grid"]["fixed_params"]
    fixed_params["adaptive_intimacy"] = adaptive_intimacy
    fixed_params["include_leader_ties"] = include_leader_ties
    #fixed_params["directed"] = directed

    print(f"\nRunning combination: {label}")

    batch = run_from_config(config)

    # Add text file to the output folder for this combination
    output_dir = Path(batch["output_dir"])
    (output_dir / "combination.txt").write_text(
        f"combination = {label}\n"
        f"adaptive_intimacy = {adaptive_intimacy}\n"
        f"include_leader_ties = {include_leader_ties}\n",
    #    f"directed = {directed}\n",
        encoding="utf-8",
    )

# For when undirected is functional:
# (True,  True,  True,  "TTT"),
# (True,  True,  False, "TTF"),
# (True,  False, True,  "TFT"),
# (False, True,  True,  "FTT"),
# (False, False, True,  "FFT"),
# (True,  False, False, "TFF"),
# (False, True,  False, "FTF"),
# (False, False, False, "FFF")