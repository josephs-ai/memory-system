import os
import subprocess
import json
from pathlib import Path

def run_ablation(name, env_vars):
    print(f"Running ablation: {name}")
    env = os.environ.copy()
    env.update(env_vars)
    cmd = ["python3", "scripts/retrieval_benchmark.py", "run", "--target", "context_hydrator", "--output", f"ablation_{name}.json"]
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True)
        with open(f"ablation_{name}.json") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed {name}: {e}")
        return None

def main():
    # We will just write the script now to see if we can execute it.
    print("Ablation script ready.")

if __name__ == "__main__":
    main()
