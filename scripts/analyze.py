import subprocess
import pandas as pd
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIG ---
EDGE_LIST_DIR = "../data/generated_graphs"
METADATA_FILE = os.path.join(EDGE_LIST_DIR, "metadata.csv")
INIT_MATRIX = [0.9, 0.5, 0.6, 0.3]
ITERATIONS = 100
LEARNING_RATE = 0.00001
PRED_DIR = "gsdl_predictions"
OUTPUT_CSV = "gsdl_all_predictions.csv"
NUM_WORKERS = 16  # Adjust based on how many cores you want to use

os.makedirs(PRED_DIR, exist_ok=True)

# --- Load Metadata ---
meta = pd.read_csv(METADATA_FILE)

# --- Define task function ---
def process_file(row):
    fname = row["filename"]
    k = row["k_power"]
    edge_path = os.path.join(EDGE_LIST_DIR, fname)
    base = os.path.splitext(fname)[0]
    pred_path = os.path.join(PRED_DIR, f"{base}_gsdl_fit.json")

    # Run prediction if not already done
    if not os.path.exists(pred_path):
        subprocess.run([
            "python3", "../gsdlfit.py", edge_path,
            str(INIT_MATRIX[0]), str(INIT_MATRIX[1]),
            str(INIT_MATRIX[2]), str(INIT_MATRIX[3]),
            str(ITERATIONS), str(LEARNING_RATE)
        ], check=True)

    # Read prediction
    with open(pred_path, "r") as f:
        matrix = json.load(f)
        flat_values = " ".join([str(x) for row in matrix for x in row])

    print(f"Done: {fname}")
    return {"k_value": k, "predictions": flat_values}

# --- Run in parallel ---
results = []
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(process_file, row) for _, row in meta.iterrows()]
    for future in as_completed(futures):
        results.append(future.result())

# --- Save to CSV ---
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nAll predictions saved to: {OUTPUT_CSV}")

