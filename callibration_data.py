import random
import json
from pathlib import Path

def create_and_save_calibration_indices(dataset_size, num_samples, seed=42, output_path="calibration_indices.json"):
    random.seed(seed)
    indices = random.sample(range(dataset_size), num_samples)

    with open(output_path, "w") as f:
        json.dump(indices, f)

    print(f"Saved {num_samples} calibration indices to {output_path}")

# 예시 사용
if __name__ == "__main__":
    dataset_size = 1319  # GSM8K test set 예시
    num_samples = 200
    create_and_save_calibration_indices(dataset_size, num_samples)
