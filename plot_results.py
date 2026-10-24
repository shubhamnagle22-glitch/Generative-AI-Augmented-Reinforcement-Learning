
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = {}


def add_result(label, filename, score_columns, position_columns):
    path = os.path.join(PROJECT_DIR, filename)

    if not os.path.exists(path):
        print(f"Skipping missing file: {filename}")
        return

    df = pd.read_csv(path)

    if not all(c in df.columns for c in score_columns + position_columns):
        print(f"Unexpected columns in {filename}: {list(df.columns)}")
        return

    scores = df[score_columns].to_numpy().flatten()
    positions = df[position_columns].to_numpy().flatten()

    results[label] = {
        "average_score": float(np.mean(scores)),
        "first_place_rate": float(np.mean(positions == 0) * 100),
        "scores": scores,
    }


# Random baseline: use the random-vs-random comparison results.
comparison_path = os.path.join(
    PROJECT_DIR, "agent_comparison_results.csv"
)

if os.path.exists(comparison_path):
    df = pd.read_csv(comparison_path)
    random_df = df[df["condition"] == "Random_vs_random"]

    if not random_df.empty:
        results["Random agent (P0)"] = {
            "average_score": float(random_df["score"].mean()),
            "first_place_rate": float(
                (random_df["position"] == 0).mean() * 100
            ),
            "scores": random_df["score"].to_numpy(),
        }
else:
    print("Missing agent_comparison_results.csv")

# DQN evaluation: use P0 only for consistency with the other conditions.
add_result(
    "DQN (P0)",
    "evaluation_results.csv",
    ["P0_score"],
    ["P0_position"],
)

# Generative augmentation evaluation: P0 only.
add_result(
    "Generative augmentation (P0)",
    "generative_augmentation_results.csv",
    ["P0_score"],
    ["P0_position"],
)

if not results:
    raise RuntimeError("No compatible result files found.")

labels = list(results.keys())

# Average score chart
plt.figure()
plt.bar(labels, [results[x]["average_score"] for x in labels])
plt.ylabel("Average score")
plt.title("Agent Comparison: Average Score")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "average_score.png"),
    dpi=200,
)
plt.close()

# First-place rate chart
plt.figure()
plt.bar(labels, [results[x]["first_place_rate"] for x in labels])
plt.ylabel("First-place rate (%)")
plt.title("Agent Comparison: First-Place Rate")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "first_place_rate.png"),
    dpi=200,
)
plt.close()

# Score distribution chart; use the older Matplotlib-compatible labels argument.
plt.figure()
plt.boxplot(
    [results[x]["scores"] for x in labels],
    labels=labels,
)
plt.ylabel("Score")
plt.title("Score Distribution")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "score_distribution.png"),
    dpi=200,
)
plt.close()

# Save summary table
summary = pd.DataFrame([
    {
        "agent": label,
        "average_score": values["average_score"],
        "first_place_rate_percent": values["first_place_rate"],
        "sample_count": len(values["scores"]),
    }
    for label, values in results.items()
])

summary.to_csv(
    os.path.join(OUTPUT_DIR, "summary.csv"),
    index=False,
)

print("\nPlots and summary saved to:", OUTPUT_DIR)
print(summary.to_string(index=False))
