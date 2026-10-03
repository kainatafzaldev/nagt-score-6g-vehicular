import json
import statistics
import csv

with open("results_multiseed.json", encoding="utf-8") as f:
    data = json.load(f)

with open("model_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Head", "Metric", "Mean", "Std"])

    for model, seeds in data.items():
        for head in ["anomaly", "degradation", "propagation"]:
            for metric in ["acc", "f1", "auc"]:
                values = [
                    seeds[seed]["Test"][head][metric]
                    for seed in seeds
                ]
                writer.writerow([
                    model, head, metric,
                    f"{statistics.mean(values):.4f}",
                    f"{statistics.stdev(values):.4f}"
                ])

        values = [
            seeds[seed]["Test"]["timing_rmse"]
            for seed in seeds
        ]
        writer.writerow([
            model, "timing", "RMSE",
            f"{statistics.mean(values):.4f}",
            f"{statistics.stdev(values):.4f}"
        ])

print("Comparison completed.")
print("Saved to model_comparison.csv")
print("Models:", ", ".join(data.keys()))
