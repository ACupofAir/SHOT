import matplotlib.pyplot as plt
import numpy as np

# Extracted per-class accuracy values at each iteration
iterations = [
    230, 460, 690, 920, 1150, 1380, 1610, 1840, 2070, 2300, 2530, 2760, 2990, 3220, 3450, 3462
]

# Per-class accuracy values (13 values per iteration)
per_class_accuracies = np.array([
    [89.5, 67.37, 81.49, 61.22, 90.51, 95.76, 85.73, 78.18, 85.12, 55.55, 85.72, 39.01, 76.26],
    [93.61, 78.79, 80.85, 62.47, 91.66, 96.14, 85.84, 79.53, 86.04, 71.81, 85.72, 45.78, 79.85],
    [93.28, 79.19, 79.32, 60.18, 91.96, 96.39, 85.08, 78.72, 86.83, 77.51, 85.01, 49.64, 80.26],
    [93.8, 79.25, 80.36, 58.9, 91.69, 96.39, 86.42, 79.05, 85.71, 81.02, 83.92, 51.77, 80.69],
    [94.02, 82.01, 79.96, 61.62, 92.24, 95.66, 85.99, 79.8, 86.46, 82.86, 85.41, 48.63, 81.22],
    [93.8, 80.6, 80.62, 60.64, 92.73, 94.8, 87.18, 78.62, 85.62, 86.28, 85.88, 50.34, 81.43],
    [94.46, 84.12, 77.08, 58.42, 93.9, 95.57, 86.73, 79.47, 84.59, 84.31, 85.74, 54.81, 81.60],
    [93.94, 83.65, 79.7, 59.96, 92.75, 95.71, 85.33, 78.18, 87.29, 85.4, 85.17, 52.51, 81.63],
    [93.69, 83.48, 78.72, 61.25, 92.6, 94.7, 86.01, 79.75, 89.82, 86.1, 85.93, 52.96, 82.08],
    [94.84, 82.79, 79.83, 59.31, 92.58, 95.66, 84.3, 78.15, 86.0, 87.2, 83.69, 52.94, 81.44],
    [93.86, 84.09, 76.78, 61.46, 93.29, 95.13, 85.02, 78.62, 88.99, 87.2, 86.9, 54.07, 82.12],
    [94.38, 85.78, 77.44, 58.26, 92.47, 95.76, 84.04, 79.7, 86.72, 85.97, 85.06, 52.07, 81.47],
    [93.53, 82.07, 80.26, 58.24, 92.05, 95.71, 82.23, 79.82, 86.48, 87.51, 84.49, 54.38, 81.40],
    [94.3, 84.17, 79.57, 58.54, 91.92, 94.94, 85.63, 78.08, 88.66, 88.29, 86.05, 51.77, 81.83],
    [93.88, 83.37, 78.53, 56.42, 93.05, 95.71, 85.51, 79.27, 88.48, 86.63, 86.87, 54.24, 81.83],
    [94.68, 85.12, 80.43, 60.44, 93.92, 95.66, 85.61, 78.3, 88.74, 85.58, 84.63, 52.14, 82.11],
])

class_map = {
    0: "aeroplane",
    1: "bicycle",
    2: "bus",
    3: "car",
    4: "horse",
    5: "knife",
    6: "motorcycle",
    7: "person",
    8: "plant",
    9: "skateboard",
    10: "train",
    11: "truck",
}

# Correct the data: keep all 13 class accuracies (exclude only the last overall accuracy column)
per_class_accuracies = per_class_accuracies[:, :-1].T  # Remove only the overall accuracy, now with 13 classes

# Plotting again with updated font size
plt.figure(figsize=(14, 8))
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X']
for i in range(12):
    plt.plot(iterations, per_class_accuracies[i], label=f'{class_map[i]}', marker=markers[i])
plt.xlabel("Iteration", fontsize=24)
plt.ylabel("Accuracy (%)", fontsize=24)
plt.title("Per-Class Accuracy over Iterations (Task: TV)", fontsize=24)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()