import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ✅ 1. Load the dataset
file_path = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/bhv/2_clean/all_probes3.csv"  # Update if needed
df = pd.read_csv(file_path)

# Ensure relevant columns exist
target_variables = ["attention", "intention", "somnolence"]

# Define a consistent color palette for each measure
colors = {"attention": "royalblue", "intention": "darkorange", "somnolence": "seagreen"}

# ✅ 2. Check if 'subj' column exists and prepare data
if "subj" in df.columns:
    # Group by subject and get the mean values of each measure
    df_grouped = df.groupby("subj")[target_variables].mean().reset_index()

    # Set position for each subject
    subjects = df_grouped["subj"]
    indices = np.arange(len(subjects))  # Numeric x-axis positions

    # Define bar width
    bar_width = 0.6

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 7))

    # Stack the bars
    bottom_values = np.zeros(len(subjects))  # Initialize bottom position for stacking
    for measure in target_variables:
        ax.bar(indices, df_grouped[measure], label=measure.capitalize(), color=colors[measure], width=bar_width, bottom=bottom_values)
        bottom_values += df_grouped[measure]  # Update bottom position

    # ✅ 3. Improve readability
    ax.set_xticks(indices)
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=12)  # Rotate labels for better readability
    ax.set_ylabel("Value", fontsize=14)
    ax.set_xlabel("Subjects", fontsize=14)
    ax.set_title("Stacked Bar Chart of Mind Wandering Measures Across Participants", fontsize=16)
    ax.legend(title="Measure", loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside for clarity
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # ✅ 4. Show the plot
    plt.show()

else:
    print("⚠️ 'subj' column not found in dataset. Unable to compare across participants.")
