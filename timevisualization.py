import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/bhv/2_clean/all_probes3.csv"
df = pd.read_csv(file_path, delimiter=",", encoding="utf-8")

# Filter only relevant columns for visualization
df_filtered = df[['BV', 'AE', 'ON_OFF', 'cond']].dropna()

# Set Seaborn style
sns.set_style("whitegrid")

# Create figure with subplots for BV and AE
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

# Define a color palette
palette = "Set2"

# Boxplot for BV (Behavioral Variability)
sns.boxplot(
    data=df_filtered, 
    x='cond', 
    y='BV', 
    hue='ON_OFF', 
    order=["TRNS", "SHAM", "TACS"], 
    palette=palette, 
    width=0.6, 
    showfliers=False, 
    ax=axes[0]
)
sns.stripplot(
    data=df_filtered, 
    x='cond', 
    y='BV', 
    hue='ON_OFF', 
    dodge=True, 
    palette="dark:.3", 
    alpha=0.4, 
    jitter=True, 
    ax=axes[0]
)
axes[0].set_xlabel("Condition", fontsize=14)
axes[0].set_ylabel("BV (Behavioral Variability)", fontsize=14)
axes[0].set_title("Boxplot of BV across Conditions by ON/OFF State", fontsize=16)
axes[0].legend().remove()  # Remove duplicate legend

# Boxplot for AE (Approximate Entropy)
sns.boxplot(
    data=df_filtered, 
    x='cond', 
    y='AE', 
    hue='ON_OFF', 
    order=["TRNS", "SHAM", "TACS"], 
    palette=palette, 
    width=0.6, 
    showfliers=False, 
    ax=axes[1]
)
sns.stripplot(
    data=df_filtered, 
    x='cond', 
    y='AE', 
    hue='ON_OFF', 
    dodge=True, 
    palette="dark:.3", 
    alpha=0.4, 
    jitter=True, 
    ax=axes[1]
)
axes[1].set_xlabel("Condition", fontsize=14)
axes[1].set_ylabel("AE (Approximate Entropy)", fontsize=14)
axes[1].set_title("Boxplot of AE across Conditions by ON/OFF State", fontsize=16)

# Improve legend placement
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles[:2], labels[:2], title="ON/OFF State", loc="upper right")

# Remove unnecessary spines
sns.despine(left=True, bottom=True)

# Show plot
plt.tight_layout()
plt.show()
