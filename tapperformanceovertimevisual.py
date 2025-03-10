import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (assuming you will upload it)
file_path = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/bhv/2_clean/data2.csv"  
df = pd.read_csv(file_path, delimiter=";")

# Ensure time column is sorted
df = df.sort_values(by=["global_time"])

# Set Seaborn style
sns.set_style("whitegrid")

# Create figure
plt.figure(figsize=(12, 6))

# Line plot of tap differences over time
sns.lineplot(data=df, x="global_time", y="tap_diff", hue="condition", alpha=0.6)

# Customize labels and title
plt.xlabel("Global Time", fontsize=14)
plt.ylabel("Tap Difference", fontsize=14)
plt.title("Tap Performance Over Time", fontsize=16)
plt.legend(title="Condition")

# Show plot
plt.show()

