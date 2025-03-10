import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Load the dataset
file_path = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/bhv/2_clean/all_probes3.csv"  # Update if needed
df = pd.read_csv(file_path, delimiter=";")

# ✅ Ensure necessary columns exist
features = ["attention", "intention", "somnolence", "pupil_size_mean"]
target = "ON_OFF"  # Change if another column defines task status

# ✅ Check if required columns exist
if all(col in df.columns for col in features + [target]):
    # Pairplot to visualize relationships
    sns.pairplot(df, hue=target, vars=features, palette="coolwarm", plot_kws={'alpha': 0.6, 's': 50})

    plt.suptitle("Feature Relationships Based on Task Condition (ON vs. OFF)", fontsize=14, y=1.02)
    plt.show()
else:
    print(f"⚠️ One or more required columns ({features + [target]}) not found in dataset. Check column names.")
