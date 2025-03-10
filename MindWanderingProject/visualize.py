import pandas as pd
data2_path = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/bhv/2_clean/data2.csv"
probes2_path = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/bhv/2_clean/probes2.csv"

data2_df = pd.read_csv(data2_path)
probes2_df = pd.read_csv(probes2_path)

data2_info = data2_df.info()
probes2_info = probes2_df.info()

data2_head = data2_df.head()
probes2_head = probes2_df.head()
data2_df = pd.read_csv(data2_path, delimiter=";")
probes2_df = pd.read_csv(probes2_path, delimiter=";")

data2_info = data2_df.info()
probes2_info = probes2_df.info()

data2_head = data2_df.head()
probes2_head = probes2_df.head()

data2_info, data2_head, probes2_info, probes2_head
