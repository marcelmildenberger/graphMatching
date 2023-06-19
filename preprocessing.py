import pandas as pd
import numpy as np
import random

from utils import save_csv

OVERLAP = 0.5

data = pd.read_csv("./data/feb14.csv")
#data = data.head(1000)
data.replace(np.nan, "nan", inplace=True)
data = data[["given_name", "surname", "date_of_birth"]]

eve = data

# Creates a list of the indexes in Eve's dataset
ind = list(range(eve.shape[0]))
# Random sampling: Shuffle and select the first n=OVERLAP*len(ind) entries
random.shuffle(ind)
ind = ind[:int(OVERLAP*len(ind))]
alice = eve.iloc[ind]

# Ground truth for evaluation purposes (Indicates which index in Eve's dataset corresponds to which
# index in Alice's dataset)
ground_truth = [("E_"+str(ind[i]), "A_"+str(i)) for i in range(len(ind))]

# Save data
alice.to_csv("./data/alice.csv", index=False)
eve.to_csv("./data/eve.csv", index=False)
save_csv(ground_truth, "./data/eve_to_alice.tsv")

print("Done")


