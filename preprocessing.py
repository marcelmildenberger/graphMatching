import pandas as pd
import random


OVERLAP = 1.0

data = pd.read_csv("./data/feb14.csv")
data = data.dropna()
data = data.head(1000)
data = data[["given_name", "surname", "date_of_birth"]]
data = data.astype({'date_of_birth': 'int'})


# Creates a list of the indexes in Eve's dataset
ind = list(range(data.shape[0]))

data["uid"] = ind
eve = data

# Random sampling: Shuffle and select the first n=OVERLAP*len(ind) entries
random.shuffle(ind)
ind = ind[:int(OVERLAP*len(ind))]
alice = eve.iloc[ind]


# Save data
alice.to_csv("./data/alice.tsv", index=False, sep="\t")
eve.to_csv("./data/eve.tsv", index=False, sep = "\t")

print("Done")


