import numpy as np
import pandas as pd
import random


#OVERLAP = 1

#data = pd.read_csv("./data/feb14.csv")
data = pd.read_csv("./data/FakeName.csv")
#data = data.dropna()
data.replace(np.nan, "", inplace=True)

data = data[["GivenName", "Surname", "Birthday"]]
data = data.astype({'Birthday': 'str'})


# Creates a list of the indexes in Eve's dataset
ind = list(range(data.shape[0]))

data["uid"] = ind
#eve = data

# Random sampling: Shuffle and select the first n=OVERLAP*len(ind) entries
random.shuffle(ind)
#ind = ind[:int(OVERLAP*len(ind))]
data = data.iloc[ind]
data = data.head(50000)

#data["uid"] = data["uid"] + 100

# Save data
data.to_csv("./data/fakename_50k.tsv", index=False, sep="\t")
#eve.to_csv("./data/eve.tsv", index=False, sep = "\t")

print("Done")


