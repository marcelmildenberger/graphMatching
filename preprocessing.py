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

if False:
    data_red = data.head(5000).copy()
    ind = list(range(data_red.shape[0]))
    data_red["uid"] = ind

    data_disj = data.tail(5000).copy()
    ind = list(range(data_disj.shape[0]))
    data_disj["uid"] = ind
    # Save data
    data_red.to_csv("./data/fakename_5k.tsv", index=False, sep="\t")
    data_disj.to_csv("./data/disjoint_5k.tsv", index=False, sep="\t")
else:
    data = data.head(5000)
    data.to_csv("./data/fakename_5k.tsv", index=False, sep="\t")
#eve.to_csv("./data/eve.tsv", index=False, sep = "\t")

print("Done")


