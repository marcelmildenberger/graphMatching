import pandas as pd
import numpy as np

from typing import Sequence

data = pd.read_csv("./data/feb14.csv")
#data = data.head(1000)
data.replace(np.nan, "nan", inplace=True)
data = data[["given_name", "surname", "date_of_birth"]]

data.to_csv("./data/alice.csv", index=False)
data.to_csv("./data/eve.csv", index=False)
print("Done")


