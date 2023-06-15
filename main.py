import csv
import numpy as np
import pandas as pd

from encoders.bf_encoder import BFEncoder
from embedders.node2vec import N2VEmbedder

alice = pd.read_csv("./data/feb14.csv")
alice = alice.head(1000)
alice.replace(np.nan, "", inplace=True)

alice = alice[["given_name", "surname", "address_2"]]
alice = [[str(row[1]["given_name"]), str(row[1]["surname"]), str(row[1]["address_2"])] for row in alice.iterrows()]

alice_bloom_encoder = BFEncoder("secret", 1024, 30, 2)

print("Encoding")
alice_enc = alice_bloom_encoder.encode_and_compare(alice, metric="dice")
print("Done")
alice_sims = [e[2] for e in alice_enc]
tres = np.quantile(alice_sims, 0.99)
alice_minsim = [e for e in alice_enc if e[2]>tres]

with open("./data/testfile.edg", "w") as f:
    csvwriter = csv.writer(f, delimiter="\t")
    csvwriter.writerows(alice_minsim)

n2v_embedder = N2VEmbedder(walk_length=100, n_walks=20, p=0.5, q=2, dim_embeddings=128, context_size=10, epochs=30,
                           seed=42, workers=6)

n2v_embedder.train("./data/testfile.edg")
