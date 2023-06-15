import csv
import numpy as np

from encoders.bf_encoder import BFEncoder
from embedders.node2vec import N2VEmbedder

# Some global parameters

# Configuration for Bloom filters
BF_CONFIG = {
    "AliceData": "./data/feb14.csv",
    "AliceSecret": "SuperSecretSalt1337",
    "AliceBFLength": 1024,
    "AliceBits": 30,
    "AliceN": 2,
    "EveData": "./data/feb14.csv",
    "EveSecret": "ATotallyDifferentString",
    "EveBFLength": 1024,
    "EveBits": 30,
    "eveN": 2,
}

# Load and encode Alice's Data

alice_bloom_encoder = BFEncoder("secret", 1024, 30, 2)

print("Encoding")
alice_enc = alice_bloom_encoder.encode_and_compare(alice, metric="dice")
print("Done")
alice_sims = [e[2] for e in alice_enc]
tres = np.quantile(alice_sims, 0.99)
alice_minsim = [e for e in alice_enc if e[2]>tres]

with open("./data/testfile.edg", "w", newline="") as f:
    csvwriter = csv.writer(f, delimiter="\t")
    csvwriter.writerows(alice_minsim)

n2v_embedder = N2VEmbedder(walk_length=100, n_walks=20, p=0.5, q=2, dim_embeddings=128, context_size=10, epochs=30,
                           seed=42, workers=6)

n2v_embedder.train("./data/testfile.edg")
n2v_embedder.save_model()

embeddings = n2v_embedder.get_vectors()
