import csv
import numpy as np

from utils import *
from encoders.bf_encoder import BFEncoder
from embedders.node2vec import N2VEmbedder
from aligners.wasserstein_procrustes import WassersteinAligner

# Some global parameters

# Configuration for Bloom filters
ENC_CONFIG = {
    "AliceData": "./data/alice.csv",
    "AliceSecret": "SuperSecretSalt1337",
    "AliceBFLength": 1024,
    "AliceBits": 30,
    "AliceN": 2,
    "AliceMetric": "dice",
    "AliceQuantile": 0.99,
    "EveData": "./data/eve.csv",
    "EveSecret": "ATotallyDifferentString",
    "EveBFLength": 1024,
    "EveBits": 30,
    "EveN": 2,
    "EveMetric": "dice",
    "EveQuantile": 0.99
}

EMB_CONFIG = {
    "AliceWalkLen":100,
    "AliceNWalks": 20,
    "AliceP": 0.5,
    "AliceQ": 2,
    "AliceDim": 128,
    "AliceContext": 10,
    "AliceEpochs": 30,
    "AliceSeed": 42,
    "EveWalkLen": 100,
    "EveNWalks": 20,
    "EveP": 0.5,
    "EveQ": 2,
    "EveDim": 128,
    "EveContext": 10,
    "EveEpochs": 30,
    "EveSeed": 42,
    "Workers": -1
}

ALIGN_CONFIG = {
    "Maxload": 200000,
    "RegWS": 0.9,
    "RegInit": 0.2,
    "Batchsize": 800,
    "LR": 70.0,
    "NIterWS": 500,
    "NIterInit": 800,
    "NEpochWS": 10,
    "VocabSize": 800,
    "LRDecay": 0.9,
    "Sqrt": True,
    "EarlyStopping": 2,
    "Verbose": True
}

# Load and encode Alice's Data
print("Loading Alice's data")
alice_data = read_csv(ENC_CONFIG["AliceData"])
alice_bloom_encoder = BFEncoder(ENC_CONFIG["AliceSecret"], ENC_CONFIG["AliceBFLength"],
                                ENC_CONFIG["AliceBits"], ENC_CONFIG["AliceN"])

print("Encoding Alice's Data")
alice_enc = alice_bloom_encoder.encode_and_compare(alice_data, metric=ENC_CONFIG["AliceMetric"])

print("Computing Thresholds and subsetting data for Alice")
tres = np.quantile([e[2] for e in alice_enc], ENC_CONFIG["AliceQuantile"])
alice_enc = [e for e in alice_enc if e[2] > tres]
print("Done processing Alice's data")

# Load and encode Eve's Data
print("Loading Eve's data")
eve_data = read_csv(ENC_CONFIG["EveData"])
eve_bloom_encoder = BFEncoder(ENC_CONFIG["EveSecret"], ENC_CONFIG["EveBFLength"],
                              ENC_CONFIG["EveBits"], ENC_CONFIG["EveN"])

print("Encoding Eve's Data")
eve_enc = eve_bloom_encoder.encode_and_compare(eve_data, metric=ENC_CONFIG["EveMetric"])

print("Computing Thresholds and subsetting data for Alice")
tres = np.quantile([e[2] for e in eve_enc], ENC_CONFIG["EveQuantile"])
eve_enc = [e for e in eve_enc if e[2] > tres]
print("Done processing Eve's data")

print("Storing results")
save_csv(alice_enc, "data/edgelists/alice.edg")
save_csv(eve_enc, "data/edgelists/eve.edg")
print("Done encoding Data")

print("Start calculating embeddings. This may take a while...")
print("Embedding Alice's data")
alice_embedder = N2VEmbedder(walk_length=EMB_CONFIG["AliceWalkLen"], n_walks=EMB_CONFIG["AliceNWalks"],
                           p=EMB_CONFIG["AliceP"], q=EMB_CONFIG["AliceQ"], dim_embeddings=EMB_CONFIG["AliceDim"],
                           context_size=EMB_CONFIG["AliceContext"], epochs=EMB_CONFIG["AliceEpochs"],
                           seed=EMB_CONFIG["AliceSeed"], workers=EMB_CONFIG["Workers"])

alice_embedder.train("./data/edgelists/alice.edg")
alice_embedder.save_model("./data/models", "alice.mod")

print("Embedding Eve's data")
eve_embedder = N2VEmbedder(walk_length=EMB_CONFIG["EveWalkLen"], n_walks=EMB_CONFIG["EveNWalks"],
                           p=EMB_CONFIG["EveP"], q=EMB_CONFIG["EveQ"], dim_embeddings=EMB_CONFIG["EveDim"],
                           context_size=EMB_CONFIG["EveContext"], epochs=EMB_CONFIG["EveEpochs"],
                           seed=EMB_CONFIG["EveSeed"], workers=EMB_CONFIG["Workers"])

eve_embedder.train("./data/edgelists/eve.edg")
eve_embedder.save_model("./data/models", "eve.mod")

print("Done learning embeddings.")

alice_embeddings = alice_embedder.get_vectors()
eve_embeddings = eve_embedder.get_vectors()

# Clean up
del alice_bloom_encoder, eve_bloom_encoder, alice_data, eve_data, alice_enc, eve_enc, alice_embedder, eve_embedder, tres

# Alignment
print("Aligning vectors. This may take a while.")

aligner = WassersteinAligner(ALIGN_CONFIG["Maxload"], ALIGN_CONFIG["RegInit"], ALIGN_CONFIG["RegWS"],
                             ALIGN_CONFIG["Batchsize"], ALIGN_CONFIG["LR"],ALIGN_CONFIG["NIterInit"],
                             ALIGN_CONFIG["NIterWS"], ALIGN_CONFIG["NEpochWS"], ALIGN_CONFIG["VocabSize"],
                             ALIGN_CONFIG["LRDecay"], ALIGN_CONFIG["Sqrt"], ALIGN_CONFIG["EarlyStopping"],
                             ALIGN_CONFIG["Verbose"])


alice_embeddings, eve_embeddings = aligner.align(alice_embeddings, eve_embeddings)

print("Done.")