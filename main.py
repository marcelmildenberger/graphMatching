import numpy as np
import networkx as nx
import math
import pickle
from tqdm import trange
from utils import *
from encoders.bf_encoder import BFEncoder
from embedders.node2vec import N2VEmbedder
from aligners.wasserstein_procrustes import WassersteinAligner
from aligners.closed_form_procrustes import ProcrustesAligner, normalized
from matchers.bipartite import MinWeightMatcher

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Some global parameters
# Development Mode, saves some intermediate results to the /dev directory
DEV_MODE = True
# Benchmark Mode
BENCH_MODE = True
# Print Status Messages?
VERBOSE = True

# Configuration for Bloom filters
ENC_CONFIG = {
    "AliceData": "./data/alice.tsv",
    "AliceSecret": "SuperSecretSalt1337",
    "AliceBFLength": 1024,
    "AliceBits": 30,
    "AliceN": 2,
    "AliceMetric": "dice",
    "AliceQuantile": 0.95, #0.99
    "AliceDiscretize": False,
    "EveData": "./data/eve.tsv",
    "EveSecret": "ATotallyDifferentString",
    "EveBFLength": 1024,
    "EveBits": 30,
    "EveN": 2,
    "EveMetric": "dice",
    "EveQuantile": 0.95, #0.99
    "EveDiscretize": False
}

EMB_CONFIG = {
    "AliceWalkLen":100,
    "AliceNWalks": 20,
    "AliceP": 0.5, #0.5
    "AliceQ": 2,    #2
    "AliceDim": 64,
    "AliceContext": 10,
    "AliceEpochs": 30,
    "AliceSeed": 42,
    "EveWalkLen": 100,
    "EveNWalks": 20,
    "EveP": 0.5, #0.5
    "EveQ": 2, #2
    "EveDim": 64,
    "EveContext": 10,
    "EveEpochs": 30,
    "EveSeed": 42,
    "Workers": -1
}

ALIGN_CONFIG = {
    "Maxload": 200000,
    "RegWS": 0.9,
    "RegInit": 0.2,
    "Batchsize": 500,
    "LR": 70.0,
    "NIterWS": 500,
    "NIterInit": 800, # 800
    "NEpochWS": 15,
    "VocabSize": 500,
    "LRDecay": 0.9,
    "Sqrt": True,
    "EarlyStopping": 2,
    "Selection": "Degree",
    "Wasserstein": True,
    "Verbose": VERBOSE
}

supported_selections = ["Degree", "GroundTruth", "Centroids", "None"]
assert ALIGN_CONFIG["Selection"] in supported_selections, "Error: Selection method for alignment subset must be one of %s" % ((supported_selections))

# Load and encode Alice's Data
if VERBOSE:
    print("Loading Alice's data")
alice_data, alice_uids = read_tsv(ENC_CONFIG["AliceData"])
alice_bloom_encoder = BFEncoder(ENC_CONFIG["AliceSecret"], ENC_CONFIG["AliceBFLength"],
                                ENC_CONFIG["AliceBits"], ENC_CONFIG["AliceN"])

if VERBOSE:
    print("Encoding Alice's Data")
alice_enc = alice_bloom_encoder.encode_and_compare(alice_data, alice_uids, metric=ENC_CONFIG["AliceMetric"])

if DEV_MODE:
    save_tsv(alice_enc, "dev/alice.edg")

if VERBOSE:
    print("Computing Thresholds and subsetting data for Alice")
# Compute the threshold value for subsetting
tres = np.quantile([e[2] for e in alice_enc], ENC_CONFIG["AliceQuantile"])

# Only keep edges if their similarity is greater than the threshold
alice_enc = [e for e in alice_enc if e[2] > tres]

# Discretize the data, i.e. replace all similarities with 1 (thus creating an unweighted graph)
if ENC_CONFIG["AliceDiscretize"]:
    alice_enc = [(e[0], e[1], 1) for e in alice_enc]

if VERBOSE:
    print("Done processing Alice's data")

# Load and encode Eve's Data
if VERBOSE:
    print("Loading Eve's data")
eve_data, eve_uids = read_tsv(ENC_CONFIG["EveData"])
eve_bloom_encoder = BFEncoder(ENC_CONFIG["EveSecret"], ENC_CONFIG["EveBFLength"],
                              ENC_CONFIG["EveBits"], ENC_CONFIG["EveN"])

if VERBOSE:
    print("Encoding Eve's Data")
eve_enc = eve_bloom_encoder.encode_and_compare(eve_data, eve_uids, metric=ENC_CONFIG["EveMetric"])

if DEV_MODE:
    save_tsv(eve_enc, "dev/eve.edg")

if VERBOSE:
    print("Computing Thresholds and subsetting data for Alice")
tres = np.quantile([e[2] for e in eve_enc], ENC_CONFIG["EveQuantile"])
eve_enc = [e for e in eve_enc if e[2] > tres]
if ENC_CONFIG["EveDiscretize"]:
    eve_enc = [(e[0], e[1], 1) for e in eve_enc]
if VERBOSE:
    print("Done processing Eve's data")


if VERBOSE:
    print("Storing results")
save_tsv(alice_enc, "data/edgelists/alice.edg")
save_tsv(eve_enc, "data/edgelists/eve.edg")
if VERBOSE:
    print("Done encoding Data")

if VERBOSE:
    print("Start calculating embeddings. This may take a while...")
    print("Embedding Alice's data")
alice_embedder = N2VEmbedder(walk_length=EMB_CONFIG["AliceWalkLen"], n_walks=EMB_CONFIG["AliceNWalks"],
                           p=EMB_CONFIG["AliceP"], q=EMB_CONFIG["AliceQ"], dim_embeddings=EMB_CONFIG["AliceDim"],
                           context_size=EMB_CONFIG["AliceContext"], epochs=EMB_CONFIG["AliceEpochs"],
                           seed=EMB_CONFIG["AliceSeed"], workers=EMB_CONFIG["Workers"])

alice_embedder.train("./data/edgelists/alice.edg")

if DEV_MODE:
    alice_embedder.save_model("./dev", "alice.mod")

if VERBOSE:
    print("Embedding Eve's data")
eve_embedder = N2VEmbedder(walk_length=EMB_CONFIG["EveWalkLen"], n_walks=EMB_CONFIG["EveNWalks"],
                           p=EMB_CONFIG["EveP"], q=EMB_CONFIG["EveQ"], dim_embeddings=EMB_CONFIG["EveDim"],
                           context_size=EMB_CONFIG["EveContext"], epochs=EMB_CONFIG["EveEpochs"],
                           seed=EMB_CONFIG["EveSeed"], workers=EMB_CONFIG["Workers"])

eve_embedder.train("./data/edgelists/eve.edg")

if DEV_MODE:
    eve_embedder.save_model("./dev", "eve.mod")

print("Done learning embeddings.")

# We have to redefine the uids to account for the fact that nodes might have been dropped while ensuring minimum
# similarity.
alice_embeddings, alice_uids = alice_embedder.get_vectors()
eve_embeddings, eve_uids = eve_embedder.get_vectors()

if DEV_MODE:
    with open("./dev/alice_emb.pck", "wb") as f:
        pickle.dump((alice_embeddings, alice_uids), f)

    with open("./dev/eve_emb.pck", "wb") as f:
        pickle.dump((eve_embeddings, eve_uids), f)

# Clean up
del alice_bloom_encoder, eve_bloom_encoder, alice_data, eve_data, alice_enc, eve_enc, tres

# Alignment

if ALIGN_CONFIG["Selection"] == "Degree":
    # Read the edgelists as NetworkX graphs so we can determine the degrees of the nodes.
    edgelist_alice = nx.read_weighted_edgelist("data/edgelists/alice.edg")
    edgelist_eve = nx.read_weighted_edgelist("data/edgelists/eve.edg")
    degrees_alice = sorted(edgelist_alice.degree, key=lambda x: x[1], reverse=True)
    degrees_eve = sorted(edgelist_eve.degree, key=lambda x: x[1], reverse=True)
    alice_sub = [alice_embedder.get_vector(k[0]) for k in degrees_alice[:ALIGN_CONFIG["Batchsize"]]]
    eve_sub = [eve_embedder.get_vector(k[0]) for k in degrees_eve[:ALIGN_CONFIG["Batchsize"]]]

elif ALIGN_CONFIG["Selection"] == "GroundTruth":
    alice_sub = [alice_embedder.get_vector(k) for k in alice_uids[:ALIGN_CONFIG["Batchsize"]]]
    eve_sub = [eve_embedder.get_vector(k) for k in alice_uids[:ALIGN_CONFIG["Batchsize"]]]

elif ALIGN_CONFIG["Selection"] == "Centroids":
    sil = []
    kmax = 300
    print("Determining optimal number of clusters K.")
    for k in trange(2, kmax + 1, 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=50).fit(alice_embeddings)
        labels = kmeans.labels_
        sil.append((k, silhouette_score(alice_embeddings, labels, metric='euclidean')))

    tmp = 0
    optimal_k = 0
    for s in sil:
        if s[1] > tmp:
            tmp = s[1]
            optimal_k = s[0]

    if VERBOSE:
        print("Optimal K is %i with silhouette %f" % (optimal_k, tmp))

    ALIGN_CONFIG["Batchsize"] = optimal_k
    ALIGN_CONFIG["VocabSize"] = optimal_k

    kmeans_alice = KMeans(n_clusters=optimal_k, random_state=0, n_init=50).fit(alice_embeddings)
    kmeans_eve = KMeans(n_clusters=optimal_k, random_state=0, n_init=50).fit(eve_embeddings)

    alice_sub = kmeans_alice.cluster_centers_
    eve_sub = kmeans_eve.cluster_centers_

else:
    alice_sub = alice_embeddings
    eve_sub = eve_embeddings
    #eve_sub = eve_embeddings[np.random.choice(eve_embeddings.shape[0], ALIGN_CONFIG["Batchsize"], replace=False), :]

if ALIGN_CONFIG["Selection"] in ["Degree", "GroundTruth"]:
    alice_sub = np.stack(alice_sub, axis=0)
    eve_sub = np.stack(eve_sub, axis=0)

if VERBOSE:
    print("Aligning vectors. This may take a while.")

if ALIGN_CONFIG["Wasserstein"]:
    aligner = WassersteinAligner(ALIGN_CONFIG["Batchsize"], ALIGN_CONFIG["RegInit"], ALIGN_CONFIG["RegWS"],
                                  ALIGN_CONFIG["Batchsize"], ALIGN_CONFIG["LR"],ALIGN_CONFIG["NIterInit"],
                                  ALIGN_CONFIG["NIterWS"], ALIGN_CONFIG["NEpochWS"], ALIGN_CONFIG["VocabSize"],
                                  ALIGN_CONFIG["LRDecay"], ALIGN_CONFIG["Sqrt"], ALIGN_CONFIG["EarlyStopping"],
                                  ALIGN_CONFIG["Verbose"])
else:
    aligner = ProcrustesAligner()

transformation_matrix = aligner.align(alice_sub, eve_sub)

# with open("./dev/list.pck", "rb") as f:
#     selected = pickle.load(f)
#
# alice_sub = alice_embeddings
# eve_sub = eve_embeddings[selected, :]
#
# min_criterion = math.inf
# best_eve_sub = None
#
# for i in range(20):
#     eve_sub = eve_embeddings[np.random.choice(eve_embeddings.shape[0], ALIGN_CONFIG["Batchsize"], replace=False), :]
#     _, tmp = aligner.convex_init(alice_sub, eve_sub)
#     if tmp < min_criterion:
#         print("Found new best matching subset: " + str(tmp))
#         min_criterion = tmp
#         best_eve_sub = eve_sub

alice_embeddings = alice_embeddings / np.linalg.norm(alice_embeddings, 2, 1).reshape([-1, 1])

eve_embeddings = eve_embeddings / np.linalg.norm(eve_embeddings, 2, 1).reshape([-1, 1])
eve_embeddings = np.dot(eve_embeddings, transformation_matrix.T)

# alice_embeddings = normalized(alice_embeddings)
# eve_embeddings = normalized(eve_embeddings)
# eve_embeddings = np.matmul(eve_embeddings, transformation_matrix)

if VERBOSE:
    print("Done.")
    print("Performing bipartite graph matching")

matcher = MinWeightMatcher("cosine")
mapping = matcher.match(alice_embeddings, alice_uids, eve_embeddings, eve_uids)

# Evaluation
correct = 0
for eve, alice in mapping.items():
    if eve[0] == "A":
        continue
    if eve[1:] == alice[1:]:
        correct += 1

print(correct/len(alice_uids))