import os
import platform
import pickle
import random
import time
from hashlib import md5

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import trange

from aligners.closed_form_procrustes import ProcrustesAligner
from aligners.wasserstein_procrustes import WassersteinAligner
from embedders.node2vec import N2VEmbedder
from embedders.netmf import NetMFEmbedder
from encoders.bf_encoder import BFEncoder
from encoders.non_encoder import NonEncoder
from matchers.bipartite import MinWeightMatcher, GaleShapleyMatcher, SymmetricMatcher

from utils import *

def run(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, BYPASS_CPU = True):

    if "intel" not in platform.processor().lower():
        print("This code uses Intel MKL to speed up computations, but a non-Intel CPU was detected on your system. "
              "We will now bypass the library's CPU check. If you encouter problems, set the BYPASS_CPU flag in main.py "
              "to False.")

        if BYPASS_CPU: os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

    supported_matchings = ["MinWeight","Stable", "Symmetric"]
    assert GLOBAL_CONFIG["Matching"] in supported_matchings, "Error: Matching method must be one of %s" % ((supported_matchings))

    supported_selections = ["Degree", "GroundTruth", "Centroids", "Random", "None", None]
    assert ALIGN_CONFIG["Selection"] in supported_selections, "Error: Selection method for alignment subset must be one of %s" % ((supported_selections))

    if GLOBAL_CONFIG["BenchMode"]:
        start_total = time.time()

    # Compute hashes of configuration to store/load data and thus avoid redundant computations.
    # Using MD5 because Python's native hash() is not stable across processes
    eve_enc_hash = md5(("%s-%s-%s-%s-%s" % (ENC_CONFIG["EveSecret"], ENC_CONFIG["EveBFLength"],ENC_CONFIG["EveBits"],
                                            ENC_CONFIG["EveN"],GLOBAL_CONFIG["Data"])).encode()).hexdigest()
    alice_enc_hash = md5(("%s-%s-%s-%s-%s-%s" % (ENC_CONFIG["EveSecret"], ENC_CONFIG["EveBFLength"],ENC_CONFIG["EveBits"],
                                            ENC_CONFIG["EveN"],GLOBAL_CONFIG["Data"],
                          GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

    emb_config_hash = md5(("%s-%s-%s-%s" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                            GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

    if os.path.isfile("./data/encoded/alice-%s.pck" % alice_enc_hash):
        if GLOBAL_CONFIG["Verbose"]:
            print("Found stored data for Alice's encoded records")

        with open("./data/encoded/alice-%s.pck" % alice_enc_hash, "rb") as f:
            alice_enc, n_alice = pickle.load(f)

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_alice_enc = -1

    else:
        # Load and encode Alice's Data
        if GLOBAL_CONFIG["Verbose"]:
            print("Loading Alice's data")

        alice_data, alice_uids = read_tsv(GLOBAL_CONFIG["Data"])

        # Subset and shuffle
        selected = random.sample(range(len(alice_data)), int(GLOBAL_CONFIG["Overlap"] * len(alice_data)))
        alice_data = [alice_data[i] for i in selected]
        alice_uids = [alice_uids[i] for i in selected]
        n_alice = len(alice_uids) # Initial number of records in alice's dataset. Required for success calculation

        if GLOBAL_CONFIG["BenchMode"]:
            start_alice_enc = time.time()

        alice_bloom_encoder = BFEncoder(ENC_CONFIG["AliceSecret"], ENC_CONFIG["AliceBFLength"],
                                        ENC_CONFIG["AliceBits"], ENC_CONFIG["AliceN"])

        if GLOBAL_CONFIG["Verbose"]:
            print("Encoding Alice's Data")
        alice_enc = alice_bloom_encoder.encode_and_compare(alice_data, alice_uids, metric=ENC_CONFIG["AliceMetric"])

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_alice_enc = time.time() - start_alice_enc

        if GLOBAL_CONFIG["DevMode"]:
            save_tsv(alice_enc, "dev/alice.edg")

        if GLOBAL_CONFIG["Verbose"]:
            print("Done encoding Alice's data")

        with open("./data/encoded/alice-%s.pck" % alice_enc_hash, "wb") as f:
            pickle.dump((alice_enc, n_alice), f, protocol=5)

    if GLOBAL_CONFIG["Verbose"]:
            print("Computing Thresholds and subsetting data for Alice")
    # Compute the threshold value for subsetting
    tres = np.quantile([e[2] for e in alice_enc], EMB_CONFIG["AliceQuantile"])

    # Only keep edges if their similarity is greater than the threshold
    alice_enc = [e for e in alice_enc if e[2] > tres]

    # Discretize the data, i.e. replace all similarities with 1 (thus creating an unweighted graph)
    if EMB_CONFIG["AliceDiscretize"]:
        alice_enc = [(e[0], e[1], 1) for e in alice_enc]

    if GLOBAL_CONFIG["Verbose"]:
        print("Done processing Alice's data.")

    if os.path.isfile("./data/encoded/eve-%s.pck" % eve_enc_hash):
        if GLOBAL_CONFIG["Verbose"]:
            print("Found stored data for Eve's encoded records")

        with open("./data/encoded/eve-%s.pck" % eve_enc_hash, "rb") as f:
            eve_enc, n_eve = pickle.load(f)

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_eve_enc = -1
    else:

        # Load and encode Eve's Data
        if GLOBAL_CONFIG["Verbose"]:
            print("Loading Eve's data")
        eve_data, eve_uids = read_tsv(GLOBAL_CONFIG["Data"])

        selected = random.sample(range(len(eve_data)), len(eve_data))
        eve_data = [eve_data[i] for i in selected]
        eve_uids = [eve_uids[i] for i in selected]

        n_eve = len(eve_uids)

        if GLOBAL_CONFIG["BenchMode"]:
            start_eve_enc = time.time()

        if ENC_CONFIG["EveEnc"]:
            eve_encoder = BFEncoder(ENC_CONFIG["EveSecret"], ENC_CONFIG["EveBFLength"],
                                          ENC_CONFIG["EveBits"], ENC_CONFIG["EveN"])
        else:
            eve_encoder = NonEncoder(ENC_CONFIG["EveN"])

        if GLOBAL_CONFIG["Verbose"]:
            print("Encoding Eve's Data")

        eve_enc = eve_encoder.encode_and_compare(eve_data, eve_uids, metric=ENC_CONFIG["EveMetric"])

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_eve_enc = time.time() - start_eve_enc

        if GLOBAL_CONFIG["DevMode"]:
            save_tsv(eve_enc, "dev/eve.edg")

        with open("./data/encoded/eve-%s.pck" % eve_enc_hash, "wb") as f:
            pickle.dump((eve_enc, n_eve), f, protocol=5)

        if GLOBAL_CONFIG["Verbose"]:
            print("Done encoding Eve's data")

    if GLOBAL_CONFIG["Verbose"]:
        print("Computing Thresholds and subsetting data for Eve")

    tres = np.quantile([e[2] for e in eve_enc], EMB_CONFIG["EveQuantile"])
    eve_enc = [e for e in eve_enc if e[2] > tres]

    if EMB_CONFIG["EveDiscretize"]:
        eve_enc = [(e[0], e[1], 1) for e in eve_enc]

    if GLOBAL_CONFIG["Verbose"]:
        print("Done processing Eve's data.")

    if os.path.isfile("./data/embeddings/alice-%s.pck" % emb_config_hash):
        if GLOBAL_CONFIG["Verbose"]:
            print("Found stored data for Alice's embeddings")

        with open("./data/embeddings/alice-%s.pck" % emb_config_hash, "rb") as f:
            alice_embedder = pickle.load(f)

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_alice_emb = -1

    else:

        if GLOBAL_CONFIG["BenchMode"]:
            start_alice_emb = time.time()

        # # Compute the threshold value for subsetting
        # tres = np.quantile([e[2] for e in alice_enc], EMB_CONFIG["AliceQuantile"])
        #
        # # Only keep edges if their similarity is greater than the threshold
        # alice_enc = [e for e in alice_enc if e[2] > tres]
        #
        # # Discretize the data, i.e. replace all similarities with 1 (thus creating an unweighted graph)
        # if EMB_CONFIG["AliceDiscretize"]:
        #     alice_enc = [(e[0], e[1], 1) for e in alice_enc]

        save_tsv(alice_enc, "data/edgelists/alice.edg")

        if GLOBAL_CONFIG["Verbose"]:
            print("Embedding Alice's data. This may take a while...")

        if EMB_CONFIG["Algo"] == "Node2Vec":
            alice_embedder = N2VEmbedder(walk_length=EMB_CONFIG["AliceWalkLen"], n_walks=EMB_CONFIG["AliceNWalks"],
                                         p=EMB_CONFIG["AliceP"], q=EMB_CONFIG["AliceQ"], dim_embeddings=EMB_CONFIG["AliceDim"],
                                         context_size=EMB_CONFIG["AliceContext"], epochs=EMB_CONFIG["AliceEpochs"],
                                         seed=EMB_CONFIG["AliceSeed"], workers=EMB_CONFIG["Workers"])
        else:
            alice_embedder = NetMFEmbedder(EMB_CONFIG["AliceDim"], EMB_CONFIG["AliceContext"], EMB_CONFIG["AliceNegative"],
                                           EMB_CONFIG["AliceNormalize"])

        alice_embedder.train("./data/edgelists/alice.edg")

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_alice_emb = time.time() - start_alice_emb

        if GLOBAL_CONFIG["DevMode"]:
            alice_embedder.save_model("./dev", "alice.mod")

        if GLOBAL_CONFIG["Verbose"]:
            print("Done embedding Alice's data.")

        # We have to redefine the uids to account for the fact that nodes might have been dropped while ensuring minimum
        # similarity.
        with open("./data/embeddings/alice-%s.pck" % emb_config_hash, "wb") as f:
            pickle.dump(alice_embedder, f, protocol=5)

    alice_embeddings, alice_uids = alice_embedder.get_vectors()

    if ALIGN_CONFIG["Batchsize"] == "Auto":
        ALIGN_CONFIG["Batchsize"] = len(alice_uids) - 50

    if os.path.isfile("./data/embeddings/eve-%s.pck" % emb_config_hash):
        if GLOBAL_CONFIG["Verbose"]:
            print("Found stored data for Eve's embeddings")

        with open("./data/embeddings/eve-%s.pck" % emb_config_hash, "rb") as f:
            eve_embedder = pickle.load(f)

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_eve_emb = -1

    else:

        if GLOBAL_CONFIG["BenchMode"]:
            start_eve_emb = time.time()

        save_tsv(eve_enc, "data/edgelists/eve.edg")

        if GLOBAL_CONFIG["Verbose"]:
            print("Embedding Eve's data. This may take a while...")

        if EMB_CONFIG["Algo"] == "Node2Vec":
            eve_embedder = N2VEmbedder(walk_length=EMB_CONFIG["EveWalkLen"], n_walks=EMB_CONFIG["EveNWalks"],
                                       p=EMB_CONFIG["EveP"], q=EMB_CONFIG["EveQ"], dim_embeddings=EMB_CONFIG["EveDim"],
                                       context_size=EMB_CONFIG["EveContext"], epochs=EMB_CONFIG["EveEpochs"],
                                       seed=EMB_CONFIG["EveSeed"], workers=EMB_CONFIG["Workers"])
        else:
            eve_embedder = NetMFEmbedder(EMB_CONFIG["EveDim"], EMB_CONFIG["EveContext"],
                                           EMB_CONFIG["EveNegative"],
                                           EMB_CONFIG["EveNormalize"])

        eve_embedder.train("./data/edgelists/eve.edg")

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_eve_emb = time.time() - start_eve_emb

        if GLOBAL_CONFIG["DevMode"]:
            eve_embedder.save_model("./dev", "eve.mod")

        if GLOBAL_CONFIG["Verbose"]:
            print("Done embedding Eve's data.")

        with open("./data/embeddings/eve-%s.pck" % emb_config_hash, "wb") as f:
            pickle.dump(eve_embedder, f, protocol=5)

    eve_embeddings, eve_uids = eve_embedder.get_vectors()

    # Alignment
    if GLOBAL_CONFIG["BenchMode"]:
        start_align_prep = time.time()

    if ALIGN_CONFIG["Selection"] == "Degree":
        # Read the edgelists as NetworkX graphs so we can determine the degrees of the nodes.
        #edgelist_alice = nx.read_weighted_edgelist("data/edgelists/alice.edg")
        #edgelist_eve = nx.read_weighted_edgelist("data/edgelists/eve.edg")
        edgelist_alice = nx.from_pandas_edgelist(pd.DataFrame(alice_enc, columns=["source","target","weight"]), edge_attr=True)
        edgelist_eve = nx.from_pandas_edgelist(pd.DataFrame(eve_enc, columns=["source","target","weight"]), edge_attr=True)
        degrees_alice = sorted(edgelist_alice.degree, key=lambda x: x[1], reverse=True)
        degrees_eve = sorted(edgelist_eve.degree, key=lambda x: x[1], reverse=True)

        if GLOBAL_CONFIG["DevMode"]:
            save_tsv(degrees_eve, "./dev/degrees_eve.tsv")
            save_tsv(degrees_alice, "./dev/degrees_alice.tsv")


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

        if GLOBAL_CONFIG["Verbose"]:
            print("Optimal K is %i with silhouette %f" % (optimal_k, tmp))

        ALIGN_CONFIG["Batchsize"] = optimal_k
        ALIGN_CONFIG["VocabSize"] = optimal_k

        kmeans_alice = KMeans(n_clusters=optimal_k, random_state=0, n_init=50).fit(alice_embeddings)
        kmeans_eve = KMeans(n_clusters=optimal_k, random_state=0, n_init=50).fit(eve_embeddings)

        alice_sub = kmeans_alice.cluster_centers_
        eve_sub = kmeans_eve.cluster_centers_

    elif ALIGN_CONFIG["Selection"] == "Random":
        alice_sub = alice_embeddings
        eve_sub = eve_embeddings[np.random.choice(eve_embeddings.shape[0], alice_embeddings.shape[0], replace=False), :]

    else:
        alice_sub = alice_embeddings
        eve_sub = eve_embeddings

    if ALIGN_CONFIG["Selection"] in ["Degree", "GroundTruth"]:
        alice_sub = np.stack(alice_sub, axis=0)
        eve_sub = np.stack(eve_sub, axis=0)

    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_align_prep = time.time() - start_align_prep
        start_align = time.time()

    if GLOBAL_CONFIG["Verbose"]:
        print("Aligning vectors. This may take a while.")

    if ALIGN_CONFIG["Wasserstein"]:
        if ALIGN_CONFIG["RegWS"] == "Auto":
            ALIGN_CONFIG["RegWS"] = min(0.2, len(alice_uids)*0.0001)

        aligner = WassersteinAligner(ALIGN_CONFIG["Batchsize"], ALIGN_CONFIG["RegInit"], ALIGN_CONFIG["RegWS"],
                                      ALIGN_CONFIG["Batchsize"], ALIGN_CONFIG["LR"],ALIGN_CONFIG["NIterInit"],
                                      ALIGN_CONFIG["NIterWS"], ALIGN_CONFIG["NEpochWS"], len(alice_uids),
                                      ALIGN_CONFIG["LRDecay"], ALIGN_CONFIG["Sqrt"], ALIGN_CONFIG["EarlyStopping"],
                                      ALIGN_CONFIG["Verbose"])
    else:
        aligner = ProcrustesAligner()

    transformation_matrix = aligner.align(alice_sub, eve_sub)
    eve_embeddings = np.dot(eve_embeddings, transformation_matrix.T)

    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_align = time.time() - start_align

    if GLOBAL_CONFIG["Verbose"]:
        print("Done.")
        print("Performing bipartite graph matching")

    if GLOBAL_CONFIG["BenchMode"]:
        start_mapping = time.time()

    if GLOBAL_CONFIG["Matching"] == "MinWeight":
        matcher = MinWeightMatcher(GLOBAL_CONFIG["MatchingMetric"])
    elif GLOBAL_CONFIG["Matching"] == "Stable":
        matcher = GaleShapleyMatcher(GLOBAL_CONFIG["MatchingMetric"])
    elif GLOBAL_CONFIG["Matching"] == "Symmetric":
        matcher = SymmetricMatcher(GLOBAL_CONFIG["MatchingMetric"])

    mapping = matcher.match(alice_embeddings, alice_uids, eve_embeddings, eve_uids)

    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_mapping = time.time() - start_mapping

    # Evaluation
    correct = 0
    for eve, alice in mapping.items():
        if eve[0] == "A":
            continue
        if eve[1:] == alice[1:]:
            correct += 1

    success_rate = correct/n_alice
    print("Correct: " + str(correct))
    print(success_rate)

    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_total = time.time() - start_total
        keys = ["timestamp"]
        vals = [time.time()]
        for key, val in EMB_CONFIG.items():
            keys.append(key)
            vals.append(val)

        keys += ["success_rate","correct", "n_alice", "n_eve", "elapsed_total", "elapsed_alice_enc", "elapsed_eve_enc",
                 "elapsed_alice_emb", "elapsed_eve_emb", "elapsed_align_prep", "elapsed_align", "elapsed_mapping"]

        vals += [success_rate, correct, n_alice, n_eve, elapsed_total, elapsed_alice_enc, elapsed_eve_enc,
                 elapsed_alice_emb, elapsed_eve_emb, elapsed_align_prep, elapsed_align, elapsed_mapping]

        if not os.path.isfile("./data/benchmark.tsv"):
            save_tsv([keys], "./data/benchmark.tsv")

        save_tsv([vals], "./data/benchmark.tsv", mode="a")

    return mapping


if __name__ == "__main__":
    # Some global parameters

    GLOBAL_CONFIG = {
        "Data": "./data/fakename_20k.tsv",
        "Overlap": 0.3,
        "DevMode": False,  # Development Mode, saves some intermediate results to the /dev directory
        "BenchMode": False,  # Benchmark Mode
        "Verbose": True,  # Print Status Messages?
        "MatchingMetric": "euclidean",
        "Matching": "Stable"
    }

    ENC_CONFIG = {
        "AliceSecret": "SuperSecretSalt1337",
        "AliceBFLength": 1024,
        "AliceBits": 30,
        "AliceN": 2,
        "AliceMetric": "dice",
        "EveEnc": True,
        "EveSecret": "ATotallyDifferentString",
        "EveBFLength": 1024,
        "EveBits": 30,
        "EveN": 2,
        "EveMetric": "dice",
    }

    EMB_CONFIG = {
        "Algo": "NetMF",
        "AliceQuantile": 0.9,
        "AliceDiscretize": False,
        "AliceDim": 128,
        "AliceContext": 10,
        "AliceNegative": 1,
        "AliceNormalize": True,
        "EveQuantile": 0.9,
        "EveDiscretize": False,
        "EveDim": 128,
        "EveContext": 10,
        "EveNegative": 1,
        "EveNormalize": True,
        "Workers": -1,
    }

    ALIGN_CONFIG = {
        "RegWS": "Auto",
        "RegInit": 0.2,
        "Batchsize": "Auto",
        "LR": 500.0,
        "NIterWS": 2,
        "NIterInit": 10,  # 800
        "NEpochWS": 200,
        "LRDecay": 0.95,
        "Sqrt": False,
        "EarlyStopping": 50,
        "Selection": "None",
        "Wasserstein": True,
        "Verbose": GLOBAL_CONFIG["Verbose"]
    }

    #ae, ee, au, eu = run(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG)
    mp = run(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, BYPASS_CPU=False)
