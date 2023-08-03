import hashlib
import os
import platform
import pickle
import random
import time

import networkx as nx
import numpy as np
import pandas as pd

from hashlib import md5
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import trange

from aligners.closed_form_procrustes import ProcrustesAligner
from aligners.wasserstein_procrustes import WassersteinAligner
from embedders.node2vec import N2VEmbedder
from embedders.netmf import NetMFEmbedder
from encoders.bf_encoder import BFEncoder
from encoders.tmh_encoder import TMHEncoder
from encoders.tsh_encoder import TSHEncoder
from encoders.non_encoder import NonEncoder
from matchers.bipartite import MinWeightMatcher, GaleShapleyMatcher, SymmetricMatcher
from matchers.spatial import NNMatcher
from utils import *


def run(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG):


    # Sanity Check: Ensure that valid options were specified by the user
    supported_matchings = ["MinWeight", "Stable", "Symmetric", "NearestNeighbor"]
    assert GLOBAL_CONFIG["Matching"] in supported_matchings, "Error: Matching method must be one of %s" % (
        (supported_matchings))

    supported_selections = ["Degree", "GroundTruth", "Centroids", "Random", "None", None]
    assert ALIGN_CONFIG[
               "Selection"] in supported_selections, "Error: Selection method for alignment subset must be one of %s" % (
        (supported_selections))

    supported_drops = ["Alice", "Eve", "Both"]
    assert GLOBAL_CONFIG[
               "DropFrom"] in supported_drops, "Error: Data must be dropped from one of %s" % (
        (supported_drops))

    supported_encs = ["BloomFilter", "TabMinHash", "TwoStepHash", "None", None]
    assert (ENC_CONFIG["AliceAlgo"] in supported_encs and ENC_CONFIG["EveAlgo"] in supported_encs), "Error: Encoding " \
                                    "method must be one of %s" % ((supported_encs))

    if GLOBAL_CONFIG["BenchMode"]:
        start_total = time.time()

    # Compute hashes of configuration to store/load data and thus avoid redundant computations.
    # Using MD5 because Python's native hash() is not stable across processes
    if GLOBAL_CONFIG["DropFrom"] == "Alice":

        eve_enc_hash = md5(
            ("%s-%s-DropAlice" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()).hexdigest()
        alice_enc_hash = md5(
            ("%s-%s-%s-DropAlice" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                     GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        eve_emb_hash = md5(
            ("%s-%s-%s-DropAlice" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()).hexdigest()

        alice_emb_hash = md5(("%s-%s-%s-%s-DropAlice" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                                         GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()
    elif GLOBAL_CONFIG["DropFrom"] == "Eve":

        eve_enc_hash = md5(
            ("%s-%s-%s-DropEve" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                   GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        alice_enc_hash = md5(("%s-%s-DropEve" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()).hexdigest()

        eve_emb_hash = md5(("%s-%s-%s-%s-DropEve" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                                     GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        alice_emb_hash = md5(("%s-%s-%s-DropEve" % (str(EMB_CONFIG), str(ENC_CONFIG),
                                                    GLOBAL_CONFIG["Data"])).encode()).hexdigest()
    else:
        eve_enc_hash = md5(
            ("%s-%s-%s-DropBoth" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                    GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        alice_enc_hash = md5(
            ("%s-%s-%s-DropBoth" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                    GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        eve_emb_hash = md5(("%s-%s-%s-%s-DropBoth" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                                      GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        alice_emb_hash = md5(("%s-%s-%s-%s-DropBoth" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                                        GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

    if os.path.isfile("./data/encoded/alice-%s.pck" % alice_enc_hash):
        if GLOBAL_CONFIG["Verbose"]:
            print("Found stored data for Alice's encoded records")

        with open("./data/encoded/alice-%s.pck" % alice_enc_hash, "rb") as f:
            alice_enc, n_alice = pickle.load(f)
            # alice_enc, n_alice = joblib.load(f)

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_alice_enc = -1

    else:
        # Load and encode Alice's Data
        if GLOBAL_CONFIG["Verbose"]:
            print("Loading Alice's data")

        alice_data, alice_uids = read_tsv(GLOBAL_CONFIG["Data"])

        if GLOBAL_CONFIG["DropFrom"] == "Both":
            # Compute the maximum number of overlapping records possible for the given dataset size and overlap
            overlap_count = int(-(GLOBAL_CONFIG["Overlap"] * len(alice_data) / (GLOBAL_CONFIG["Overlap"] - 2)))
            # Randomly select the overlapping records from the set of available records (all records in the data)
            available = list(range(len(alice_data)))
            selected_overlap = random.sample(available, overlap_count)
            # Remove the overlapping records from the set of available records to ensure that the remaining records are
            # disjoint.
            available = [i for i in available if i not in selected_overlap]
            # Randomly select the records exclusively held by Alice
            selected_alice = random.sample(available, int((len(alice_data) - overlap_count) / 2))
            # Remove Alice's records from the set of available records
            available = [i for i in available if i not in selected_alice]
            # Merge Alice's records with the overlapping records
            selected_alice += selected_overlap
            # Shuffle again because otherwise the order of the overlapping rows would be identical for Eve's and
            # Alice's data.
            selected_alice = random.sample(selected_alice, len(selected_alice))

        else:
            # Subset and shuffle
            alice_ratio = GLOBAL_CONFIG["Overlap"] if GLOBAL_CONFIG["DropFrom"] == "Alice" else 1
            selected_alice = random.sample(range(len(alice_data)), int(alice_ratio * len(alice_data)))

        alice_data = [alice_data[i] for i in selected_alice]
        alice_uids = [alice_uids[i] for i in selected_alice]
        n_alice = len(alice_uids)  # Initial number of records in alice's dataset. Required for success calculation

        if GLOBAL_CONFIG["BenchMode"]:
            start_alice_enc = time.time()

        if ENC_CONFIG["AliceAlgo"] == "BloomFilter":
            alice_encoder = BFEncoder(ENC_CONFIG["AliceSecret"], ENC_CONFIG["AliceBFLength"],
                                      ENC_CONFIG["AliceBits"], ENC_CONFIG["AliceN"])
        elif ENC_CONFIG["AliceAlgo"] == "TabMinHash":
            alice_encoder = TMHEncoder(ENC_CONFIG["AliceBits"], ENC_CONFIG["AliceTables"],
                                       ENC_CONFIG["AliceKeyLen"], ENC_CONFIG["AliceValLen"], hashlib.md5,
                                       ENC_CONFIG["AliceN"],
                                       random_seed=ENC_CONFIG["AliceSecret"], verbose=GLOBAL_CONFIG["Verbose"])
        elif ENC_CONFIG["AliceAlgo"] == "TwoStepHash":
            alice_encoder = TSHEncoder(ENC_CONFIG["AliceNHashFunc"], ENC_CONFIG["AliceNHashCol"], ENC_CONFIG["AliceN"],
                                       ENC_CONFIG["AliceRandMode"], secret=ENC_CONFIG["AliceSecret"],
                                       verbose=GLOBAL_CONFIG["Verbose"])
        else:
            alice_encoder = NonEncoder(ENC_CONFIG["AliceN"])

        if GLOBAL_CONFIG["Verbose"]:
            print("Encoding Alice's Data")
        alice_enc = alice_encoder.encode_and_compare(alice_data, alice_uids, metric=ENC_CONFIG["AliceMetric"], sim=False)

        del alice_data

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_alice_enc = time.time() - start_alice_enc

        if GLOBAL_CONFIG["DevMode"]:
            save_tsv(alice_enc, "dev/alice.edg")

        if GLOBAL_CONFIG["Verbose"]:
            print("Done encoding Alice's data")

        with open("./data/encoded/alice-%s.pck" % alice_enc_hash, "wb") as f:
            pickle.dump((alice_enc, n_alice), f, protocol=5)
            # joblib.dump((alice_enc, n_alice), f, protocol=5, compress=3)

    if GLOBAL_CONFIG["Verbose"]:
        print("Computing Thresholds and subsetting data for Alice")
    # Compute the threshold value for subsetting
    tres = np.quantile([e[2] for e in alice_enc], EMB_CONFIG["AliceQuantile"])

    # Only keep edges if their similarity is greater than the threshold
    alice_enc = [e for e in alice_enc if e[2] < tres]

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
            # eve_enc, n_eve = joblib.load(f)

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_eve_enc = -1
    else:

        # Load and encode Eve's Data
        if GLOBAL_CONFIG["Verbose"]:
            print("Loading Eve's data")
        eve_data, eve_uids = read_tsv(GLOBAL_CONFIG["Data"])

        if GLOBAL_CONFIG["DropFrom"] == "Both":
            selected_eve = selected_overlap + available
            selected_eve = random.sample(selected_eve, len(selected_eve))
        else:
            eve_ratio = GLOBAL_CONFIG["Overlap"] if GLOBAL_CONFIG["DropFrom"] == "Eve" else 1
            selected_eve = random.sample(range(len(eve_data)), int(eve_ratio * len(eve_data)))

        eve_data = [eve_data[i] for i in selected_eve]
        eve_uids = [eve_uids[i] for i in selected_eve]

        n_eve = len(eve_uids)

        if GLOBAL_CONFIG["BenchMode"]:
            start_eve_enc = time.time()

        if ENC_CONFIG["EveAlgo"] == "BloomFilter":
            eve_encoder = BFEncoder(ENC_CONFIG["EveSecret"], ENC_CONFIG["EveBFLength"],
                                    ENC_CONFIG["EveBits"], ENC_CONFIG["EveN"])
        elif ENC_CONFIG["EveAlgo"] == "TabMinHash":
            eve_encoder = TMHEncoder(ENC_CONFIG["EveBits"], ENC_CONFIG["EveTables"],
                                     ENC_CONFIG["EveKeyLen"], ENC_CONFIG["EveValLen"], hashlib.md5,
                                     ENC_CONFIG["EveN"],
                                     random_seed=ENC_CONFIG["EveSecret"], verbose=GLOBAL_CONFIG["Verbose"])
        elif ENC_CONFIG["EveAlgo"] == "TwoStepHash":
            eve_encoder = TSHEncoder(ENC_CONFIG["EveNHashFunc"], ENC_CONFIG["EveNHashCol"], ENC_CONFIG["EveN"],
                                       ENC_CONFIG["EveRandMode"], secret=ENC_CONFIG["EveSecret"],
                                       verbose=GLOBAL_CONFIG["Verbose"])
        else:
            eve_encoder = NonEncoder(ENC_CONFIG["EveN"])

        if GLOBAL_CONFIG["Verbose"]:
            print("Encoding Eve's Data")

        eve_enc = eve_encoder.encode_and_compare(eve_data, eve_uids, metric=ENC_CONFIG["EveMetric"], sim=False)
        del eve_data

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_eve_enc = time.time() - start_eve_enc

        if GLOBAL_CONFIG["DevMode"]:
            save_tsv(eve_enc, "dev/eve.edg")

        if GLOBAL_CONFIG["Verbose"]:
            print("Done encoding Eve's data")

        with open("./data/encoded/eve-%s.pck" % eve_enc_hash, "wb") as f:
            pickle.dump((eve_enc, n_eve), f, protocol=5)

    if GLOBAL_CONFIG["Verbose"]:
        print("Computing Thresholds and subsetting data for Eve")

    tres = np.quantile([e[2] for e in eve_enc], EMB_CONFIG["EveQuantile"])
    eve_enc = [e for e in eve_enc if e[2] < tres]

    if EMB_CONFIG["EveDiscretize"]:
        eve_enc = [(e[0], e[1], 1) for e in eve_enc]

    if GLOBAL_CONFIG["Verbose"]:
        print("Done processing Eve's data.")

    if GLOBAL_CONFIG["BenchMode"]:
        start_alice_emb = time.time()

    if os.path.isfile("./data/embeddings/alice-%s.pck" % alice_emb_hash):
        if GLOBAL_CONFIG["Verbose"]:
            print("Found stored data for Alice's embeddings")

        with open("./data/embeddings/alice-%s.pck" % alice_emb_hash, "rb") as f:
            alice_embedder = pickle.load(f)
            # alice_embedder = joblib.load(f)

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_alice_emb = -1

    else:

        if GLOBAL_CONFIG["Verbose"]:
            print("Embedding Alice's data. This may take a while...")

        if EMB_CONFIG["Algo"] == "Node2Vec":
            save_tsv(alice_enc, "data/edgelists/alice.edg")

            alice_embedder = N2VEmbedder(walk_length=EMB_CONFIG["AliceWalkLen"], n_walks=EMB_CONFIG["AliceNWalks"],
                                         p=EMB_CONFIG["AliceP"], q=EMB_CONFIG["AliceQ"],
                                         dim_embeddings=EMB_CONFIG["AliceDim"],
                                         context_size=EMB_CONFIG["AliceContext"], epochs=EMB_CONFIG["AliceEpochs"],
                                         seed=EMB_CONFIG["AliceSeed"], workers=EMB_CONFIG["Workers"])
            alice_embedder.train("./data/edgelists/alice.edg")

        else:
            alice_embedder = NetMFEmbedder(EMB_CONFIG["AliceDim"], EMB_CONFIG["AliceContext"],
                                           EMB_CONFIG["AliceNegative"],
                                           EMB_CONFIG["AliceNormalize"])

            alice_embedder.train(alice_enc)

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_alice_emb = time.time() - start_alice_emb

        if GLOBAL_CONFIG["DevMode"]:
            alice_embedder.save_model("./dev", "alice.mod")

        if GLOBAL_CONFIG["Verbose"]:
            print("Done embedding Alice's data.")

        with open("./data/embeddings/alice-%s.pck" % alice_emb_hash, "wb") as f:
            pickle.dump(alice_embedder, f, protocol=5)
            # joblib.dump(alice_embedder, f, protocol=5, compress=3)

    # We have to redefine the uids to account for the fact that nodes might have been dropped while ensuring minimum
    # similarity.
    alice_embeddings, alice_uids = alice_embedder.get_vectors()

    if os.path.isfile("./data/embeddings/eve-%s.pck" % eve_emb_hash):
        if GLOBAL_CONFIG["Verbose"]:
            print("Found stored data for Eve's embeddings")

        with open("./data/embeddings/eve-%s.pck" % eve_emb_hash, "rb") as f:
            eve_embedder = pickle.load(f)
            # eve_embedder = joblib.load(f)

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_eve_emb = -1

    else:

        if GLOBAL_CONFIG["BenchMode"]:
            start_eve_emb = time.time()

        if GLOBAL_CONFIG["Verbose"]:
            print("Embedding Eve's data. This may take a while...")

        if EMB_CONFIG["Algo"] == "Node2Vec":
            save_tsv(eve_enc, "data/edgelists/eve.edg")

            eve_embedder = N2VEmbedder(walk_length=EMB_CONFIG["EveWalkLen"], n_walks=EMB_CONFIG["EveNWalks"],
                                       p=EMB_CONFIG["EveP"], q=EMB_CONFIG["EveQ"], dim_embeddings=EMB_CONFIG["EveDim"],
                                       context_size=EMB_CONFIG["EveContext"], epochs=EMB_CONFIG["EveEpochs"],
                                       seed=EMB_CONFIG["EveSeed"], workers=EMB_CONFIG["Workers"])
            eve_embedder.train("./data/edgelists/eve.edg")
        else:
            eve_embedder = NetMFEmbedder(EMB_CONFIG["EveDim"], EMB_CONFIG["EveContext"],
                                         EMB_CONFIG["EveNegative"],
                                         EMB_CONFIG["EveNormalize"])
            eve_embedder.train(eve_enc)

        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_eve_emb = time.time() - start_eve_emb

        if GLOBAL_CONFIG["DevMode"]:
            eve_embedder.save_model("./dev", "eve.mod")

        if GLOBAL_CONFIG["Verbose"]:
            print("Done embedding Eve's data.")

        with open("./data/embeddings/eve-%s.pck" % eve_emb_hash, "wb") as f:
            pickle.dump(eve_embedder, f, protocol=5)

    eve_embeddings, eve_uids = eve_embedder.get_vectors()

    if ALIGN_CONFIG["Batchsize"] == "Auto":
        ALIGN_CONFIG["Batchsize"] = min(len(alice_uids), len(eve_uids))
        if ENC_CONFIG["EveAlgo"] == "TwoStepHash" or ENC_CONFIG["AliceAlgo"] == "TwoStepHash":
            ALIGN_CONFIG["Batchsize"] = int(0.85 * ALIGN_CONFIG["Batchsize"])

    # Alignment
    if GLOBAL_CONFIG["BenchMode"]:
        start_align_prep = time.time()

    if ALIGN_CONFIG["Selection"] == "Degree":
        # Read the edgelists as NetworkX graphs so we can determine the degrees of the nodes.
        # edgelist_alice = nx.read_weighted_edgelist("data/edgelists/alice.edg")
        # edgelist_eve = nx.read_weighted_edgelist("data/edgelists/eve.edg")
        edgelist_alice = nx.from_pandas_edgelist(pd.DataFrame(alice_enc, columns=["source", "target", "weight"]),
                                                 edge_attr=True)
        edgelist_eve = nx.from_pandas_edgelist(pd.DataFrame(eve_enc, columns=["source", "target", "weight"]),
                                               edge_attr=True)
        degrees_alice = sorted(edgelist_alice.degree, key=lambda x: x[1], reverse=True)
        degrees_eve = sorted(edgelist_eve.degree, key=lambda x: x[1], reverse=True)

        if GLOBAL_CONFIG["DevMode"]:
            save_tsv(degrees_eve, "./dev/degrees_eve.tsv")
            save_tsv(degrees_alice, "./dev/degrees_alice.tsv")

        alice_sub = [alice_embedder.get_vector(k[0]) for k in degrees_alice[:ALIGN_CONFIG["Batchsize"]]]
        eve_sub = [eve_embedder.get_vector(k[0]) for k in degrees_eve[:ALIGN_CONFIG["Batchsize"]]]

        del edgelist_eve, edgelist_alice, degrees_eve, degrees_alice

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
        if len(eve_uids) < len(alice_uids):
            alice_sub = alice_embeddings
            eve_sub = eve_embeddings[
                      np.random.choice(eve_embeddings.shape[0], alice_embeddings.shape[0], replace=False), :]
        else:
            eve_sub = eve_embeddings
            alice_sub = alice_embeddings[
                        np.random.choice(alice_embeddings.shape[0], eve_embeddings.shape[0], replace=False), :]

    else:
        alice_sub = alice_embeddings
        eve_sub = eve_embeddings

    del alice_enc, eve_enc, alice_embedder, eve_embedder

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
            if ENC_CONFIG["EveAlgo"] == "TwoStepHash" or ENC_CONFIG["AliceAlgo"] == "TwoStepHash":
                ALIGN_CONFIG["RegWS"] = 0.1
            else:
                smallest_dataset_size = min(len(alice_uids), len(eve_uids))
                ALIGN_CONFIG["RegWS"] = min(0.1, max(0.01, smallest_dataset_size * 0.00001))
            # est_overlap = min(len(alice_uids), len(eve_uids)) / max(len(alice_uids),len(eve_uids))
            # if est_overlap > 0.5:
            #     ALIGN_CONFIG["RegWS"] = 0.1
            # else:
            #     ALIGN_CONFIG["RegWS"] = max(0.02, (est_overlap/10)-0.01)

        aligner = WassersteinAligner(ALIGN_CONFIG["Batchsize"], ALIGN_CONFIG["RegInit"], ALIGN_CONFIG["RegWS"],
                                     ALIGN_CONFIG["Batchsize"], ALIGN_CONFIG["LR"], ALIGN_CONFIG["NIterInit"],
                                     ALIGN_CONFIG["NIterWS"], ALIGN_CONFIG["NEpochWS"], len(alice_uids),
                                     ALIGN_CONFIG["LRDecay"], ALIGN_CONFIG["Sqrt"], ALIGN_CONFIG["EarlyStopping"],
                                     verbose=GLOBAL_CONFIG["Verbose"])
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
    elif GLOBAL_CONFIG["Matching"] == "NearestNeighbor":
        matcher = NNMatcher(GLOBAL_CONFIG["MatchingMetric"])

    mapping = matcher.match(alice_embeddings, alice_uids, eve_embeddings, eve_uids)

    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_mapping = time.time() - start_mapping
        elapsed_relevant = time.time() - start_alice_emb

    # Evaluation
    correct = 0
    for smaller, larger in mapping.items():
        if smaller[0] == "L":
            continue
        if smaller[1:] == larger[1:]:
            correct += 1

    if GLOBAL_CONFIG["DropFrom"] == "Both":
        success_rate = correct / overlap_count
        print("Correct: %i of %i" % (correct, overlap_count))
    else:
        success_rate = correct / min(n_alice, n_eve)
        print("Correct: %i of %i" % (correct, min(n_alice, n_eve)))

    print("Success rate: %f" % success_rate)

    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_total = time.time() - start_total
        keys = ["timestamp"]
        vals = [time.time()]
        for key, val in EMB_CONFIG.items():
            keys.append(key)
            vals.append(val)
        for key, val in ENC_CONFIG.items():
            keys.append(key)
            vals.append(val)
        for key, val in GLOBAL_CONFIG.items():
            keys.append(key)
            vals.append(val)
        for key, val in ALIGN_CONFIG.items():
            keys.append(key)
            vals.append(val)
        keys += ["success_rate", "correct", "n_alice", "n_eve", "elapsed_total", "elapsed_alice_enc", "elapsed_eve_enc",
                 "elapsed_alice_emb", "elapsed_eve_emb", "elapsed_align_prep", "elapsed_align", "elapsed_mapping",
                 "elapsed_relevant"]

        vals += [success_rate, correct, n_alice, n_eve, elapsed_total, elapsed_alice_enc, elapsed_eve_enc,
                 elapsed_alice_emb, elapsed_eve_emb, elapsed_align_prep, elapsed_align, elapsed_mapping,
                 elapsed_relevant]

        if not os.path.isfile("./data/benchmark.tsv"):
            save_tsv([keys], "./data/benchmark.tsv")

        save_tsv([vals], "./data/benchmark.tsv", mode="a")

    return mapping


if __name__ == "__main__":
    # Some global parameters

    GLOBAL_CONFIG = {
        "Data": "./data/fakename_20k.tsv",
        "Overlap": 0.8,
        "DropFrom": "Alice",
        "DevMode": False,  # Development Mode, saves some intermediate results to the /dev directory
        "BenchMode": False,  # Benchmark Mode
        "Verbose": True,  # Print Status Messages?
        "MatchingMetric": "euclidean",
        "Matching": "NearestNeighbor"
    }

    ENC_CONFIG = {
        "AliceAlgo": "TwoStepHash",
        "AliceSecret": "SuperSecretSalt1337",
        "AliceBFLength": 1024,
        "AliceBits": 30, # BF: 30, TMH: 1000
        "AliceN": 2,
        "AliceMetric": "dice",
        "EveAlgo": "TwoStepHash",
        "EveSecret": "ATotallyDifferentString",
        "EveBFLength": 1024,
        "EveBits": 30, # BF: 30, TMH: 1000
        "EveN": 2,
        "EveMetric": "dice",
        # For TMH encoding
        "AliceTables": 8,
        "AliceKeyLen": 8,
        "AliceValLen": 128,
        "EveTables": 8,
        "EveKeyLen": 8,
        "EveValLen": 128,
        # For 2SH encoding
        "AliceNHashFunc": 10,
        "AliceNHashCol": 1000,
        "AliceRandMode": "PNG",
        "EveNHashFunc": 10,
        "EveNHashCol": 1000,
        "EveRandMode": "PNG"
    }

    EMB_CONFIG = {
        "Algo": "Node2Vec",
        "AliceQuantile": 0.1,
        "AliceDiscretize": False,
        "AliceDim": 128,
        "AliceContext": 10,
        "AliceNegative": 1,
        "AliceNormalize": True,
        "EveQuantile": 0.1,
        "EveDiscretize": False,
        "EveDim": 128,
        "EveContext": 10,
        "EveNegative": 1,
        "EveNormalize": True,
        "Workers": -1,
        # For Node2Vec
        "AliceWalkLen":100,
        "AliceNWalks": 20,
        "AliceP": 250, #0.5
        "AliceQ": 300,    #2
        "AliceEpochs": 5,
        "AliceSeed": 42,
        "EveWalkLen": 100,
        "EveNWalks": 20,
        "EveP": 250, #0.5
        "EveQ": 300, #2
        "EveEpochs": 5,
        "EveSeed": 42,
    }

    ALIGN_CONFIG = {
        "RegWS": "Auto",
        "RegInit": 0.25,# For BF 1
        "Batchsize": "Auto",
        "LR": 300.0,
        "NIterWS": 5,
        "NIterInit": 50,  # 800
        "NEpochWS": 200,
        "LRDecay": 0.9,
        "Sqrt": False,
        "EarlyStopping": 20,
        "Selection": "None",
        "Wasserstein": True,
    }

    mp = run(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG)
