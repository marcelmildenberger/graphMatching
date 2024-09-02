import os
import hickle as hkl
import pickle
import random
import time

import numpy as np

from hashlib import md5

from aligners.closed_form_procrustes import ProcrustesAligner
from aligners.wasserstein_procrustes import WassersteinAligner
from embedders.node2vec import N2VEmbedder
from embedders.netmf import NetMFEmbedder
from encoders.bf_encoder import BFEncoder
from encoders.tmh_encoder import TMHEncoder
from encoders.tsh_encoder import TSHEncoder
from encoders.non_encoder import NonEncoder
from matchers.bipartite import GaleShapleyMatcher, SymmetricMatcher, MinWeightMatcher

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

    ##############################################
    #    ENCODING/SIMILARITY GRAPH GENERATION    #
    ##############################################

    # Check if Alice's data has been encoded before. If yes, load stored data.
    if os.path.isfile("./data/encoded/alice-%s.h5" % alice_enc_hash):
        if GLOBAL_CONFIG["Verbose"]:
            print("Found stored data for Alice's encoded records")

        # Loads the pairwise similarities of the encoded records from disk. Similarities are stored as single-precision
        # floats to save memory.
        alice_enc = hkl.load("./data/encoded/alice-%s.h5" % alice_enc_hash).astype(np.float32)
        # First row contains the number of records initially present in Alice's dataset. This is explicitly stored to
        # avoid re-calculating it from the pairwise similarities.
        # Extract the value and omit first row.
        n_alice = int(alice_enc[0][2])
        alice_enc = alice_enc[1:]
        # If records were dropped from both datasets, we load the number of overlapping records. This is required
        # to correctly compute the success rate later on
        if GLOBAL_CONFIG["DropFrom"] == "Both":
            with open("./data/encoded/overlap-%s.pck" % alice_enc_hash, "rb") as f:
                overlap_count = pickle.load(f)

        # Set duration of encoding to -1 to indicate that stored records were used (Only relevant when benchmarking)
        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_alice_enc = -1

    else:
        # If no pre-computed encoding are found, load and encode Alice's Data
        if GLOBAL_CONFIG["Verbose"]:
            print("Loading Alice's data")

        alice_data, alice_uids = read_tsv(GLOBAL_CONFIG["Data"])

        if GLOBAL_CONFIG["DropFrom"] == "Both":
            # Compute the maximum number of overlapping records possible for the given dataset size and overlap
            overlap_count = int(-(GLOBAL_CONFIG["Overlap"] * len(alice_data) / (GLOBAL_CONFIG["Overlap"] - 2)))
            with open("./data/encoded/overlap-%s.pck" % alice_enc_hash, "wb") as f:
                pickle.dump(overlap_count, f, protocol=5)
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
            # Shuffle because otherwise the order of the overlapping rows would be identical for Eve's and
            # Alice's data.
            selected_alice = random.sample(selected_alice, len(selected_alice))

        else:
            # Randomly select the rows held by Alice. If we drop from Eve, Alice holds all (100%) of the records.
            # In this case the selection is essentially a random shuffle of the rows.
            alice_ratio = GLOBAL_CONFIG["Overlap"] if GLOBAL_CONFIG["DropFrom"] == "Alice" else 1
            selected_alice = random.sample(range(len(alice_data)), int(alice_ratio * len(alice_data)))

        # Sampling was done using the row indices. Now we have to build the actual dataset.
        alice_data = [alice_data[i] for i in selected_alice]
        alice_uids = [alice_uids[i] for i in selected_alice]
        n_alice = len(alice_uids)  # Initial number of records in alice's dataset. Required for success calculation

        # Start timer for measuring the duration of the encoding of Alice's data.
        if GLOBAL_CONFIG["BenchMode"]:
            start_alice_enc = time.time()

        # Define the encoder to be used for Alice's data.
        ##############################
        # ADD FUTURE EXTENSIONS HERE #
        ##############################
        if ENC_CONFIG["AliceAlgo"] == "BloomFilter":
            alice_encoder = BFEncoder(ENC_CONFIG["AliceSecret"], ENC_CONFIG["AliceBFLength"],
                                    ENC_CONFIG["AliceBits"], ENC_CONFIG["AliceN"], ENC_CONFIG["AliceDiffuse"],
                                    ENC_CONFIG["AliceEldLength"], ENC_CONFIG["AliceT"])
        elif ENC_CONFIG["AliceAlgo"] == "TabMinHash":
            alice_encoder = TMHEncoder(ENC_CONFIG["AliceNHash"], ENC_CONFIG["AliceNHashBits"],
                                    ENC_CONFIG["AliceNSubKeys"], ENC_CONFIG["AliceN"],
                                    ENC_CONFIG["Alice1BitHash"],
                                    random_seed=ENC_CONFIG["AliceSecret"], verbose=GLOBAL_CONFIG["Verbose"],
                                        workers=GLOBAL_CONFIG["Workers"])
        elif ENC_CONFIG["AliceAlgo"] == "TwoStepHash":
            alice_encoder = TSHEncoder(ENC_CONFIG["AliceNHashFunc"], ENC_CONFIG["AliceNHashCol"], ENC_CONFIG["AliceN"],
                                    ENC_CONFIG["AliceRandMode"], secret=ENC_CONFIG["AliceSecret"],
                                    verbose=GLOBAL_CONFIG["Verbose"], workers=GLOBAL_CONFIG["Workers"])
        else:
            alice_encoder = NonEncoder(ENC_CONFIG["AliceN"])

        if GLOBAL_CONFIG["Verbose"]:
            print("Encoding Alice's Data")

        # Encode Alice's data and compute pairwise similarities of the encodings.
        # Result is a Float32 Numpy-Array of form [(UID1, UID2, Sim),...]
        alice_enc = alice_encoder.encode_and_compare(alice_data, alice_uids, metric=ENC_CONFIG["AliceMetric"], sim=True,
                                                        store_encs=GLOBAL_CONFIG["SaveAliceEncs"])

        del alice_data

        # Compute duration of Alice's encoding
        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_alice_enc = time.time() - start_alice_enc

        # Optionally export the pairwise similarities of the encodings as a human-readable edgelist (Tab separated)
        if GLOBAL_CONFIG["DevMode"]:
            np.savetxt("dev/alice.edg", alice_enc, delimiter="\t", fmt=["%1.0f", "%1.0f", "%1.16f"])

        if GLOBAL_CONFIG["Verbose"]:
            print("Done encoding Alice's data")

        # Prepend the initial number of records in Alice's dataset to the similarities and save them to disk.
        # Uses HDF Format for increased performance.
        # TODO: np.vstack tends to be slow for large arrays. Maybe replace with something faster/save in own file
        hkl.dump(np.vstack([np.array([-1, -1, n_alice]).astype(np.float32), alice_enc]),
                 "./data/encoded/alice-%s.h5" % alice_enc_hash, mode='w')

    if GLOBAL_CONFIG["Verbose"]:
        print("Computing Thresholds and subsetting data for Alice")
    # Compute the threshold value for subsetting: Only keep the X% highest similarities.
    tres = np.quantile(alice_enc[:,2], EMB_CONFIG["AliceQuantile"])
    # Only keep edges if their similarity is above the threshold
    alice_enc = alice_enc[(alice_enc[:, 2] > tres), :]

    # Discretize the data, i.e. replace all similarities with 1 (thus creating an unweighted graph)
    if EMB_CONFIG["AliceDiscretize"]:
        alice_enc[:, 2] = 1.0

    if GLOBAL_CONFIG["Verbose"]:
        print("Done processing Alice's data.")

    # Check if Eve's data has been encoded before. If yes, load stored data.
    if os.path.isfile("./data/encoded/eve-%s.h5" % eve_enc_hash):
        if GLOBAL_CONFIG["Verbose"]:
            print("Found stored data for Eve's encoded records")

        # Loads the pairwise similarities of the encoded records from disk. Similarities are stored as single-precision
        # floats to save memory.
        eve_enc = hkl.load("./data/encoded/eve-%s.h5" % eve_enc_hash).astype(np.float32)
        # First row contains the number of records initially present in Eve's dataset. This is explicitly stored to
        # avoid re-calculating it from the pairwise similarities.
        # Extract the value and omit first row.
        n_eve = int(eve_enc[0][2])
        eve_enc = eve_enc[1:]
        # Set duration of encoding to -1 to indicate that stored records were used (Only relevant when benchmarking)
        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_eve_enc = -1
    else:
        # If no pre-computed encoding are found, load and encode Eve's Data
        if GLOBAL_CONFIG["Verbose"]:
            print("Loading Eve's data")
        eve_data, eve_uids = read_tsv(GLOBAL_CONFIG["Data"])

        # If records are dropped from both datasets, Eve's dataset consists of the overlapping records and the
        # available records, i.e. those records that have not been added to Alice's dataset.
        if GLOBAL_CONFIG["DropFrom"] == "Both":
            selected_eve = selected_overlap + available
            # Randomly shuffle the rows to avoid unintentionally leaking ground truth
            selected_eve = random.sample(selected_eve, len(selected_eve))
        else:
            # Randomly select the rows held by Eve. If we drop from Alice, Eve holds all (100%) of the records.
            # In this case the selection is essentially a random shuffle of the rows.
            eve_ratio = GLOBAL_CONFIG["Overlap"] if GLOBAL_CONFIG["DropFrom"] == "Eve" else 1
            selected_eve = random.sample(range(len(eve_data)), int(eve_ratio * len(eve_data)))

        # Sampling was done using the row indices. Now we have to build the actual dataset.
        eve_data = [eve_data[i] for i in selected_eve]
        eve_uids = [eve_uids[i] for i in selected_eve]
        n_eve = len(eve_uids)

        # Start timer for measuring the duration of the encoding of Alice's data.
        if GLOBAL_CONFIG["BenchMode"]:
            start_eve_enc = time.time()

        # Define the encoder to be used for Eve's data.
        ##############################
        # ADD FUTURE EXTENSIONS HERE #
        ##############################

        if ENC_CONFIG["EveAlgo"] == "BloomFilter":
            eve_encoder = BFEncoder(ENC_CONFIG["EveSecret"], ENC_CONFIG["EveBFLength"],
                                    ENC_CONFIG["EveBits"], ENC_CONFIG["EveN"], ENC_CONFIG["EveDiffuse"],
                                    ENC_CONFIG["EveEldLength"], ENC_CONFIG["EveT"])
        elif ENC_CONFIG["EveAlgo"] == "TabMinHash":
            eve_encoder = TMHEncoder(ENC_CONFIG["EveNHash"], ENC_CONFIG["EveNHashBits"],
                                    ENC_CONFIG["EveNSubKeys"], ENC_CONFIG["EveN"],
                                    ENC_CONFIG["Eve1BitHash"],
                                    random_seed=ENC_CONFIG["EveSecret"], verbose=GLOBAL_CONFIG["Verbose"],
                                        workers=GLOBAL_CONFIG["Workers"])
        elif ENC_CONFIG["EveAlgo"] == "TwoStepHash":
            eve_encoder = TSHEncoder(ENC_CONFIG["EveNHashFunc"], ENC_CONFIG["EveNHashCol"], ENC_CONFIG["EveN"],
                                    ENC_CONFIG["EveRandMode"], secret=ENC_CONFIG["EveSecret"],
                                    verbose=GLOBAL_CONFIG["Verbose"])
        else:
            eve_encoder = NonEncoder(ENC_CONFIG["EveN"])

        if GLOBAL_CONFIG["Verbose"]:
            print("Encoding Eve's Data")

        # Encode Alice's data and compute pairwise similarities of the encodings.
        # Result is a Float32 Numpy-Array of form [(UID1, UID2, Sim),...]

        eve_enc = eve_encoder.encode_and_compare(eve_data, eve_uids, metric=ENC_CONFIG["EveMetric"], sim=True,
                                                 store_encs=GLOBAL_CONFIG["SaveEveEncs"])
        del eve_data

        # Compute duration of Alice's encoding
        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_eve_enc = time.time() - start_eve_enc

        # Optionally export the pairwise similarities of the encodings as a human-readable edgelist (Tab separated)
        if GLOBAL_CONFIG["DevMode"]:
            np.savetxt("dev/eve.edg", eve_enc, delimiter="\t", fmt=["%1.0f", "%1.0f", "%1.16f"])

        if GLOBAL_CONFIG["Verbose"]:
            print("Done encoding Eve's data")

        # Prepend the initial number of records in Eve's dataset to the similarities and save them to disk.
        # Uses HDF Format for increased performance.
        # TODO: np.vstack tends to be slow for large arrays. Maybe replace with something faster/save in own file
        hkl.dump(np.vstack([np.array([-1, -1, n_eve]).astype(np.float32), eve_enc]),
                 "./data/encoded/eve-%s.h5" % eve_enc_hash, mode='w')

    if GLOBAL_CONFIG["Verbose"]:
        print("Computing Thresholds and subsetting data for Eve")

    # Compute the threshold value for subsetting: Only keep the X% highest similarities.
    tres = np.quantile(eve_enc[:, 2], EMB_CONFIG["EveQuantile"])
    eve_enc = eve_enc[(eve_enc[:, 2] > tres), :]

    # Optionally sets all remaining similarities to 1, essentially creating an unweighted graph.
    if EMB_CONFIG["EveDiscretize"]:
        eve_enc[:, 2] = 1.0

    if GLOBAL_CONFIG["Verbose"]:
        print("Done processing Eve's data.")

    ###################
    #    EMBEDDING    #
    ###################

    # Start timer to measure duration of embedding for Alice's data
    if GLOBAL_CONFIG["BenchMode"]:
        start_alice_emb = time.time()

    # Check if data has been embedded before. If yes, load stored embeddings from disk.
    if os.path.isfile("./data/embeddings/alice-%s.h5" % alice_emb_hash):
        if GLOBAL_CONFIG["Verbose"]:
            print("Found stored data for Alice's embeddings")

        # If embeddings are present, load them
        alice_embeddings = hkl.load("./data/embeddings/alice-%s.h5" % alice_emb_hash).astype(np.float32)

        # Loads the UIDs of the embeddings
        with open("./data/embeddings/alice_uids-%s.pck" % alice_emb_hash, "rb") as f:
            alice_uids = pickle.load(f)

        # Set embedding duration to -1 to inidicate that pre-computed embeddings were used.
        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_alice_emb = -1

    else:
        # If no pre-computed embeddings are found, embed the encoded data
        if GLOBAL_CONFIG["Verbose"]:
            print("Embedding Alice's data. This may take a while...")

        # Define the embedding algorithm to be used for Alice's data.
        ##############################
        # ADD FUTURE EXTENSIONS HERE #
        ##############################

        if EMB_CONFIG["Algo"] == "Node2Vec":
            # PecanPy expects an edgelist on disk: Save similarities to edgelist format
            # TODO: Check if we can somehow pass the data directly to PecanPy
            np.savetxt("data/edgelists/alice.edg", alice_enc, delimiter="\t", fmt=["%1.0f", "%1.0f", "%1.16f"])

            alice_embedder = N2VEmbedder(walk_length=EMB_CONFIG["AliceWalkLen"], n_walks=EMB_CONFIG["AliceNWalks"],
                                         p=EMB_CONFIG["AliceP"], q=EMB_CONFIG["AliceQ"],
                                         dim_embeddings=EMB_CONFIG["AliceDim"],
                                         context_size=EMB_CONFIG["AliceContext"], epochs=EMB_CONFIG["AliceEpochs"],
                                         seed=EMB_CONFIG["AliceSeed"], workers=GLOBAL_CONFIG["Workers"],
                                         verbose=GLOBAL_CONFIG["Verbose"])
            alice_embedder.train("./data/edgelists/alice.edg")
        #elif EMB_CONFIG["Explicit"] == "Node2Vec":
        else:
            alice_embedder = NetMFEmbedder(EMB_CONFIG["AliceDim"], EMB_CONFIG["AliceContext"],
                                           EMB_CONFIG["AliceNegative"],
                                           EMB_CONFIG["AliceNormalize"])

            alice_embedder.train(alice_enc)



        # Compute the duration of the embedding
        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_alice_emb = time.time() - start_alice_emb

        # Optionally save trained model for further inspection
        if GLOBAL_CONFIG["DevMode"]:
            alice_embedder.save_model("./dev", "alice.mod")

        if GLOBAL_CONFIG["Verbose"]:
            print("Done embedding Alice's data.")

        # We have to redefine the uids to account for the fact that nodes might have been dropped while ensuring minimum
        # similarity.
        alice_embeddings, alice_uids = alice_embedder.get_vectors()
        del alice_embedder

        # Save Embeddings and UIDs to disk (rows in embedding matrix are ordered according to the uids)
        hkl.dump(alice_embeddings, "./data/embeddings/alice-%s.h5" % alice_emb_hash, mode='w')
        with open("./data/embeddings/alice_uids-%s.pck" % alice_emb_hash, "wb") as f:
            pickle.dump(alice_uids, f, protocol=5)

    # Create a dictionary that maps UIDs to their respective row index (Only used if alignment using ground truth is
    # selected)
    alice_indexdict = dict(zip(alice_uids, range(len(alice_uids))))

    # Check if Eve's data has been embedded before. If yes, load stored embeddings from disk.
    if os.path.isfile("./data/embeddings/eve-%s.h5" % eve_emb_hash):
        if GLOBAL_CONFIG["Verbose"]:
            print("Found stored data for Eve's embeddings")

        # If embeddings are present, load them. Single precision floats to save memory.
        eve_embeddings = hkl.load("./data/embeddings/eve-%s.h5" % eve_emb_hash).astype(np.float32)

        # Loads the UIDs of the embeddings
        with open("./data/embeddings/eve_uids-%s.pck" % eve_emb_hash, "rb") as f:
            eve_uids = pickle.load(f)

        # Set embedding duration to -1 to inidicate that pre-computed embeddings were used.
        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_eve_emb = -1

    else:
        # If no pre-computed embeddings are found, embed the encoded data
        # Start timer
        if GLOBAL_CONFIG["BenchMode"]:
            start_eve_emb = time.time()

        if GLOBAL_CONFIG["Verbose"]:
            print("Embedding Eve's data. This may take a while...")

        # Define the embedding algorithm to be used for Eve's data.
        ##############################
        # ADD FUTURE EXTENSIONS HERE #
        ##############################

        if EMB_CONFIG["Algo"] == "Node2Vec":
            # PecanPy expects an edgelist on disk: Save similarities to edgelist format
            # TODO: Check if we can somehow pass the data directly to PecanPy

            np.savetxt("data/edgelists/eve.edg", eve_enc, delimiter="\t", fmt=["%1.0f", "%1.0f", "%1.16f"])

            eve_embedder = N2VEmbedder(walk_length=EMB_CONFIG["EveWalkLen"], n_walks=EMB_CONFIG["EveNWalks"],
                                       p=EMB_CONFIG["EveP"], q=EMB_CONFIG["EveQ"], dim_embeddings=EMB_CONFIG["EveDim"],
                                       context_size=EMB_CONFIG["EveContext"], epochs=EMB_CONFIG["EveEpochs"],
                                       seed=EMB_CONFIG["EveSeed"], workers=GLOBAL_CONFIG["Workers"],
                                       verbose=GLOBAL_CONFIG["Verbose"])
            eve_embedder.train("./data/edgelists/eve.edg")
        else:
            eve_embedder = NetMFEmbedder(EMB_CONFIG["EveDim"], EMB_CONFIG["EveContext"],
                                         EMB_CONFIG["EveNegative"],
                                         EMB_CONFIG["EveNormalize"])
            eve_embedder.train(eve_enc)

        # Compute the duration of the embedding
        if GLOBAL_CONFIG["BenchMode"]:
            elapsed_eve_emb = time.time() - start_eve_emb

        # Optionally save trained model for further inspection
        if GLOBAL_CONFIG["DevMode"]:
            eve_embedder.save_model("./dev", "eve.mod")

        if GLOBAL_CONFIG["Verbose"]:
            print("Done embedding Eve's data.")

        # We have to redefine the uids to account for the fact that nodes might have been dropped while ensuring minimum
        # similarity.
        eve_embeddings, eve_uids = eve_embedder.get_vectors()

        del eve_embedder

        # Save Embeddings and UIDs to disk (rows in embedding matrix are ordered according to the uids)
        hkl.dump(eve_embeddings, "./data/embeddings/eve-%s.h5" % eve_emb_hash, mode='w')
        with open("./data/embeddings/eve_uids-%s.pck" % eve_emb_hash, "wb") as f:
            pickle.dump(eve_uids, f, protocol=5)

    # Create a dictionary that maps UIDs to their respective row index (Only used if alignment using ground truth is
    # selected)
    eve_indexdict = dict(zip(eve_uids, range(len(eve_uids))))

    #############################
    #    EMBEDDING ALIGNMENT    #
    #############################

    # Start the timer for measuring the duration of alignment
    if GLOBAL_CONFIG["BenchMode"]:
        start_align_prep = time.time()

    # Select the Data to be used for alignment:
    # GroundTruth:  If Ground Truth is known, the first "MaxLoad" UIDs of Alices Data are selected. The corresponding
    #               embeddings of Alice and Eve added to the alignment dataset. This results in two equally sized
    #               lists (one for ALice, one for Eve) of 1D Arrays. In both lists, the sameindices refer to the
    #               embeddings of the same UID, thus allowing alignment via orthogonal procrustes.
    #
    # Random:       Randomly selects "MaxLoad" records from Alice's and Eve's data. This results in to equally shaped
    #               matrices, however, there is no guaranteed correspondence of the rows.
    #
    # None/Else:    Use entire datasets for alignment

    if ALIGN_CONFIG["Selection"] == "GroundTruth":
        alice_sub = alice_embeddings[[alice_indexdict[k] for k in alice_uids[:ALIGN_CONFIG["MaxLoad"]]], :]
        eve_sub = eve_embeddings[[eve_indexdict[k] for k in alice_uids[:ALIGN_CONFIG["MaxLoad"]]], :]

    elif ALIGN_CONFIG["Selection"] == "Random":
        eve_sub = eve_embeddings[
                  np.random.choice(eve_embeddings.shape[0], ALIGN_CONFIG["MaxLoad"], replace=False), :]
        alice_sub = alice_embeddings[
                    np.random.choice(alice_embeddings.shape[0], ALIGN_CONFIG["MaxLoad"], replace=False), :]

    else:
        alice_sub = alice_embeddings
        eve_sub = eve_embeddings


    # Sets the Batchsize: "Auto" sets it to 85% of the smaller dataset. Numbers smaller or equal to 1 are interpreted
    # as percentages of the smaller dataset. Batchsize is capped to 20,000.

    if ALIGN_CONFIG["Batchsize"] == "Auto":
        bs = min(len(alice_sub), len(eve_sub))
        bs = int(0.85 * bs)
        ALIGN_CONFIG["Batchsize"] = bs

    if ALIGN_CONFIG["Batchsize"] <= 1:
        bs = int(ALIGN_CONFIG["Batchsize"]*min(len(alice_sub), len(eve_sub)))
        ALIGN_CONFIG["Batchsize"] = bs

    ALIGN_CONFIG["Batchsize"] = min(ALIGN_CONFIG["Batchsize"], 20000)

    del alice_enc, eve_enc

    # Adjust data format (Turn list of 1D-Arrays into 2D-Array).

    if ALIGN_CONFIG["Selection"] in ["GroundTruth"]:
        alice_sub = np.stack(alice_sub, axis=0)
        eve_sub = np.stack(eve_sub, axis=0)

    # Calculate duration of preprocessing for alignment
    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_align_prep = time.time() - start_align_prep
        start_align = time.time()

    if GLOBAL_CONFIG["Verbose"]:
        print("Aligning vectors. This may take a while.")

    # Define alignment methods
    if ALIGN_CONFIG["Wasserstein"]:
        # Heuristically sets regularization if not specified otherwise
        if ALIGN_CONFIG["RegWS"] == "Auto":
            if ENC_CONFIG["EveAlgo"] == "TwoStepHash" or ENC_CONFIG["AliceAlgo"] == "TwoStepHash":
                ALIGN_CONFIG["RegWS"] = 0.1
            else:
                ALIGN_CONFIG["RegWS"] = 0.05

        aligner = WassersteinAligner(ALIGN_CONFIG["RegInit"], ALIGN_CONFIG["RegWS"],
                                     ALIGN_CONFIG["Batchsize"], ALIGN_CONFIG["LR"], ALIGN_CONFIG["NIterInit"],
                                     ALIGN_CONFIG["NIterWS"], ALIGN_CONFIG["NEpochWS"],
                                     ALIGN_CONFIG["LRDecay"], ALIGN_CONFIG["Sqrt"], ALIGN_CONFIG["EarlyStopping"],
                                     verbose=GLOBAL_CONFIG["Verbose"])
    else:
        aligner = ProcrustesAligner()

    # Compute transformation matrix
    transformation_matrix = aligner.align(alice_sub, eve_sub)
    # Projects Eve's embeddings into Alice's space by multiplying Eve's embeddings with the transformation matrix.
    eve_embeddings = np.dot(eve_embeddings, transformation_matrix.T)

    # Compute duration of alignment.
    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_align = time.time() - start_align

    if GLOBAL_CONFIG["Verbose"]:
        print("Done.")
        print("Performing bipartite graph matching")

    if GLOBAL_CONFIG["BenchMode"]:
        start_mapping = time.time()

    # Creates the "matcher" object responsible for the bipartite graph matching.
    # MinWeight:    Minimum Weight bipartite matching: Finds a full 1-to-1 mapping such that the overall sum of weights
    #               (distances between nodes) is minimized.
    #
    # Stable:       Computes a 1-to-1 matching by solving the stable marriage problem:
    #               https://en.wikipedia.org/wiki/Stable_marriage_problem
    #
    # Symmetric:    Computes a symmetric 1-to-1 matching. Two nodes A and B are matched if and only if sim(A,B) is the
    #               highest similarity of A to any other node AND of B to any other node. This does not guarantee a full
    #               matching.
    #
    # NearestNeigbor:   Matches each node to its closest neighbor (i.e. the one with the lowest distance). This is
    #                   considerably more efficient than the bipartite matchings, especially on larger datasets.
    #                   However, it does not guarantee 1-to-1 mappings.
    if GLOBAL_CONFIG["Matching"] == "MinWeight":
        matcher = MinWeightMatcher(GLOBAL_CONFIG["MatchingMetric"])
    elif GLOBAL_CONFIG["Matching"] == "Stable":
        matcher = GaleShapleyMatcher(GLOBAL_CONFIG["MatchingMetric"])
    elif GLOBAL_CONFIG["Matching"] == "Symmetric":
        matcher = SymmetricMatcher(GLOBAL_CONFIG["MatchingMetric"])
    elif GLOBAL_CONFIG["Matching"] == "NearestNeighbor":
        matcher = NNMatcher(GLOBAL_CONFIG["MatchingMetric"])

    # Compute the mapping. Results in a list of the form [("S_1","L_2"),...], where "L_XXX" represents the UIDs in the
    # larger dataset and "S_XXX" represents the UIDs in the smaller dataset.
    # Note that mappings are included twice: Once as a mapping from S to L and once fom L to S.
    # These redundant mappings must be ignored when computing the success rate.
    mapping = matcher.match(alice_embeddings, alice_uids, eve_embeddings, eve_uids)

    # Compute durations of mapping and overall attack.
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

        if not os.path.isfile("data/benchmark.tsv"):
            save_tsv([keys], "data/benchmark.tsv")

        save_tsv([vals], "data/benchmark.tsv", mode="a")

    return mapping


if __name__ == "__main__":
    # Parameters for running the GMA
    # See ./docs/parameters.md for details.

    GLOBAL_CONFIG = {
        "Data": "./data/titanic_full.tsv",
        "Overlap": 1,
        "DropFrom": "Alice",
        "DevMode": False,  # Development Mode, saves some intermediate results to the /dev directory
        "BenchMode": False,  # Benchmark Mode
        "Verbose": True,  # Print Status Messages?
        "MatchingMetric": "cosine",
        "Matching": "MinWeight",
        "Workers": -1,
        "SaveAliceEncs": False,
        "SaveEveEncs": False
    }

    ENC_CONFIG = {
        "AliceAlgo": "TwoStepHash",
        "AliceSecret": "SuperSecretSalt1337",
        "AliceN": 2,
        "AliceMetric": "dice",
        "EveAlgo": None,
        "EveSecret": "ATotallyDifferentString42",
        "EveN": 2,
        "EveMetric": "dice",
        # For BF encoding
        "AliceBFLength": 1024,
        "AliceBits": 10,
        "AliceDiffuse": False,
        "AliceT": 10,
        "AliceEldLength": 1024,
        "EveBFLength": 1024,
        "EveBits": 10,
        "EveDiffuse": False,
        "EveT": 10,
        "EveEldLength": 1024,
        # For TMH encoding
        "AliceNHash": 1024,
        "AliceNHashBits": 64,
        "AliceNSubKeys": 8,
        "Alice1BitHash": True,
        "EveNHash": 1024,
        "EveNHashBits": 64,
        "EveNSubKeys": 8,
        "Eve1BitHash": True,
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
        # For Node2Vec
        "AliceWalkLen": 100,
        "AliceNWalks": 20,
        "AliceP": 250,
        "AliceQ": 300,
        "AliceEpochs": 5,
        "AliceSeed": 42,
        "EveWalkLen": 100,
        "EveNWalks": 20,
        "EveP": 250,
        "EveQ": 300,
        "EveEpochs": 5,
        "EveSeed": 42
    }

    ALIGN_CONFIG = {
        "RegWS": max(0.1, GLOBAL_CONFIG["Overlap"]/2), #0005
        "RegInit":1, # For BF 0.25
        "Batchsize": 1, # 1 = 100%
        "LR": 200.0,
        "NIterWS": 100,
        "NIterInit": 5 ,  # 800
        "NEpochWS": 100,
        "LRDecay": 1,
        "Sqrt": True,
        "EarlyStopping": 10,
        "Selection": "None",
        "MaxLoad": None,
        "Wasserstein": True
    }
    mp = run(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG)
