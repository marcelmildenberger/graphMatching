from encoders.bf_encoder import BFEncoder
from encoders.tmh_encoder import TMHEncoder
from encoders.tsh_encoder import TSHEncoder
from encoders.non_encoder import NonEncoder
from embedders.explicit import ExplicitEmbedder
from blockers.minhash import MinHashLSH
from matchers.bipartite import GaleShapleyMatcher, SymmetricMatcher, MinWeightMatcher
from matchers.spatial import NNMatcher

from utils import *

import numpy as np


def run(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, BLOCKING_CONFIG):
    """
    This is a re-implementation of the Graph Matching Attack as described by Vidanage et al.
    https://doi.org/10.1145/3340531.3411931

    Note that this implementation is able to simulate a preliminary MinHashLSH blocking on the plaintext data
    prior to computing the embeddings. This is a step only present in the reference implementation of Vidanage et al.
    and not described in the original paper.
    If you wish to disable this blocking step, set the Disable parameter in BLOCKING_CONFIG to True.

    :param GLOBAL_CONFIG: Global Parameters
    :param ENC_CONFIG: Encoding Parameters
    :param EMB_CONFIG: Embedding Parameters
    :param BLOCKING_CONFIG: Blocking Parameters
    :return: None
    """

    QG_min_hash = MinHashLSH(BLOCKING_CONFIG["PlainSampleSize"], BLOCKING_CONFIG["PlainNumSamples"],
                             BLOCKING_CONFIG["AliceRandomSeed"])

    BA_min_hash = MinHashLSH(BLOCKING_CONFIG["PlainSampleSize"], BLOCKING_CONFIG["PlainNumSamples"],
                             BLOCKING_CONFIG["EveRandomSeed"])

    eve_data, eve_uids = read_tsv(GLOBAL_CONFIG["Data"])
    alice_data, alice_uids = read_tsv(GLOBAL_CONFIG["Data"])

    alice_ngrams = ["".join(d).replace(" ", "").lower() for d in alice_data]
    alice_ngrams = [[b[i:i + ENC_CONFIG["AliceN"]] for i in range(len(b) - ENC_CONFIG["AliceN"] + 1)] for b in
                    alice_ngrams]

    eve_ngrams = ["".join(d).replace(" ", "").lower() for d in eve_data]
    eve_ngrams = [[b[i:i + ENC_CONFIG["EveN"]] for i in range(len(b) - ENC_CONFIG["EveN"] + 1)] for b in eve_ngrams]

    if not BLOCKING_CONFIG["Disable"]:
        alice_uids_to_blocks, alice_blocks_to_uids = create_blocks(alice_uids, alice_ngrams, QG_min_hash)
        eve_uids_to_blocks, eve_blocks_to_uids = create_blocks(eve_uids, eve_ngrams, BA_min_hash)

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

    alice_enc = alice_encoder.encode_and_compare(alice_data, alice_uids, metric=ENC_CONFIG["AliceMetric"], sim=True)
    eve_enc = eve_encoder.encode_and_compare(eve_data, eve_uids, metric=ENC_CONFIG["EveMetric"], sim=True)

    # Only keep edged of nodes that belong to the same block.
    # CAUTION: This leaks ground truth to the attacker!
    if not BLOCKING_CONFIG["Disable"]:
        keep_alice = simulate_blocking(alice_enc, alice_uids_to_blocks)
        keep_eve = simulate_blocking(eve_enc, eve_uids_to_blocks)
        alice_enc = alice_enc[keep_alice, :]
        eve_enc = eve_enc[keep_eve, :]

    alice_enc = alice_enc[(alice_enc[:, 2] > EMB_CONFIG["MinSim"]), :]
    eve_enc = eve_enc[(eve_enc[:, 2] > EMB_CONFIG["MinSim"]), :]

    np.savetxt("././graphMatching/data/edgelists/alice.edg", alice_enc, delimiter="\t", fmt=["%1.0f", "%1.0f", "%1.16f"])
    np.savetxt("././graphMatching/data/edgelists/eve.edg", eve_enc, delimiter="\t", fmt=["%1.0f", "%1.0f", "%1.16f"])

    alice_embedder = ExplicitEmbedder(alice_enc, alice_uids,
                                      min_component_size=EMB_CONFIG["MinComponentSize"],
                                      verbose=GLOBAL_CONFIG["Verbose"])

    eve_embedder = ExplicitEmbedder(eve_enc, eve_uids,
                                    min_component_size=EMB_CONFIG["MinComponentSize"], verbose=GLOBAL_CONFIG["Verbose"])

    alice_embedder.train("././graphMatching/data/edgelists/alice.edg")
    eve_embedder.train("././graphMatching/data/edgelists/eve.edg")

    hist_features = max(alice_embedder.max_log_degree, eve_embedder.max_log_degree)

    alice_embedder.set_hist_features(hist_features)
    eve_embedder.set_hist_features(hist_features)

    alice_emb, alice_uids = alice_embedder.get_vectors()
    eve_emb, eve_uids = eve_embedder.get_vectors()

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
    mapping = matcher.match(alice_emb, alice_uids, eve_emb, eve_uids)

    correct = 0
    for smaller, larger in mapping.items():
        if smaller[0] == "L":
            continue
        if smaller[1:] == larger[1:]:
            correct += 1

    print("Correct: " + str(correct) + " of " + str(len(alice_uids)))
    print("Success Rate: " + str(correct/len(alice_uids)))

if __name__ == "__main__":
    GLOBAL_CONFIG = {
        "Data": "./data/titanic_full.tsv",
        "Verbose": True,  # Print Status Messages?
        "MatchingMetric": "cosine",
        "Matching": "MinWeight"
    }

    ENC_CONFIG = {
        "AliceAlgo": "BloomFilter",
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
        "MinComponentSize": 5,
        "MinSim": 0.2
    }

    BLOCKING_CONFIG = {
        "Disable": False,
        "PlainSampleSize": 4,
        "PlainNumSamples": 50,
        "AliceRandomSeed": 42,
        "EveRandomSeed": 17
    }
    run(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, BLOCKING_CONFIG)