import os
import pickle
import random

import hickle as hkl
import numpy as np

from graphMatching.aligners.closed_form_procrustes import ProcrustesAligner
from graphMatching.aligners.wasserstein_procrustes import WassersteinAligner
from graphMatching.embedders.netmf import NetMFEmbedder
from graphMatching.embedders.node2vec import N2VEmbedder
from graphMatching.encoders.non_encoder import NonEncoder
from graphMatching.encoders.precomputed_binary_encoder import PrecomputedBinaryEncoder
from graphMatching.encoders.precomputed_set_encoder import PrecomputedSetEncoder
from graphMatching.encoders.precomputed_tmh_encoder import PrecomputedTMHEncoder
from graphMatching.matchers.bipartite import GaleShapleyMatcher, MinWeightMatcher, SymmetricMatcher
from graphMatching.matchers.spatial import NNMatcher
from utils.data_pipeline import read_tsv, resolve_encoded_dataset_path, save_tsv


def _load_overlap_count(enc_hash):
    with open("./graphMatching/data/encoded/overlap-%s.pck" % enc_hash, "rb") as f:
        return pickle.load(f)


def _save_overlap_count(enc_hash, overlap_count):
    with open("./graphMatching/data/encoded/overlap-%s.pck" % enc_hash, "wb") as f:
        pickle.dump(overlap_count, f, protocol=5)


def _load_source_dataset(data_path, algo, diffuse=False):
    if algo not in {"None", None}:
        encoded_path = resolve_encoded_dataset_path(data_path, algo, diffuse=diffuse)
        return (*read_tsv(encoded_path, skip_header=False), True)
    return (*read_tsv(data_path, skip_header=False), False)


def _build_uid_row_map(data, uids):
    uid_row_map = {}
    for row, uid in zip(data, uids):
        normalized_uid = str(uid)
        if normalized_uid in uid_row_map:
            raise ValueError(f"Duplicate uid encountered while preparing GMA data: {normalized_uid}")
        uid_row_map[normalized_uid] = row
    return uid_row_map


def _prepare_header(raw_header, algo, is_precomputed):
    header = list(raw_header)
    if is_precomputed:
        return header
    if algo not in {"None", None}:
        header.insert(-1, algo.lower())
    return header


def _create_encoder(party, GLOBAL_CONFIG, ENC_CONFIG, is_precomputed):
    algo = ENC_CONFIG[f"{party}Algo"]

    if is_precomputed:
        if algo in {"BloomFilter", "RSE"}:
            return PrecomputedBinaryEncoder()
        if algo == "TabMinHash":
            return PrecomputedTMHEncoder(one_bit_hash=ENC_CONFIG[f"{party}1BitHash"])
        if algo == "TwoStepHash":
            return PrecomputedSetEncoder(workers=GLOBAL_CONFIG["Workers"])
        raise ValueError(f"Unsupported precomputed encoder: {algo}")

    return NonEncoder(ENC_CONFIG[f"{party}N"])


def _select_uid_sets(common_uids, GLOBAL_CONFIG, rng):
    if not common_uids:
        raise ValueError("Graph matching requires at least one UID shared between Alice and Eve source data.")

    drop_from = GLOBAL_CONFIG["DropFrom"]
    overlap = GLOBAL_CONFIG["Overlap"]
    common_uids = list(common_uids)

    if drop_from == "Both":
        overlap_count = int(-(overlap * len(common_uids) / (overlap - 2)))
        selected_overlap = rng.sample(common_uids, overlap_count)
        remaining = [uid for uid in common_uids if uid not in selected_overlap]
        selected_alice_only = rng.sample(remaining, int((len(common_uids) - overlap_count) / 2))
        selected_eve_only = [uid for uid in remaining if uid not in selected_alice_only]
        selected_alice = rng.sample(
            selected_alice_only + selected_overlap,
            len(selected_alice_only) + len(selected_overlap),
        )
        selected_eve = rng.sample(
            selected_eve_only + selected_overlap,
            len(selected_eve_only) + len(selected_overlap),
        )
        return selected_alice, selected_eve, overlap_count

    alice_ratio = overlap if drop_from == "Alice" else 1
    eve_ratio = overlap if drop_from == "Eve" else 1
    selected_alice = rng.sample(common_uids, int(alice_ratio * len(common_uids)))
    selected_eve = rng.sample(common_uids, int(eve_ratio * len(common_uids)))
    return selected_alice, selected_eve, None


def _ensure_nonzero_similarity_graph(enc_sim, verbose, party_label):
    skip_thresholding = False
    if sum(enc_sim[:, 2]) == 0:
        if verbose:
            print(f"Warning: All edges in {party_label}'s similarity graph are Zero.")
        enc_sim[:, 2] = 0.5
        skip_thresholding = True
    return enc_sim, skip_thresholding


def _persist_similarity_cache(cache_path, n_records, enc_sim):
    hkl.dump(
        np.vstack([np.array([-1, -1, n_records]).astype(np.float32), enc_sim]),
        cache_path,
        mode="w",
    )


def _postprocess_similarity_graph(enc_sim, quantile, discretize, skip_thresholding):
    if not skip_thresholding:
        threshold = np.quantile(enc_sim[:, 2], quantile)
        enc_sim = enc_sim[(enc_sim[:, 2] > 0), :]
        enc_sim = enc_sim[(enc_sim[:, 2] >= threshold), :]

    if discretize:
        enc_sim[:, 2] = 1.0

    return enc_sim


def validate_gma_configuration(GLOBAL_CONFIG, ENC_CONFIG, ALIGN_CONFIG):
    supported_matchings = ["MinWeight", "Stable", "Symmetric", "NearestNeighbor"]
    assert GLOBAL_CONFIG["Matching"] in supported_matchings, "Error: Matching method must be one of %s" % (
        supported_matchings,
    )

    supported_selections = ["Degree", "GroundTruth", "Centroids", "Random", "None", None]
    assert ALIGN_CONFIG["Selection"] in supported_selections, (
        "Error: Selection method for alignment subset must be one of %s" % (supported_selections,)
    )

    supported_drops = ["Alice", "Eve", "Both"]
    assert GLOBAL_CONFIG["DropFrom"] in supported_drops, "Error: Data must be dropped from one of %s" % (
        supported_drops,
    )

    supported_encs = ["BloomFilter", "TabMinHash", "TwoStepHash", "RSE", "None", None]
    assert (
        ENC_CONFIG["AliceAlgo"] in supported_encs and ENC_CONFIG["EveAlgo"] in supported_encs
    ), "Error: Encoding method must be one of %s" % (supported_encs,)


def prepare_similarity_graph_inputs(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, eve_enc_hash, alice_enc_hash):
    alice_encoded_cache_path = "./graphMatching/data/encoded/alice-%s.h5" % alice_enc_hash
    eve_encoded_cache_path = "./graphMatching/data/encoded/eve-%s.h5" % eve_enc_hash
    need_source_data = not (os.path.isfile(alice_encoded_cache_path) and os.path.isfile(eve_encoded_cache_path))

    source_context = {}
    if need_source_data:
        alice_source_data, alice_source_uids, alice_raw_header, alice_is_precomputed = _load_source_dataset(
            GLOBAL_CONFIG["Data"],
            ENC_CONFIG["AliceAlgo"],
            diffuse=ENC_CONFIG.get("AliceDiffuse", False),
        )
        eve_source_data, eve_source_uids, eve_raw_header, eve_is_precomputed = _load_source_dataset(
            GLOBAL_CONFIG["Data"],
            ENC_CONFIG["EveAlgo"],
            diffuse=ENC_CONFIG.get("EveDiffuse", False),
        )

        alice_uid_map = _build_uid_row_map(alice_source_data, alice_source_uids)
        eve_uid_map = _build_uid_row_map(eve_source_data, eve_source_uids)
        common_uids = [uid for uid in alice_source_uids if str(uid) in eve_uid_map]
        selection_rng = random.Random(
            f"{alice_enc_hash}:{eve_enc_hash}:{GLOBAL_CONFIG['DropFrom']}:{GLOBAL_CONFIG['Overlap']}"
        )
        selected_alice_uids, selected_eve_uids, overlap_count = _select_uid_sets(common_uids, GLOBAL_CONFIG, selection_rng)

        source_context = {
            "alice_uid_map": alice_uid_map,
            "eve_uid_map": eve_uid_map,
            "selected_alice_uids": selected_alice_uids,
            "selected_eve_uids": selected_eve_uids,
            "overlap_count": overlap_count,
            "alice_is_precomputed": alice_is_precomputed,
            "eve_is_precomputed": eve_is_precomputed,
            "alice_header_prepared": _prepare_header(alice_raw_header, ENC_CONFIG["AliceAlgo"], alice_is_precomputed),
            "eve_header_prepared": _prepare_header(eve_raw_header, ENC_CONFIG["EveAlgo"], eve_is_precomputed),
        }

    reidentified_individuals_header = source_context.get("alice_header_prepared")
    not_reidentified_individuals_header = None
    if reidentified_individuals_header is not None:
        not_reidentified_individuals_header = reidentified_individuals_header[-2:]

    alice_skip_thresholding = False
    alice_data_combined_with_encoding = None

    if os.path.isfile(alice_encoded_cache_path):
        if GLOBAL_CONFIG["Verbose"]:
            print("Found stored data for Alice's encoded records")
        alice_enc_sim = hkl.load(alice_encoded_cache_path).astype(np.float32)

        alice_enc = hkl.load("./data/available_to_eve/alice_data_encoded_%s.h5" % alice_enc_hash)
        alice_full = hkl.load("./data/dev/alice_data_complete_with_encoding_%s.h5" % alice_enc_hash)
        alice_header = alice_enc[0]
        alice_header_full = alice_full[0]
        alice_data_encoded = alice_enc[1:]
        not_reidentified_individuals_header = alice_header
        reidentified_individuals_header = alice_header_full

        n_alice = int(alice_enc_sim[0][2])
        alice_enc_sim = alice_enc_sim[1:]
        overlap_count = source_context.get("overlap_count")
        if GLOBAL_CONFIG["DropFrom"] == "Both":
            overlap_count = _load_overlap_count(alice_enc_hash)
    else:
        if GLOBAL_CONFIG["Verbose"]:
            print("Loading Alice's data")

        overlap_count = source_context.get("overlap_count")
        if GLOBAL_CONFIG["DropFrom"] == "Both":
            _save_overlap_count(alice_enc_hash, overlap_count)

        alice_header = source_context["alice_header_prepared"]
        alice_data = [source_context["alice_uid_map"][str(uid)] for uid in source_context["selected_alice_uids"]]
        alice_uids = source_context["selected_alice_uids"]
        n_alice = len(alice_uids)
        alice_encoder = _create_encoder("Alice", GLOBAL_CONFIG, ENC_CONFIG, source_context["alice_is_precomputed"])

        if GLOBAL_CONFIG["Verbose"]:
            print("Encoding Alice's Data")

        alice_enc_sim, alice_data_combined_with_encoding = alice_encoder.encode_and_compare_and_append(
            alice_data,
            alice_uids,
            metric=ENC_CONFIG["AliceMetric"],
            sim=True,
            store_encs=GLOBAL_CONFIG["SaveAliceEncs"],
        )
        alice_data_encoded = [row[-2:] for row in alice_data_combined_with_encoding]

        alice_data_combined_with_encoding = np.vstack((alice_header, alice_data_combined_with_encoding))
        alice_data_encoded = np.vstack((alice_header[-2:], alice_data_encoded))

        hkl.dump(
            alice_data_combined_with_encoding,
            "./data/dev/alice_data_complete_with_encoding_%s.h5" % alice_enc_hash,
            mode="w",
        )
        hkl.dump(alice_data_encoded, "./data/available_to_eve/alice_data_encoded_%s.h5" % alice_enc_hash, mode="w")
        if GLOBAL_CONFIG["DevMode"]:
            save_tsv(alice_data_encoded, "./data/available_to_eve/alice_data_encoded_%s.tsv" % alice_enc_hash)
            save_tsv(
                alice_data_combined_with_encoding,
                "./data/dev/alice_data_complete_with_encoding_%s.tsv" % alice_enc_hash,
            )

        alice_enc_sim, alice_skip_thresholding = _ensure_nonzero_similarity_graph(
            alice_enc_sim,
            GLOBAL_CONFIG["Verbose"],
            "Alice",
        )

        del alice_data

        if GLOBAL_CONFIG["Verbose"]:
            print("Done encoding Alice's data")

        _persist_similarity_cache(alice_encoded_cache_path, n_alice, alice_enc_sim)

    if GLOBAL_CONFIG["Verbose"]:
        print("Computing Thresholds and subsetting data for Alice")

    alice_enc_sim = _postprocess_similarity_graph(
        alice_enc_sim,
        EMB_CONFIG["AliceQuantile"],
        EMB_CONFIG["AliceDiscretize"],
        alice_skip_thresholding,
    )

    if GLOBAL_CONFIG["Verbose"]:
        print("Done processing Alice's data.")

    eve_skip_thresholding = False

    if os.path.isfile(eve_encoded_cache_path):
        if GLOBAL_CONFIG["Verbose"]:
            print("Found stored data for Eve's encoded records")

        eve_enc_sim = hkl.load(eve_encoded_cache_path).astype(np.float32)
        eve_enc = hkl.load("./data/available_to_eve/eve_data_combined_with_encodings_%s.h5" % eve_enc_hash)
        eve_data_combined_with_encoding = eve_enc[1:]
        n_eve = int(eve_enc_sim[0][2])
        eve_enc_sim = eve_enc_sim[1:]
    else:
        if GLOBAL_CONFIG["Verbose"]:
            print("Loading Eve's data")

        eve_header = source_context["eve_header_prepared"]
        eve_data = [source_context["eve_uid_map"][str(uid)] for uid in source_context["selected_eve_uids"]]
        eve_uids = source_context["selected_eve_uids"]
        n_eve = len(eve_uids)
        eve_encoder = _create_encoder("Eve", GLOBAL_CONFIG, ENC_CONFIG, source_context["eve_is_precomputed"])

        if GLOBAL_CONFIG["Verbose"]:
            print("Encoding Eve's Data")

        eve_enc_sim, eve_data_combined_with_encoding = eve_encoder.encode_and_compare_and_append(
            eve_data,
            eve_uids,
            metric=ENC_CONFIG["EveMetric"],
            sim=True,
            store_encs=GLOBAL_CONFIG["SaveEveEncs"],
        )

        eve_data_combined_with_encoding = np.vstack((eve_header, eve_data_combined_with_encoding))
        hkl.dump(
            eve_data_combined_with_encoding,
            "./data/available_to_eve/eve_data_combined_with_encodings_%s.h5" % eve_enc_hash,
            mode="w",
        )
        if GLOBAL_CONFIG["DevMode"]:
            save_tsv(
                eve_data_combined_with_encoding,
                "./data/available_to_eve/eve_data_combined_with_encoding_%s.tsv" % eve_enc_hash,
            )

        eve_enc_sim, eve_skip_thresholding = _ensure_nonzero_similarity_graph(
            eve_enc_sim,
            GLOBAL_CONFIG["Verbose"],
            "Eve",
        )

        del eve_data

        if GLOBAL_CONFIG["Verbose"]:
            print("Done encoding Eve's data")

        _persist_similarity_cache(eve_encoded_cache_path, n_eve, eve_enc_sim)

    if GLOBAL_CONFIG["Verbose"]:
        print("Computing Thresholds and subsetting data for Eve")

    eve_enc_sim = _postprocess_similarity_graph(
        eve_enc_sim,
        EMB_CONFIG["EveQuantile"],
        EMB_CONFIG["EveDiscretize"],
        eve_skip_thresholding,
    )

    if GLOBAL_CONFIG["Verbose"]:
        print("Done processing Eve's data.")

    return {
        "alice_enc_sim": alice_enc_sim,
        "eve_enc_sim": eve_enc_sim,
        "alice_data_encoded": alice_data_encoded,
        "eve_data_combined_with_encoding": eve_data_combined_with_encoding,
        "reidentified_individuals_header": reidentified_individuals_header,
        "not_reidentified_individuals_header": not_reidentified_individuals_header,
        "n_alice": n_alice,
        "n_eve": n_eve,
        "overlap_count": overlap_count,
        "alice_data_combined_with_encoding": alice_data_combined_with_encoding,
    }


def _load_or_train_embeddings(party, enc_sim, emb_hash, GLOBAL_CONFIG, EMB_CONFIG):
    embeddings_path = "./graphMatching/data/embeddings/%s-%s.h5" % (party.lower(), emb_hash)
    uids_path = "./graphMatching/data/embeddings/%s_uids-%s.pck" % (party.lower(), emb_hash)

    if os.path.isfile(embeddings_path):
        if GLOBAL_CONFIG["Verbose"]:
            print(f"Found stored data for {party}'s embeddings")
        embeddings = hkl.load(embeddings_path).astype(np.float32)
        with open(uids_path, "rb") as f:
            uids = pickle.load(f)
        return embeddings, uids

    if GLOBAL_CONFIG["Verbose"]:
        print(f"Embedding {party}'s data. This may take a while...")

    if EMB_CONFIG["Algo"] == "Node2Vec":
        edge_path = "./graphMatching/data/edgelists/%s.edg" % party.lower()
        np.savetxt(edge_path, enc_sim, delimiter="\t", fmt=["%1.0f", "%1.0f", "%1.16f"])
        embedder = N2VEmbedder(
            walk_length=EMB_CONFIG[f"{party}WalkLen"],
            n_walks=EMB_CONFIG[f"{party}NWalks"],
            p=EMB_CONFIG[f"{party}P"],
            q=EMB_CONFIG[f"{party}Q"],
            dim_embeddings=EMB_CONFIG[f"{party}Dim"],
            context_size=EMB_CONFIG[f"{party}Context"],
            epochs=EMB_CONFIG[f"{party}Epochs"],
            seed=EMB_CONFIG[f"{party}Seed"],
            workers=GLOBAL_CONFIG["Workers"],
            verbose=GLOBAL_CONFIG["Verbose"],
        )
        embedder.train(edge_path)
    else:
        embedder = NetMFEmbedder(
            EMB_CONFIG[f"{party}Dim"],
            EMB_CONFIG[f"{party}Context"],
            EMB_CONFIG[f"{party}Negative"],
            EMB_CONFIG[f"{party}Normalize"],
        )
        embedder.train(enc_sim)

    if GLOBAL_CONFIG["Verbose"]:
        print(f"Done embedding {party}'s data.")

    embeddings, uids = embedder.get_vectors()
    del embedder

    hkl.dump(embeddings, embeddings_path, mode="w")
    with open(uids_path, "wb") as f:
        pickle.dump(uids, f, protocol=5)

    return embeddings, uids


def _select_alignment_subsets(alice_embeddings, alice_uids, eve_embeddings, eve_uids, ALIGN_CONFIG):
    alice_indexdict = dict(zip(alice_uids, range(len(alice_uids))))
    eve_indexdict = dict(zip(eve_uids, range(len(eve_uids))))

    if ALIGN_CONFIG["Selection"] == "GroundTruth":
        alice_sub = alice_embeddings[[alice_indexdict[k] for k in alice_uids[: ALIGN_CONFIG["MaxLoad"]]], :]
        eve_sub = eve_embeddings[[eve_indexdict[k] for k in alice_uids[: ALIGN_CONFIG["MaxLoad"]]], :]
    elif ALIGN_CONFIG["Selection"] == "Random":
        eve_sub = eve_embeddings[np.random.choice(eve_embeddings.shape[0], ALIGN_CONFIG["MaxLoad"], replace=False), :]
        alice_sub = alice_embeddings[
            np.random.choice(alice_embeddings.shape[0], ALIGN_CONFIG["MaxLoad"], replace=False),
            :,
        ]
    else:
        alice_sub = alice_embeddings
        eve_sub = eve_embeddings

    if ALIGN_CONFIG["Batchsize"] == "Auto":
        ALIGN_CONFIG["Batchsize"] = int(0.85 * min(len(alice_sub), len(eve_sub)))
    if ALIGN_CONFIG["Batchsize"] <= 1:
        ALIGN_CONFIG["Batchsize"] = int(ALIGN_CONFIG["Batchsize"] * min(len(alice_sub), len(eve_sub)))
    ALIGN_CONFIG["Batchsize"] = min(ALIGN_CONFIG["Batchsize"], 35000)

    if ALIGN_CONFIG["Selection"] == "GroundTruth":
        alice_sub = np.stack(alice_sub, axis=0)
        eve_sub = np.stack(eve_sub, axis=0)

    return alice_sub, eve_sub


def _build_aligner(ALIGN_CONFIG, ENC_CONFIG, GLOBAL_CONFIG):
    if ALIGN_CONFIG["Wasserstein"]:
        if ALIGN_CONFIG["RegWS"] == "Auto":
            if ENC_CONFIG["EveAlgo"] == "TwoStepHash" or ENC_CONFIG["AliceAlgo"] == "TwoStepHash":
                ALIGN_CONFIG["RegWS"] = 0.1
            else:
                ALIGN_CONFIG["RegWS"] = 0.05

        return WassersteinAligner(
            ALIGN_CONFIG["RegInit"],
            ALIGN_CONFIG["RegWS"],
            ALIGN_CONFIG["Batchsize"],
            ALIGN_CONFIG["LR"],
            ALIGN_CONFIG["NIterInit"],
            ALIGN_CONFIG["NIterWS"],
            ALIGN_CONFIG["NEpochWS"],
            ALIGN_CONFIG["LRDecay"],
            ALIGN_CONFIG["Sqrt"],
            ALIGN_CONFIG["EarlyStopping"],
            verbose=GLOBAL_CONFIG["Verbose"],
        )

    return ProcrustesAligner()


def _build_matcher(GLOBAL_CONFIG):
    if GLOBAL_CONFIG["Matching"] == "MinWeight":
        return MinWeightMatcher(GLOBAL_CONFIG["MatchingMetric"])
    if GLOBAL_CONFIG["Matching"] == "Stable":
        return GaleShapleyMatcher(GLOBAL_CONFIG["MatchingMetric"])
    if GLOBAL_CONFIG["Matching"] == "Symmetric":
        return SymmetricMatcher(GLOBAL_CONFIG["MatchingMetric"])
    return NNMatcher(GLOBAL_CONFIG["MatchingMetric"])


def run_embedding_and_matching(
    GLOBAL_CONFIG,
    ENC_CONFIG,
    EMB_CONFIG,
    ALIGN_CONFIG,
    eve_enc_hash,
    alice_enc_hash,
    eve_emb_hash,
    alice_emb_hash,
    state,
):
    alice_enc_sim = state["alice_enc_sim"]
    eve_enc_sim = state["eve_enc_sim"]

    alice_embeddings, alice_uids = _load_or_train_embeddings(
        "Alice", alice_enc_sim, alice_emb_hash, GLOBAL_CONFIG, EMB_CONFIG
    )
    eve_embeddings, eve_uids = _load_or_train_embeddings(
        "Eve", eve_enc_sim, eve_emb_hash, GLOBAL_CONFIG, EMB_CONFIG
    )

    alice_sub, eve_sub = _select_alignment_subsets(alice_embeddings, alice_uids, eve_embeddings, eve_uids, ALIGN_CONFIG)
    del alice_enc_sim, eve_enc_sim

    if GLOBAL_CONFIG["Verbose"]:
        print("Aligning vectors. This may take a while.")

    aligner = _build_aligner(ALIGN_CONFIG, ENC_CONFIG, GLOBAL_CONFIG)
    transformation_matrix = aligner.align(alice_sub, eve_sub)
    eve_embeddings = np.dot(eve_embeddings, transformation_matrix.T)

    if GLOBAL_CONFIG["Verbose"]:
        print("Done.")
        print("Performing bipartite graph matching")

    matcher = _build_matcher(GLOBAL_CONFIG)
    mapping = matcher.match(alice_embeddings, alice_uids, eve_embeddings, eve_uids)

    reidentified_individuals = [state["reidentified_individuals_header"]]
    reidentified_ids = []
    not_reidentified_individuals = [state["not_reidentified_individuals_header"]]

    correct = 0
    for smaller, larger in mapping.items():
        if smaller[0] == "L":
            continue
        if smaller[1:] == larger[1:]:
            correct += 1
            reidentified_ids.append(int(smaller[2:]))

    for alice_entry in state["alice_data_encoded"][1:]:
        if int(alice_entry[-1]) in reidentified_ids:
            for eve_entry in state["eve_data_combined_with_encoding"][1:]:
                if int(eve_entry[-1]) == int(alice_entry[-1]):
                    if ENC_CONFIG["EveAlgo"] != "None":
                        reidentified_individuals.append(
                            np.concatenate((eve_entry[:-2], [alice_entry[0]], [eve_entry[-1]])).tolist()
                        )
                    else:
                        reidentified_individuals.append(
                            np.concatenate((eve_entry[:-1], [alice_entry[0]], [eve_entry[-1]])).tolist()
                        )
        else:
            not_reidentified_individuals.append(alice_entry)

    output_suffix = "%s_%s_%s_%s" % (eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash)
    hkl.dump(reidentified_individuals, "./data/available_to_eve/reidentified_individuals_%s.h5" % output_suffix, mode="w")
    hkl.dump(
        not_reidentified_individuals,
        "./data/available_to_eve/not_reidentified_individuals_%s.h5" % output_suffix,
        mode="w",
    )

    if GLOBAL_CONFIG["DevMode"]:
        save_tsv(
            not_reidentified_individuals,
            "./data/available_to_eve/not_reidentified_individuals_%s.tsv" % output_suffix,
        )
        save_tsv(
            reidentified_individuals,
            "./data/available_to_eve/reidentified_individuals_%s.tsv" % output_suffix,
        )

    if GLOBAL_CONFIG["DropFrom"] == "Both":
        success_rate = correct / state["overlap_count"]
        print("Correct: %i of %i" % (correct, state["overlap_count"]))
    else:
        success_rate = correct / min(state["n_alice"], state["n_eve"])
        print("Correct: %i of %i" % (correct, min(state["n_alice"], state["n_eve"])))

    print("Success rate: %f" % success_rate)

    if state["alice_data_combined_with_encoding"] is not None:
        return reidentified_individuals, not_reidentified_individuals, state["alice_data_combined_with_encoding"]
    return reidentified_individuals, not_reidentified_individuals
