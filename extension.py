from extension_utils import *
from collections import Counter
import pickle
import os
import numpy as np
import time
from statistics import mean, median
from main import run
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

def run_extension(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG):

    mapping = run(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG)

    # Load dataset
    data, uids = read_tsv(GLOBAL_CONFIG["Data"])
    data = [["".join(d).lower()] for d in data]
    data_dict = dict(zip(uids, data))
    del data

    with open("./data/encodings/encoding_dict.pck", "rb") as f:
        enc_dict = pickle.load(f)

    known_uids = []
    known_data = []
    used_uids = []

    for alice_uid, eve_uid in mapping.items():
        alice_uid = alice_uid[2:]
        eve_uid = eve_uid[2:]
        known_uids.append(alice_uid)
        used_uids.append(eve_uid)
        known_data.append(data_dict[eve_uid])

    unknown_uids = [u for u in uids if u not in used_uids]
    unknown_data = [data_dict[u] for u in unknown_uids]

    known_plaintexts = [calc_ngram(d, 2) for d in known_data]
    unknown_plaintexts = [calc_ngram(d, 2) for d in unknown_data]

    known_alice_encs = [enc_dict[i] for i in known_uids]
    if ENC_CONFIG["AliceAlgo"] in ["BloomFilter", "TabMinHash"]:
        known_alice_encs = np.vstack(known_alice_encs)
    unknown_alice_encs = [enc_dict[i] for i in unknown_uids]

    plaintext_lengths = [len(p) for p in known_plaintexts]
    avglen = sum(plaintext_lengths) / len(plaintext_lengths)

    simple = []
    refined = []

    start_total = time.time()

    for u_ind, u_enc in tqdm(enumerate(unknown_alice_encs), total=len(unknown_alice_encs), disable=GLOBAL_CONFIG["Verbose"]):
        # if u_ind < 88:
        #    continue
        if GLOBAL_CONFIG["Verbose"]:
            print("___________________________")
        # included_ngr = set()

        # not_included_ngr = set()

        # target_sims = calc_sims(u, known_data, q_gram_dice_sim)
        if ENC_CONFIG["AliceAlgo"] == "BloomFilter":
            tmp = pairwise_distances(u_enc.reshape(1, -1), known_alice_encs, metric="jaccard")
            tmp = 1 - tmp
            # target_sims = list(reg.predict(tmp.reshape(-1, 1)).flatten())
        elif ENC_CONFIG["AliceAlgo"] == "TabMinHash":
            tmp = pairwise_dice_tmh(u_enc, known_alice_encs, ENC_CONFIG["Alice1BitHash"])
            # target_sims = list(tmp[0])
        else:
            tmp = pairwise_dice(u_enc, known_alice_encs)

        target_sims = list(tmp[0])
        tm = min(target_sims)
        # target_sims = [t - (tm) for t in target_sims]

        asc_sim_inds = np.argsort(target_sims)
        ordered_sim_inds = np.flip(asc_sim_inds)

        init_guess_list = []
        for i in range(10):
            init_guess_list += list(known_plaintexts[ordered_sim_inds[i]])

        cntr = Counter(init_guess_list)
        included_ngr = set([s[0] for s in cntr.most_common(5)])
        orig_incl_ngr = included_ngr

        init_guess_incl = set()
        # TMH 6
        for i in range(6):
            init_guess_incl = init_guess_incl.union(known_plaintexts[ordered_sim_inds[i]])

        not_included_ngr = set()
        # TMH 100
        for i in range(100):
            not_included_ngr = not_included_ngr.union(known_plaintexts[asc_sim_inds[i]])

        # TMH mit incl_ngr statt orig_incl
        not_included_ngr = guess_zero_overlap(target_sims, known_plaintexts, included_ngr=orig_incl_ngr,
                                              not_included_ngr=not_included_ngr, perc=1,
                                              verbose=GLOBAL_CONFIG["Verbose"], avglen=avglen)

        not_included_ngr = not_included_ngr.difference(init_guess_incl)
        # cntr = Counter(init_guess_list)
        # init_guess_incl = set([s[0] for s in cntr.most_common(10)])

        if GLOBAL_CONFIG["Verbose"]:
            print("Guessed %i not included N-Grams" % (len(not_included_ngr)))
            if len(not_included_ngr.intersection(unknown_plaintexts[u_ind])) > 0:
                wrong_not_incl = not_included_ngr.intersection(unknown_plaintexts[u_ind])
                print("%i of them wrongly: %s" % (len(wrong_not_incl), str(wrong_not_incl)))

        incl_size_before = float("-Inf")
        excl_size_before = float("-Inf")
        guess = ""

        est_overlaps = {}

        # ordered_sim_inds = np.flip(ordered_sim_inds)

        while (len(included_ngr) - incl_size_before) > 0 or (len(not_included_ngr) - excl_size_before) > 0:
            incl_size_before = len(included_ngr)
            excl_size_before = len(not_included_ngr)

            low_sim_ngr = []
            low_sim_count = 0
            high_sim_ngr = []
            high_sim_count = 0

            for i in ordered_sim_inds:
                kp = known_plaintexts[i]
                kp_len = len(kp)
                jacc = target_sims[i]
                est_overlap = round((jacc * (kp_len + avglen) / (jacc + 1)))
                est_overlaps[i] = est_overlap

                if GLOBAL_CONFIG["Verbose"]:
                    gt = kp.intersection(unknown_plaintexts[u_ind])
                    print("Guessed overlap of %i, True %i" % (est_overlap, len(gt)))
                # Find n-grams that are potentially part of the unknown string
                difference = kp.difference(not_included_ngr)

                # Find n-grams that are definitely part of both, the known and the unknown string
                known_intersect = kp.intersection(included_ngr)

                # first_letters = set([n[0] for n in included_ngr])
                # last_letters = set([n[-1] for n in included_ngr])

                if len(kp.difference(not_included_ngr.union(known_intersect))) == est_overlap - len(known_intersect):
                    to_add = kp.difference(not_included_ngr)
                    # add_graph = buildgraph(list(to_add))
                    # add_possibilities = [''.join(sent) for sent in allsentences(deepcopy(add_graph), 2, 30)]
                    # add_possibilities.sort(key=len)
                    # add_possibilities.reverse()
                    new_incl = to_add.difference(included_ngr)
                    # to_add = set([n for n in to_add if n[0] in last_letters or n[1] in first_letters])
                    high_sim_ngr += list(to_add)
                    included_ngr = included_ngr.union(to_add)
                    if GLOBAL_CONFIG["Verbose"] and len(new_incl) > 0:
                        print("Found %i possibly included n-grams, %i of them new" % (len(to_add), len(new_incl)))
                        if len(to_add.difference(unknown_plaintexts[u_ind])) > 0:
                            print("Wrongly included %s" % (to_add.difference(unknown_plaintexts[u_ind])))

                # Add additional n-Grams that are not included:
                if len(kp.intersection(included_ngr)) == est_overlap:
                    poss_excl_ngr = kp.difference(included_ngr)
                    poss_excl_ngr = poss_excl_ngr.difference(init_guess_incl)
                    new_excl = poss_excl_ngr.difference(not_included_ngr)
                    not_included_ngr = not_included_ngr.union(poss_excl_ngr)

                    if GLOBAL_CONFIG["Verbose"] and len(new_excl) > 0:
                        print("Found %i possibly excluded n-grams, %i of them new" % (len(poss_excl_ngr), len(new_excl)))
                        if len(poss_excl_ngr.intersection(unknown_plaintexts[u_ind])) > 0:
                            print("Wrongly excluded %s" % (poss_excl_ngr.intersection(unknown_plaintexts[u_ind])))

            # not_included_ngr = guess_zero_overlap(target_sims, known_plaintexts, included_ngr=included_ngr, not_included_ngr=not_included_ngr, perc=1, verbose=verbose, avglen=avglen)

        ground_truth = len(unknown_plaintexts[u_ind])
        if len(included_ngr) > 0:
            tp = len(included_ngr.intersection(unknown_plaintexts[u_ind]))
            precision = tp / len(included_ngr)
            recall = tp / ground_truth
            f1 = 2 * ((precision * recall) / (precision + recall))
        else:
            tp = precision = recall = f1 = 0
        if GLOBAL_CONFIG["Verbose"]:
            print("Found %i of %i N-Grams. \nPrecision: %f \nRecall: %f \nF1-Score %f" % (
                tp, ground_truth, precision, recall, f1))
        guessed = len(included_ngr)
        simple.append((u_ind, tp, ground_truth, precision, recall, f1))

        jacc_err = float("Inf")
        guess_list = [set()]
        refined_guess = set()
        # available_ngr = included_ngr

        while True:
            new_guess_list = []
            for guess in guess_list:
                je_list = []
                ngr_list = []
                available_ngr = included_ngr.difference(guess)
                for ngr in available_ngr:
                    tmp = guess.union(set([ngr]))
                    je = calc_mae_jacc(tmp, ordered_sim_inds, target_sims, known_plaintexts, est_overlaps, top_n=50)
                    je_list.append(je)
                    ngr_list.append(set([ngr]))

                inds_by_je = np.argsort(je_list)
                for j in range(min(len(je_list), 3)):
                    new_guess_list.append(guess.union(ngr_list[inds_by_je[j]]))

            guess_list += new_guess_list

            je_list = []
            for guess in guess_list:
                je = calc_mae_jacc(guess, ordered_sim_inds, target_sims, known_plaintexts, est_overlaps, top_n=50)
                je_list.append(je)

            inds_by_je = np.argsort(je_list)

            if je_list[inds_by_je[0]] < jacc_err:
                jacc_err = je_list[inds_by_je[0]]
                # print(jacc_err)
                refined_guess = guess_list[inds_by_je[0]]
            else:
                break

            if len(guess_list) > 20:
                new_guess_list = []
                je_list = []

                for i in range(20):
                    new_guess_list.append(guess_list[inds_by_je[i]])
                guess_list = new_guess_list

        guessed_ref = len(refined_guess)
        tp_ref = len(refined_guess.intersection(unknown_plaintexts[u_ind]))
        precision_ref = tp_ref / len(refined_guess)
        recall_ref = tp_ref / ground_truth
        if (precision_ref + recall_ref) > 0:
            f1_ref = 2 * ((precision_ref * recall_ref) / (precision_ref + recall_ref))
        else:
            f1_ref = 0

        # bench_vals = vals = [success_rate, correct, n_alice, n_eve, elapsed_total, elapsed_alice_enc, elapsed_eve_enc,
        #         elapsed_alice_emb, elapsed_eve_emb, elapsed_align_prep, elapsed_align, elapsed_mapping,
        #         elapsed_relevant]



        if GLOBAL_CONFIG["Verbose"]:
            print("--- REFINED GUESS ---\nFound %i of %i N-Grams. \nPrecision: %f \nRecall: %f \nF1-Score %f \nTrue: %s" % (
                tp_ref, ground_truth, precision_ref, recall_ref, f1_ref, unknown_data[u_ind]))

        refined.append((u_ind, tp_ref, ground_truth, precision_ref, recall_ref, f1_ref))

        elapsed_total = time.time() - start_total
        if GLOBAL_CONFIG["BenchMode"]:
            bench_keys = ["timestamp"]
            bench_vals = [time.time()]
            for key, val in EMB_CONFIG.items():
                bench_keys.append(key)
                bench_vals.append(val)
            for key, val in ENC_CONFIG.items():
                bench_keys.append(key)
                bench_vals.append(val)
            for key, val in GLOBAL_CONFIG.items():
                bench_keys.append(key)
                bench_vals.append(val)
            for key, val in ALIGN_CONFIG.items():
                bench_keys.append(key)
                bench_vals.append(val)
            bench_keys += ["Duration", "TrueNgrams", "GuessedNgrams", "TP", "Precision", "Recall", "F1",
                           "GuessedNgramsRefined", "TPRefined", "PrecisionRefined", "RecallRefined", "F1Refined"]

            bench_vals += [elapsed_total, ground_truth, guessed, tp, precision, recall, f1, guessed_ref, tp_ref,
                     precision_ref, recall_ref, f1_ref]


            if not os.path.isfile("data/extension_benchmark.tsv"):
                save_tsv([bench_keys], "data/extension_benchmark.tsv")

            save_tsv([bench_vals], "data/extension_benchmark.tsv", mode="a")


    simple_precs = [s[3] for s in simple]
    simple_recs = [s[4] for s in simple]
    simple_f1 = [s[5] for s in simple]

    ref_precs = [r[3] for r in refined]
    ref_recs = [r[4] for r in refined]
    ref_f1 = [r[5] for r in refined]

    report_string = """---- ATTACK SUMMARY ----
    
    **** Simple Guessing ****
    
    Minimum Precision: %0.4f
    Maximum Precision: %0.4f
    Average Precision: %0.4f
    Median Precision:  %0.4f
    
    Minimum Recall: %0.4f
    Maximum Recall: %0.4f
    Average Recall: %0.4f
    Median Recall:  %0.4f
    
    Minimum F1: %0.4f
    Maximum F1: %0.4f
    Average F1: %0.4f
    Median F1:  %0.4f
    
    **** Refined Guessing ****
    
    Minimum Precision: %0.4f
    Maximum Precision: %0.4f
    Average Precision: %0.4f
    Median Precision:  %0.4f
    
    Minimum Recall: %0.4f
    Maximum Recall: %0.4f
    Average Recall: %0.4f
    Median Recall:  %0.4f
    
    Minimum F1: %0.4f
    Maximum F1: %0.4f
    Average F1: %0.4f
    Median F1:  %0.4f
    
    ---- Attack completed after %i seconds (%0.2f Minutes). ----
    """
    print(report_string % (min(simple_precs), max(simple_precs), mean(simple_precs), median(simple_precs),
    min(simple_recs), max(simple_recs), mean(simple_recs), median(simple_recs),
    min(simple_f1), max(simple_f1), mean(simple_f1), median(simple_f1),

    min(ref_precs), max(ref_precs), mean(ref_precs), median(ref_precs),
    min(ref_recs), max(ref_recs), mean(ref_recs), median(ref_recs),
    min(ref_f1), max(ref_f1), mean(ref_f1), median(ref_f1),
    elapsed_total, elapsed_total/60))

if __name__ == "__main__":
    GLOBAL_CONFIG = {
        "Data": "./data/fakename_1k.tsv",
        "Overlap": 0.95,
        "DropFrom": "Eve",
        "DevMode": False,  # Development Mode, saves some intermediate results to the /dev directory
        "BenchMode": False,  # Benchmark Mode
        "Verbose": True,  # Print Status Messages?
        "MatchingMetric": "cosine",
        "Matching": "MinWeight",
        "Workers": -1,
        "SaveAliceEncs": True,
        "SaveEveEncs": False
    }

    ENC_CONFIG = {
        "AliceAlgo": "TabMinHash",
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
        "EveNHash": 2000,
        "EveNHashBits": 32,
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
        "AliceP": 250,  # 0.5
        "AliceQ": 300,  # 2z
        "AliceEpochs": 5,
        "AliceSeed": 42,
        "EveWalkLen": 100,
        "EveNWalks": 20,
        "EveP": 250,  # 0.5
        "EveQ": 300,  # 2
        "EveEpochs": 5,
        "EveSeed": 42
    }

    ALIGN_CONFIG = {
        "RegWS": max(0.1, GLOBAL_CONFIG["Overlap"] / 2),  # 0005
        "RegInit": 1,  # For BF 0.25
        "Batchsize": 1,  # 1 = 100%
        "LR": 200.0,
        "NIterWS": 20,
        "NIterInit": 5,  # 800
        "NEpochWS": 100,
        "LRDecay": 0.999,
        "Sqrt": True,
        "EarlyStopping": 10,
        "Selection": "None",
        "MaxLoad": None,
        "Wasserstein": True
    }

    run_extension(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG)