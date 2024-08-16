import pickle
import os
import time
import contextlib
import joblib

from statistics import mean, median
from collections import Counter
from main import run
from tqdm import tqdm, trange
from hashlib import md5
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances

from extension_utils import *
from encoders.bf_encoder import BFEncoder
from encoders.tmh_encoder import TMHEncoder
from encoders.tsh_encoder import TSHEncoder
from encoders.non_encoder import NonEncoder


def __extend(chunk, unknown_plaintexts, known_plaintexts, ts_simdict, avglen, GLOBAL_CONFIG):
    relist_simple = []
    relist_refined = []

    for u_ind in chunk:
        # included_ngr = set()

        if GLOBAL_CONFIG["Verbose"]:
            print("___________________________")

        target_sims = ts_simdict[u_ind]

        asc_sim_inds = np.argsort(target_sims)
        ordered_sim_inds = np.flip(asc_sim_inds)

        included_ngr = set()
        not_included_ngr = set()

        incl_size_before = float("-Inf")
        excl_size_before = float("-Inf")

        while (len(included_ngr) - incl_size_before) > 0 or (len(not_included_ngr) - excl_size_before) > 0:
            incl_size_before = len(included_ngr)
            excl_size_before = len(not_included_ngr)

            high_sim_ngr = []

            for i in asc_sim_inds:
                kp = known_plaintexts[i]
                kp_len = len(kp)
                # jacc = target_sims[i]
                sim = target_sims[i]
                est_overlap = (sim * (kp_len + avglen)) / 2  # Dice
                # est_overlap = round((sim * (kp_len + avglen) / (sim + 1))) # Jaccard

                if GLOBAL_CONFIG["Verbose"]:
                    gt = kp.intersection(unknown_plaintexts[u_ind])
                    print("Guessed overlap of %i, True %i" % (est_overlap, len(gt)))
                # Find n-grams that are potentially part of the unknown string
                # difference = kp.difference(not_included_ngr)

                # Find n-grams that are definitely part of both, the known and the unknown string
                known_intersect = kp.intersection(included_ngr)

                if len(kp.difference(not_included_ngr.union(known_intersect))) == est_overlap - len(known_intersect):
                    to_add = kp.difference(not_included_ngr)
                    # high_sim_ngr += list(to_add)
                    # included_ngr = included_ngr.union(to_add)
                    if GLOBAL_CONFIG["Verbose"]:
                        new_incl = to_add.difference(included_ngr)
                        if len(new_incl) > 0:
                            print("Found %i possibly included n-grams, %i of them new" % (len(to_add), len(new_incl)))
                            if len(to_add.difference(unknown_plaintexts[u_ind])) > 0:
                                print("Wrongly included %s" % (to_add.difference(unknown_plaintexts[u_ind])))

                    included_ngr = included_ngr.union(to_add)

                # Add additional n-Grams that are not included:
                if len(kp.intersection(included_ngr)) == est_overlap:
                    poss_excl_ngr = kp.difference(included_ngr)
                    poss_excl_ngr = poss_excl_ngr.difference(included_ngr)
                    if GLOBAL_CONFIG["Verbose"]:
                        new_excl = poss_excl_ngr.difference(not_included_ngr)
                        if len(new_excl) > 0:
                            print(
                                "Found %i possibly excluded n-grams, %i of them new" % (
                                len(poss_excl_ngr), len(new_excl)))
                            if len(poss_excl_ngr.intersection(unknown_plaintexts[u_ind])) > 0:
                                print("Wrongly excluded %s" % (poss_excl_ngr.intersection(unknown_plaintexts[u_ind])))
                    not_included_ngr = not_included_ngr.union(poss_excl_ngr)
        ground_truth = len(unknown_plaintexts[u_ind])

        if len(included_ngr) > 0:
            tp = len(included_ngr.intersection(unknown_plaintexts[u_ind]))
            precision = tp / len(included_ngr)
            recall = tp / ground_truth
            if (precision + recall) > 0:
                f1 = 2 * ((precision * recall) / (precision + recall))
            else:
                f1 = 0
        else:
            tp = precision = recall = f1 = 0
        if GLOBAL_CONFIG["Verbose"]:
            print("Found %i of %i N-Grams. \nPrecision: %f \nRecall: %f \nF1-Score %f" % (
                tp, ground_truth, precision, recall, f1))
        guessed = len(included_ngr)
        relist_simple.append((u_ind, tp, ground_truth, precision, recall, f1))

        # jacc_err = float("Inf")
        # guess_list = [set()]
        # refined_guess = set()
        # available_ngr = included_ngr
        # target_len = round(est_bf_elem(unknown_alice_encs[u_ind],10))

        refined_guess = set()
        best_mae = float("Inf")
        while len(included_ngr) > 0:  # Essentially While True if included_ngr has at least one element
            for ngr in included_ngr:
                cur_best_mae = float("Inf")
                cur_best_guess = set()

                tmp_guess = refined_guess.union([ngr])
                mae = calc_mse_dice(tmp_guess, ordered_sim_inds, target_sims, known_plaintexts, top_n=25)

                if mae < cur_best_mae:
                    cur_best_mae = mae
                    cur_best_guess = tmp_guess

            if cur_best_mae < best_mae:
                cur_best_mae = best_mae
                refined_guess = cur_best_guess
            else:
                break

        if len(refined_guess) > 0:
            tp_ref = len(refined_guess.intersection(unknown_plaintexts[u_ind]))
            precision_ref = tp_ref / len(refined_guess)
            recall_ref = tp_ref / ground_truth

            if (precision_ref + recall_ref) > 0:
                f1_ref = 2 * ((precision_ref * recall_ref) / (precision_ref + recall_ref))
            else:
                f1_ref = 0
        else:
            tp_ref = precision_ref = recall_ref = f1_ref = 0

        if GLOBAL_CONFIG["Verbose"]:
            print(
                "--- REFINED GUESS ---\nFound %i of %i N-Grams. \nPrecision: %f \nRecall: %f \nF1-Score %f" % (
                    tp_ref, ground_truth, precision_ref, recall_ref, f1_ref))

        relist_refined.append((u_ind, tp_ref, ground_truth, precision_ref, recall_ref, f1_ref))
    return relist_simple, relist_refined


def __simcalc(chunk, known_alice_encs, ENC_CONFIG):
    ts_simdict = {}
    #ovl_simdict = {}
    #old_simdict = {}
    for u_ind, u_enc in chunk:
        if ENC_CONFIG["AliceAlgo"] == "BloomFilter":
            #tmp = pairwise_distances(u_enc.reshape(1, -1), known_alice_encs, metric="jaccard")
            #tmp = 1 - tmp
            tmp = pairwise_dice_bf(u_enc.reshape(1, -1), known_alice_encs, ENC_CONFIG["AliceBits"])
            tmp = tmp.tolist()

        elif ENC_CONFIG["AliceAlgo"] == "TabMinHash":
            tmp = pairwise_dice_tmh(u_enc, known_alice_encs, ENC_CONFIG["Alice1BitHash"])
            tmp = list(tmp[0])
        else:
            tmp = pairwise_dice(u_enc, known_alice_encs)
            tmp = list(tmp[0])

        #overlaps = list(overlaps[0])
        #jaccs = list(jaccs[0])
        #ovl_simdict[u_ind] = overlaps
        ts_simdict[u_ind] = tmp
        #old_simdict[u_ind] = tmp
    return ts_simdict

def run_extension(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG):
    #https://stackoverflow.com/a/58936697
    @contextlib.contextmanager
    def tqdm_joblib(tqdm_object):
        """Context manager to patch joblib to report into tqdm progress bar given as argument"""

        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()


    # Optimization: Store Encodings if parameters are unchanged
    alice_enc_hash = md5(
        ("%s-%s" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()).hexdigest()

    # Load dataset
    data, uids = read_tsv(GLOBAL_CONFIG["Data"])
    data = [["".join(d).lower()] for d in data]
    data_dict = dict(zip(uids, data))

    if GLOBAL_CONFIG["RunGMA"]:
        mapping = run(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG)
        with open("./data/encodings/encoding_dict.pck", "rb") as f:
            enc_dict = pickle.load(f)
    else:

        if os.path.isfile("./data/encodings/encoding_dict-%s.pck" % alice_enc_hash):

            if GLOBAL_CONFIG["Verbose"]:
                print("Will load stored encodings for better performance.")

            with open("./data/encodings/encoding_dict-%s.pck" % alice_enc_hash, "rb") as f:
                enc_dict = pickle.load(f)

        else:
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

            enc_dict = alice_encoder.get_encoding_dict(data, uids)

            with open("./data/encodings/encoding_dict-%s.pck" % alice_enc_hash, "wb") as f:
                pickle.dump(enc_dict, f, pickle.HIGHEST_PROTOCOL)

        mapping = simulate_mapping(uids, correct_share = GLOBAL_CONFIG["CorrectShare"],
                                   matched_share=GLOBAL_CONFIG["Overlap"])
    del data

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

    start_total = time.time()

    ts_simdict = {}
    unknown_alice_encs_enum = [(u_ind, u_enc) for u_ind, u_enc in enumerate(unknown_alice_encs)]
    unknown_alice_encs_enum = split_to_chunks(unknown_alice_encs_enum, os.cpu_count() - 1)
    with tqdm_joblib(tqdm(desc="Sim Calculation", total=os.cpu_count() - 1)) as progress_bar:
        parallel = Parallel(n_jobs=-2)
        output_generator = parallel(
            delayed(__simcalc)(chunk, known_alice_encs, ENC_CONFIG) for chunk in unknown_alice_encs_enum)

    for tmp_simdict in output_generator:
        ts_simdict.update(tmp_simdict)


    ind_chunks = split_to_chunks(list(range(len(unknown_alice_encs))), os.cpu_count()-1)

    with tqdm_joblib(tqdm(desc="Extension", total=os.cpu_count()-1)) as progress_bar:
        parallel = Parallel(n_jobs=-2)
        output_generator = parallel(delayed(__extend)(chunk, unknown_plaintexts, known_plaintexts, ts_simdict, avglen, GLOBAL_CONFIG) for chunk in ind_chunks)

    simple = []
    refined = []
    for tmp_simple, tmp_refined in output_generator:
        simple += tmp_simple
        refined += tmp_refined

    simple_precs = [s[3] for s in simple]
    simple_recs = [s[4] for s in simple]
    simple_f1 = [s[5] for s in simple]

    ref_precs = [r[3] for r in refined]
    ref_recs = [r[4] for r in refined]
    ref_f1 = [r[5] for r in refined]

    elapsed_total = time.time() - start_total
    bench_vals = [time.time()]
    bench_keys = ["timestamp"]

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

    bench_keys += ["MinPrecision", "MaxPrecision","AvgPrecision", "MedPrecision", "MinRecall", "MaxRecall",
                   "AvgRecall", "MedRecall", "MinF1", "MaxF1", "AvgF1", "MedF1", "MinPrecision_Ref",
                   "MaxPrecision_Ref","AvgPrecision_Ref", "MedPrecision_Ref", "MinRecall_Ref", "MaxRecall_Ref",
                   "AvgRecall_Ref", "MedRecall_Ref", "MinF1_Ref", "MaxF1_Ref", "AvgF1_Ref", "MedF1_Ref",
                   "DuractionSec"]

    bench_vals += [min(simple_precs), max(simple_precs), mean(simple_precs), median(simple_precs),
    min(simple_recs), max(simple_recs), mean(simple_recs), median(simple_recs),
    min(simple_f1), max(simple_f1), mean(simple_f1), median(simple_f1),

    min(ref_precs), max(ref_precs), mean(ref_precs), median(ref_precs),
    min(ref_recs), max(ref_recs), mean(ref_recs), median(ref_recs),
    min(ref_f1), max(ref_f1), mean(ref_f1), median(ref_f1),
    elapsed_total]

    if not os.path.isfile("data/extension_benchmark.tsv"):
        save_tsv([bench_keys], "data/extension_benchmark.tsv")

    save_tsv([bench_vals], "data/extension_benchmark.tsv", mode="a")


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
        "Data": "./data/fakename_5k.tsv",
        "Overlap": 0.05,
        "CorrectShare": 0.9,
        "DropFrom": "Eve",
        "DevMode": False,  # Development Mode, saves some intermediate results to the /dev directory
        "BenchMode": False,  # Benchmark Mode
        "Verbose": False,  # Print Status Messages?
        "MatchingMetric": "cosine",
        "Matching": "MinWeight",
        "Workers": -1,
        "SaveAliceEncs": True,
        "SaveEveEncs": False,
        "RunGMA" : False
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