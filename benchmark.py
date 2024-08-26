from main import run

GLOBAL_CONFIG = {
    "Data": "./data/titanic_full.tsv",
    "Overlap": 1,
    "DropFrom": "Alice",
    "DevMode": False,  # Development Mode, saves some intermediate results to the /dev directory
    "BenchMode": True,  # Benchmark Mode
    "Verbose": False,  # Print Status Messages?
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
    "RegWS": -1, # Will be set in loop below
    "RegInit": 1,  # For BF 0.25
    "Batchsize": 1,  # 1 = 100%
    "LR": 200.0,
    "NIterWS": 100,
    "NIterInit": 5,  # 800
    "NEpochWS": 100,
    "LRDecay": 1,
    "Sqrt": True,
    "EarlyStopping": 10,
    "Selection": "None",
    "MaxLoad": None,
    "Wasserstein": True
}

# Encodings to iterate over
encs = ["TwoStepHash", "BloomFilter", "TabMinHash"]
diffuse = [False, True]
diff_params = [2,3,5,8,10]

# Datasets to iterate over
datasets = ["titanic_full.tsv", "fakename_1k.tsv", "fakename_2k.tsv", "fakename_5k.tsv", "fakename_10k.tsv",
            "fakename_20k.tsv", "fakename_50k.tsv", "euro_full.tsv", "ncvoter.tsv"]

drop = ["Alice","Both"]

# Overlaps to iterate over (5% to 100% in increments of 5 percentage points)
overlap = [i/100 for i in range(5, 105, 5)]

for e in encs:
    ENC_CONFIG["AliceAlgo"] = e
    # In case of BF encoding, the Attacker applies encoding too
    if e == "BloomFilter":
        ENC_CONFIG["EveAlgo"] = e
        tmp_diffuse = diffuse
    else:
        ENC_CONFIG["EveAlgo"] = None
        # Only iterate over diffusion settings when benchmarking Bloom Filters
        tmp_diffuse = [None]

    for u in tmp_diffuse:

        if u:
            tmp_diff_params = diff_params
        else:
            tmp_diff_params = [None]

        ENC_CONFIG["AliceDiffuse"] = u
        ENC_CONFIG["EveDiffuse"] = u

        for t in tmp_diff_params:
            ENC_CONFIG["AliceT"] = t
            ENC_CONFIG["EveT"] = t

            for d in datasets:
                GLOBAL_CONFIG["Data"] = "./data/" + d
                for dr in drop:
                    GLOBAL_CONFIG["DropFrom"] = dr
                    for o in overlap:
                        GLOBAL_CONFIG["Overlap"] = o

                        # Set regularization parameters according to dataset.
                        if d == "ncvoter.tsv":
                            ALIGN_CONFIG["RegWS"] = 0.1
                        elif d == "euro_full.tsv":
                            ALIGN_CONFIG["RegWS"] = max(0.5, o)
                        else:
                            ALIGN_CONFIG["RegWS"] = max(0.1, o/3)
                        # Important: Pass (deep) copies of the dictionaries to the functions, as contents would be
                        # changed otherwise!
                        run(GLOBAL_CONFIG.copy(), ENC_CONFIG.copy(), EMB_CONFIG.copy(), ALIGN_CONFIG.copy())
