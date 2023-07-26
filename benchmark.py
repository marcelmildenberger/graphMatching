from main import run

GLOBAL_CONFIG = {
    "Data": "./data/fakename_1k.tsv",
    "Overlap": 1,
    "DevMode": False,  # Development Mode, saves some intermediate results to the /dev directory
    "BenchMode": True,  # Benchmark Mode
    "Verbose": False,  # Print Status Messages?
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
    "AliceDiscretize": True,
    "AliceDim": 128,
    "AliceContext": 10,
    "AliceNegative": 1,
    "AliceNormalize": True,
    "EveQuantile": 0.9,
    "EveDiscretize": True,
    "EveDim": 128,
    "EveContext": 10,
    "EveNegative": 1,
    "EveNormalize": True,
    "Workers": -1,
}

ALIGN_CONFIG = {
    "RegWS": "Auto",
    "RegInit": 1,
    "Batchsize": "Auto",
    "LR": 500.0,
    "NIterWS": 5,
    "NIterInit": 50,  # 800
    "NEpochWS": 200,
    "LRDecay": 0.95,
    "Sqrt": False,
    "EarlyStopping": 50,
    "Selection": "None",
    "Wasserstein": True,
}

# Global params
datasets = ["fakename_1k.tsv", "fakename_2k.tsv", "fakename_5k.tsv", "fakename_10k.tsv", "fakename_20k.tsv",
            "fakename_50k.tsv", "fakename_100k.tsv"]
datasets = ["fakename_1k.tsv", "fakename_2k.tsv"]
overlap = [i/100 for i in range(10, 105, 5)]

for d in datasets:
    GLOBAL_CONFIG["Data"] = "./data/" + d
    for o in overlap:
        if "1k" in d and o < 0.15:
            continue
        GLOBAL_CONFIG["Overlap"] = o
        run(GLOBAL_CONFIG.copy(), ENC_CONFIG.copy(), EMB_CONFIG.copy(), ALIGN_CONFIG.copy())



