from main import run

GLOBAL_CONFIG = {
    "Data": "./data/fakename_1k.tsv",
    "Overlap": 0.9,
    "DropFrom": "Both",
    "DevMode": False,  # Development Mode, saves some intermediate results to the /dev directory
    "BenchMode": True,  # Benchmark Mode
    "Verbose": False,  # Print Status Messages?
    "MatchingMetric": "cosine",
    "Matching": "MinWeight",
    "Workers": -1
}

ENC_CONFIG = {
    "AliceAlgo": "BloomFilter",
    "AliceSecret": "SuperSecretSalt1337",
    "AliceN": 2,
    "AliceMetric": "dice",
    "EveAlgo": "None",
    "EveSecret": "ATotallyDifferentString42",
    "EveN": 2,
    "EveMetric": "dice",
    # For BF encoding
    "AliceBFLength": 1024,
    "AliceBits": 10,
    "AliceDiffuse": True,
    "AliceT": 10,
    "AliceEldLength": 1024,
    "EveBFLength": 1024,
    "EveBits": 10,
    "EveDiffuse": True,
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
    "RegWS": GLOBAL_CONFIG["Overlap"] / 2,  # 0005
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
# Global params
datasets = ["fakename_1k.tsv", "fakename_2k.tsv", "fakename_5k.tsv", "fakename_10k.tsv", "fakename_20k.tsv",
            "fakename_50k.tsv", "fakename_100k.tsv"]
overlap = [i/100 for i in range(5, 105, 5)]
discretize = [False]

for d in datasets:
    GLOBAL_CONFIG["Data"] = "./data/" + d
    for o in overlap:
        GLOBAL_CONFIG["Overlap"] = o
        ALIGN_CONFIG["RegWS"] = max(0.1, o/2)
        for z in discretize:
            EMB_CONFIG["AliceDiscretize"] = z
            EMB_CONFIG["EveDiscretize"] = z
            run(GLOBAL_CONFIG.copy(), ENC_CONFIG.copy(), EMB_CONFIG.copy(), ALIGN_CONFIG.copy())
