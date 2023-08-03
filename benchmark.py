from main import run

GLOBAL_CONFIG = {
    "Data": "./data/fakename_5k.tsv",
    "Overlap": 0.4,
    "DropFrom": "Alice",
    "DevMode": False,  # Development Mode, saves some intermediate results to the /dev directory
    "BenchMode": True,  # Benchmark Mode
    "Verbose": False,  # Print Status Messages?
    "MatchingMetric": "euclidean",
    "Matching": "NearestNeighbor"
}

ENC_CONFIG = {
    "AliceAlgo": "TwoStepHash",
    "AliceSecret": "SuperSecretSalt1337",
    "AliceBFLength": 1024,
    "AliceBits": 30,  # BF: 30, TMH: 1000
    "AliceN": 2,
    "AliceMetric": "dice",
    "EveAlgo": "TwoStepHash",
    "EveSecret": "ATotallyDifferentString",
    "EveBFLength": 1024,
    "EveBits": 30,  # BF: 30, TMH: 1000
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
    "Algo": "NetMF",
    "AliceQuantile": 0.1,
    "AliceDiscretize": False,
    "AliceDim": 80,
    "AliceContext": 10,
    "AliceNegative": 1,
    "AliceNormalize": True,
    "EveQuantile": 0.1,
    "EveDiscretize": False,
    "EveDim": 80,
    "EveContext": 10,
    "EveNegative": 1,
    "EveNormalize": True,
    "Workers": -1,
    # For Node2Vec
    "AliceWalkLen": 100,
    "AliceNWalks": 20,
    "AliceP": 250,  # 0.5
    "AliceQ": 300,  # 2
    "AliceEpochs": 5,
    "AliceSeed": 42,
    "EveWalkLen": 100,
    "EveNWalks": 20,
    "EveP": 250,  # 0.5
    "EveQ": 300,  # 2
    "EveEpochs": 5,
    "EveSeed": 42,
}

ALIGN_CONFIG = {
    "RegWS": "Auto",
    "RegInit": 0.25, # For BF 1
    "Batchsize": "Auto",
    "LR": 300.0,
    "NIterWS": 5,
    "NIterInit": 50,  # For BF: 10
    "NEpochWS": 200,
    "LRDecay": 0.9,
    "Sqrt": False,
    "EarlyStopping": 20,
    "Selection": "None",
    "Wasserstein": True,
}

# Global params
datasets = ["fakename_1k.tsv", "fakename_2k.tsv", "fakename_5k.tsv", "fakename_10k.tsv"]
    #, "fakename_20k.tsv", "fakename_50k.tsv", "fakename_100k.tsv"]
overlap = [i/100 for i in range(10, 105, 5)]

for d in datasets:
    GLOBAL_CONFIG["Data"] = "./data/" + d
    if d == "fakename_1k.tsv":
        EMB_CONFIG["AliceDim"] = 80
        EMB_CONFIG["EveDim"] = 80
        GLOBAL_CONFIG["Matching"] = "MinWeight"
    elif d == "fakename_2k.tsv":
        EMB_CONFIG["AliceDim"] = 100
        EMB_CONFIG["EveDim"] = 100
        GLOBAL_CONFIG["Matching"] = "MinWeight" # For 2SH: Nearest Neighbor
    elif d == "fakename_20k.tsv":
        EMB_CONFIG["AliceDim"] = 128
        EMB_CONFIG["EveDim"] = 120
        GLOBAL_CONFIG["Matching"] = "NearestNeighbor"
    elif d == "fakename_50k.tsv":
        EMB_CONFIG["AliceDim"] = 180
        EMB_CONFIG["EveDim"] = 180
        GLOBAL_CONFIG["Matching"] = "NearestNeighbor"
    elif d == "fakename_100k.tsv":
        EMB_CONFIG["AliceDim"] = 200
        EMB_CONFIG["EveDim"] = 200
        GLOBAL_CONFIG["Matching"] = "NearestNeighbor"
    else:
        EMB_CONFIG["AliceDim"] = 128
        EMB_CONFIG["EveDim"] = 128
        GLOBAL_CONFIG["Matching"] = "NearestNeighbor"

    for o in overlap:
        GLOBAL_CONFIG["Overlap"] = o
        run(GLOBAL_CONFIG.copy(), ENC_CONFIG.copy(), EMB_CONFIG.copy(), ALIGN_CONFIG.copy())



