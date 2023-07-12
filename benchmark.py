from main import run

# Some global parameters
GLOBAL_CONFIG = {
    "Data": "./data/feb14_2.tsv",
    "Overlap": 0.80,
    "DevMode": False,  # Development Mode, saves some intermediate results to the /dev directory
    "BenchMode": True,  # Benchmark Mode
    "Verbose": False  # Print Status Messages?
}

# Configuration for Bloom filters
ENC_CONFIG = {
    "AliceSecret": "SuperSecretSalt1337",
    "AliceBFLength": 1024,
    "AliceBits": 30,
    "AliceN": 2,
    "AliceMetric": "dice",
    "EveSecret": "ATotallyDifferentString",
    "EveBFLength": 1024,
    "EveBits": 30,
    "EveN": 2,
    "EveMetric": "dice",
    "Data": GLOBAL_CONFIG["Data"],
    "Overlap": GLOBAL_CONFIG["Overlap"]
}

EMB_CONFIG = {
    "AliceWalkLen": 100,
    "AliceNWalks": 20,
    "AliceP": 0.5,  # 0.5
    "AliceQ": 2,  # 2
    "AliceDim": 64,
    "AliceContext": 10,
    "AliceEpochs": 30,
    "AliceQuantile": 0.95,  # 0.99
    "AliceDiscretize": True,
    "AliceSeed": 42,
    "EveWalkLen": 100,
    "EveNWalks": 20,
    "EveP": 0.5,  # 0.5
    "EveQ": 2,  # 2
    "EveDim": 64,
    "EveContext": 10,
    "EveEpochs": 30,
    "EveQuantile": 0.95,  # 0.99
    "EveDiscretize": True,
    "EveSeed": 42,
    "Workers": -1,
    "Data": GLOBAL_CONFIG["Data"],
    "Overlap": GLOBAL_CONFIG["Overlap"]
}

ALIGN_CONFIG = {
    "Maxload": 200000,
    "RegWS": 0.9,
    "RegInit": 0.2,
    "Batchsize": 1000,
    "LR": 70.0,
    "NIterWS": 500,
    "NIterInit": 800,  # 800
    "NEpochWS": 150,
    "VocabSize": 1000,
    "LRDecay": 0.9,
    "Sqrt": True,
    "EarlyStopping": 2,
    "Selection": "Degree",
    "Wasserstein": True,
    "Verbose": GLOBAL_CONFIG["Verbose"]
}

# Global params
overlap = [i/100 for i in range(50, 105, 5)]

# Embedding Config
thresholds = [0.8, 0.9, 0.95, 0.99, 0.995]
discretize = [True, False]
walk_len = [150, 250]
ps = [0.2, 0.5, 1, 2]
qs = [0.5, 1, 2, 3]

for o in overlap[1:]:
    for t in thresholds:
        for l in walk_len:
            for p in ps:
                for q in qs:
                    GLOBAL_CONFIG["Overlap"] = o
                    EMB_CONFIG["Overlap"] = o
                    EMB_CONFIG["AliceQuantile"] = t
                    EMB_CONFIG["EveQuantile"] = t
                    EMB_CONFIG["AliceWalkLen"] = l
                    EMB_CONFIG["EveWalkLen"] = l
                    EMB_CONFIG["AliceP"] = p
                    EMB_CONFIG["EveP"] = p
                    EMB_CONFIG["AliceQ"] = q
                    EMB_CONFIG["EveQ"] = q
                    run(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG)



