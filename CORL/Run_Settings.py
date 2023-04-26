RUN_SETTINGS = {
    "Dataset": ["load"], # "BPM", "load", "random"
    "Train": ["load","online"], #["offline","online"], # ["offline","load","online"]
    "LoadModel": "Saved_Models/AWAC-LargeDataset-OfflineTrain2-04-26_1316/offline_best.pt",#"Saved_Models/AWAC-04-24_1624/epoch_-50.pt",
    "LoadDataset": "offline_data/CrissCross4_100x1000.data",
    "Test": []#["from_train", "best"]#["from_train", "best"]
}

