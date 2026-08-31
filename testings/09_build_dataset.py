import os


# ==========================================
# CORPUS PATH
# ==========================================

corpus_path = "/Users/vedantdwivedi/Desktop/IRIS/EMG-UKA-Trial-Corpus"


# ==========================================
# SEARCH FOR MATCHING FILES
# ==========================================

target = "002_001_0100"

print("Searching for:", target)
print()


for root, dirs, files in os.walk(corpus_path):

    for file in files:

        if target in file:

            full_path = os.path.join(root, file)

            print(full_path)