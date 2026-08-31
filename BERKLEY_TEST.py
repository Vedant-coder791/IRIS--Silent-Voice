import os
import json
import glob


DATASET_ROOT = "/Users/vedantdwivedi/Desktop/IRIS/emg_data"


print("=" * 70)
print("BERKELEY DATASET STRUCTURE INSPECTION")
print("=" * 70)


# ============================================================
# TOP-LEVEL DIRECTORIES
# ============================================================

print("\nTOP-LEVEL CONTENTS")
print("-" * 70)

for item in sorted(os.listdir(DATASET_ROOT)):

    path = os.path.join(
        DATASET_ROOT,
        item
    )

    if os.path.isdir(path):

        print("[DIR ]", item)

    else:

        print("[FILE]", item)


# ============================================================
# ALL DIRECTORIES
# ============================================================

print("\n" + "=" * 70)
print("DATASET DIRECTORIES")
print("=" * 70)


directories = []

for root, dirs, files in os.walk(
    DATASET_ROOT
):

    for directory in dirs:

        full_path = os.path.join(
            root,
            directory
        )

        relative_path = os.path.relpath(
            full_path,
            DATASET_ROOT
        )

        directories.append(
            relative_path
        )


for directory in sorted(
    directories
):

    print(directory)


# ============================================================
# EMG FILES BY DIRECTORY
# ============================================================

print("\n" + "=" * 70)
print("EMG FILE COUNTS BY DIRECTORY")
print("=" * 70)


directory_counts = {}


for root, dirs, files in os.walk(
    DATASET_ROOT
):

    count = sum(
        1
        for f in files
        if f.endswith("_emg.npy")
    )

    if count > 0:

        relative_path = os.path.relpath(
            root,
            DATASET_ROOT
        )

        directory_counts[
            relative_path
        ] = count


for directory, count in sorted(
    directory_counts.items()
):

    print(
        f"{count:6d} EMG files  ->  {directory}"
    )


# ============================================================
# FIND INFO FILE
# ============================================================

print("\n" + "=" * 70)
print("INSPECTING FIRST INFO FILE")
print("=" * 70)


info_files = glob.glob(
    os.path.join(
        DATASET_ROOT,
        "**",
        "*_info.json"
    ),
    recursive=True
)


print(
    "\nInfo files found:",
    len(info_files)
)


if len(info_files) == 0:

    raise SystemExit(
        "No _info.json files found."
    )


info_path = info_files[0]


print(
    "\nSelected file:"
)

print(info_path)


# ============================================================
# PRINT RAW JSON
# ============================================================

print("\n" + "=" * 70)
print("RAW JSON CONTENT")
print("=" * 70)


with open(
    info_path,
    "r"
) as f:

    info = json.load(f)


print(
    json.dumps(
        info,
        indent=4
    )
)


# ============================================================
# JSON KEYS
# ============================================================

print("\n" + "=" * 70)
print("JSON KEYS")
print("=" * 70)


for key in info.keys():

    print(
        key,
        "->",
        type(info[key]).__name__,
        "->",
        repr(info[key])
    )


# ============================================================
# CORRESPONDING EMG
# ============================================================

emg_path = info_path.replace(
    "_info.json",
    "_emg.npy"
)


print("\n" + "=" * 70)
print("CORRESPONDING EMG")
print("=" * 70)


print(
    "EMG:",
    emg_path
)


if os.path.exists(emg_path):

    import numpy as np

    emg = np.load(
        emg_path
    )

    print(
        "Shape:",
        emg.shape
    )

    print(
        "dtype:",
        emg.dtype
    )

    print(
        "Minimum:",
        np.min(emg)
    )

    print(
        "Maximum:",
        np.max(emg)
    )

else:

    print(
        "Corresponding EMG file not found."
    )


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)