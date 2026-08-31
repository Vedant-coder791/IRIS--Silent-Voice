
import os
import glob


EMG_ROOT = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "emg_data/closed_vocab/silent"
)

TEXTGRID_ROOT = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "silent_speech_alignments/text_alignments/5-9"
)


print("=" * 70)
print("BERKELEY FILE MATCHING DIAGNOSTIC")
print("=" * 70)


# ==============================================================
# FIND EMG
# ==============================================================

emg_files = sorted(
    glob.glob(
        os.path.join(
            EMG_ROOT,
            "**",
            "*.npy"
        ),
        recursive=True
    )
)


# ==============================================================
# FIND TEXTGRIDS
# ==============================================================

textgrid_files = sorted(
    glob.glob(
        os.path.join(
            TEXTGRID_ROOT,
            "**",
            "*.TextGrid"
        ),
        recursive=True
    )
)


print()
print("EMG files:", len(emg_files))
print("TextGrid files:", len(textgrid_files))


# ==============================================================
# SHOW EMG FILES
# ==============================================================

print()
print("=" * 70)
print("FIRST 30 EMG FILES")
print("=" * 70)

for path in emg_files[:30]:

    print(
        os.path.relpath(
            path,
            EMG_ROOT
        )
    )


# ==============================================================
# SHOW TEXTGRID FILES
# ==============================================================

print()
print("=" * 70)
print("FIRST 30 TEXTGRID FILES")
print("=" * 70)

for path in textgrid_files[:30]:

    print(
        os.path.relpath(
            path,
            TEXTGRID_ROOT
        )
    )


# ==============================================================
# BASENAME COMPARISON
# ==============================================================

print()
print("=" * 70)
print("BASENAME COMPARISON")
print("=" * 70)


print()
print("EMG basenames:")

for path in emg_files[:20]:

    print(
        os.path.basename(path)
    )


print()
print("TextGrid basenames:")

for path in textgrid_files[:20]:

    print(
        os.path.basename(path)
    )


# ==============================================================
# EXTRACT DIGITS FROM FILENAMES
# ==============================================================

def get_numbers(filename):

    base = os.path.basename(filename)

    numbers = []

    current = ""

    for char in base:

        if char.isdigit():

            current += char

        else:

            if current:

                numbers.append(current)
                current = ""

    if current:
        numbers.append(current)

    return numbers


print()
print("=" * 70)
print("NUMBER PATTERNS")
print("=" * 70)


print()
print("EMG:")

for path in emg_files[:30]:

    print(
        os.path.basename(path),
        "->",
        get_numbers(path)
    )


print()
print("TextGrid:")

for path in textgrid_files[:30]:

    print(
        os.path.basename(path),
        "->",
        get_numbers(path)
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)

