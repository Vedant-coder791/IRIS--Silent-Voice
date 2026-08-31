import os
import json
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/Users/vedantdwivedi/Desktop/IRIS/emg_data/closed_vocab/silent/5-19_silent"

# Start with a labeled recording
RECORDING_ID = "102"

EMG_FILE = os.path.join(BASE_DIR, f"{RECORDING_ID}_emg.npy")
INFO_FILE = os.path.join(BASE_DIR, f"{RECORDING_ID}_info.json")


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 70)
print("BERKELEY CHUNK STRUCTURE INSPECTOR")
print("=" * 70)

print(f"\nRecording ID: {RECORDING_ID}")
print(f"EMG file:    {EMG_FILE}")
print(f"INFO file:   {INFO_FILE}")


if not os.path.exists(EMG_FILE):
    raise FileNotFoundError(f"EMG file not found:\n{EMG_FILE}")

if not os.path.exists(INFO_FILE):
    raise FileNotFoundError(f"INFO file not found:\n{INFO_FILE}")


emg = np.load(EMG_FILE)

with open(INFO_FILE, "r") as f:
    info = json.load(f)


# ============================================================
# BASIC EMG INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("EMG INFORMATION")
print("=" * 70)

print("EMG shape:", emg.shape)
print("Number of samples:", emg.shape[0])
print("Number of channels:", emg.shape[1])

emg_length = emg.shape[0]


# ============================================================
# JSON STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("JSON STRUCTURE")
print("=" * 70)

print("\nTop-level keys:")

for key in info.keys():
    value = info[key]

    print(f"\n--- {key} ---")
    print("Python type:", type(value).__name__)

    if isinstance(value, list):
        print("Length:", len(value))

        if len(value) > 0:
            print("First item:", repr(value[0]))

    elif isinstance(value, dict):
        print("Number of keys:", len(value))
        print("Keys:", list(value.keys())[:20])

    else:
        print("Value:", repr(value))


# ============================================================
# TEXT / WORD INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TEXT / WORD INFORMATION")
print("=" * 70)

text = info.get("text", "")

print("\nTEXT:")
print(repr(text))

words = text.upper().split()

print("\nWORDS:")
for i, word in enumerate(words):
    print(f"Word {i + 1}: {word}")

print("\nNumber of words:", len(words))


# ============================================================
# CHUNK INFORMATION
# ============================================================

chunks = info.get("chunks", [])

print("\n" + "=" * 70)
print("CHUNK INFORMATION")
print("=" * 70)

print("\nNumber of chunks:", len(chunks))

if len(chunks) == 0:
    print("WARNING: No chunks found.")
    raise SystemExit


# ============================================================
# PRINT ALL CHUNKS
# ============================================================

print("\n" + "-" * 70)
print("ALL CHUNKS")
print("-" * 70)

print(
    f"{'Chunk':>8} "
    f"{'Value 1':>12} "
    f"{'Value 2':>12} "
    f"{'Value 3':>12} "
    f"{'Sum':>12}"
)

print("-" * 70)

for i, chunk in enumerate(chunks):

    if isinstance(chunk, (list, tuple)) and len(chunk) >= 3:

        a = float(chunk[0])
        b = float(chunk[1])
        c = float(chunk[2])

        print(
            f"{i + 1:8d} "
            f"{a:12.2f} "
            f"{b:12.2f} "
            f"{c:12.2f} "
            f"{a + b + c:12.2f}"
        )

    else:
        print(f"{i + 1:8d} {repr(chunk)}")


# ============================================================
# DETAILED CHUNK ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DETAILED CHUNK ANALYSIS")
print("=" * 70)


for i, chunk in enumerate(chunks):

    if not isinstance(chunk, (list, tuple)) or len(chunk) < 3:
        continue

    a = float(chunk[0])
    b = float(chunk[1])
    c = float(chunk[2])

    print(f"\nChunk {i + 1}")
    print("-" * 40)

    print(f"Value 1: {a}")
    print(f"Value 2: {b}")
    print(f"Value 3: {c}")

    print(f"Value 1 + Value 2 + Value 3: {a + b + c}")

    # Useful ratios
    if b != 0:
        print(f"Value 1 / Value 2: {a / b:.6f}")
        print(f"Value 3 / Value 2: {c / b:.6f}")

    # Possible relationship to EMG samples
    print(f"Value 2 / EMG length: {b / emg_length:.6f}")
    print(f"Value 2 * 16 / EMG length: {(b * 16) / emg_length:.6f}")

    # Convert the values assuming 600 Hz EMG
    FS = 600

    print(f"Value 1 as seconds @ {FS} Hz: {a / FS:.4f}")
    print(f"Value 2 as seconds @ {FS} Hz: {b / FS:.4f}")
    print(f"Value 3 as seconds @ {FS} Hz: {c / FS:.4f}")


# ============================================================
# CHUNK SUM ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CHUNK SUM ANALYSIS")
print("=" * 70)

chunk_sums = []

for chunk in chunks:

    if isinstance(chunk, (list, tuple)) and len(chunk) >= 3:
        chunk_sums.append(sum(float(x) for x in chunk[:3]))


chunk_sums = np.array(chunk_sums)

print("\nTotal chunk sum:", chunk_sums.sum())
print("EMG length:", emg_length)

print("\nRatio:")
print(chunk_sums.sum() / emg_length)

print("\nRatio / 16:")
print(chunk_sums.sum() / emg_length / 16)


# ============================================================
# SECOND-VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("SECOND VALUE ANALYSIS")
print("=" * 70)

second_values = []

for chunk in chunks:

    if isinstance(chunk, (list, tuple)) and len(chunk) >= 3:
        second_values.append(float(chunk[1]))

second_values = np.array(second_values)

print("\nSum of second values:", second_values.sum())
print("EMG length:", emg_length)

print("\nSum / EMG length:")
print(second_values.sum() / emg_length)

print("\nSum / (EMG length × 16):")
print(second_values.sum() / (emg_length * 16))


# ============================================================
# FIRST + THIRD VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FIRST / THIRD VALUE ANALYSIS")
print("=" * 70)

first_values = []
third_values = []

for chunk in chunks:

    if isinstance(chunk, (list, tuple)) and len(chunk) >= 3:
        first_values.append(float(chunk[0]))
        third_values.append(float(chunk[2]))

first_values = np.array(first_values)
third_values = np.array(third_values)

print("\nFirst-value sum:", first_values.sum())
print("Third-value sum:", third_values.sum())

print("\nFirst + Third sum:")
print(first_values.sum() + third_values.sum())

print("\nFirst + Third / EMG length:")
print((first_values.sum() + third_values.sum()) / emg_length)


# ============================================================
# CUMULATIVE SECOND VALUES
# ============================================================

print("\n" + "=" * 70)
print("CUMULATIVE SECOND-VALUE POSITIONS")
print("=" * 70)

cumulative = np.cumsum(second_values)

print(
    f"\n{'Chunk':>8} "
    f"{'Second':>12} "
    f"{'Cumulative':>15} "
    f"{'EMG equivalent':>18}"
)

print("-" * 60)

for i, (value, cum) in enumerate(zip(second_values, cumulative)):

    emg_equivalent = cum / 16.0

    print(
        f"{i + 1:8d} "
        f"{value:12.2f} "
        f"{cum:15.2f} "
        f"{emg_equivalent:18.2f}"
    )


# ============================================================
# CHECK WHETHER CHUNKS TILE THE RECORDING
# ============================================================

print("\n" + "=" * 70)
print("CHUNK COVERAGE TEST")
print("=" * 70)

coverage = cumulative[-1] / 16.0

print("\nEMG length:", emg_length)
print("Estimated coverage from second values:", coverage)

print("Difference:", coverage - emg_length)

if abs(coverage - emg_length) < 100:
    print("\n>>> VERY CLOSE MATCH <<<")
    print("The second chunk value may represent a higher-rate")
    print("time/sample quantity corresponding to the EMG recording.")

else:
    print("\nNo direct match found using this interpretation.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
Recording:
    {RECORDING_ID}

EMG:
    {emg_length} samples
    {emg.shape[1]} channels

Text:
    {repr(text)}

Words:
    {words}

Number of words:
    {len(words)}

Number of chunks:
    {len(chunks)}

Chunk sum:
    {chunk_sums.sum():.2f}

Chunk sum / EMG:
    {chunk_sums.sum() / emg_length:.6f}

Second-value sum:
    {second_values.sum():.2f}

Second-value sum / EMG:
    {second_values.sum() / emg_length:.6f}

Second-value sum / (EMG × 16):
    {second_values.sum() / (emg_length * 16):.6f}
""")

print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print("""
IMPORTANT:
Do NOT train the MLP from this output yet.

The purpose of this script is to determine what the
chunk representation means.

Send me the COMPLETE output from this script.
""")