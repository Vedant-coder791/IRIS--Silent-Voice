phone_file = "/Users/vedantdwivedi/Desktop/IRIS/EMG-UKA-Trial-Corpus/Alignments/002/001/phones_002_001_0100.txt"

with open(phone_file, "r") as f:
    lines = f.readlines()

# Remove empty lines
lines = [line.strip() for line in lines if line.strip()]

print("Number of phoneme entries:", len(lines))

# First and last entries
print("\nFirst entry:")
print(lines[0])

print("\nLast entry:")
print(lines[-1])

# Get final frame
last_parts = lines[-1].split()
last_frame = int(last_parts[1])

print("\nLast annotated frame:", last_frame)
