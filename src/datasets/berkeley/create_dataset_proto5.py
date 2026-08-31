import os
import numpy as np
import librosa
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROTO 5 DATASET GENERATOR
# ============================================================

print("\n==========================================")
print("PROTO 5 DATASET GENERATION")
print("==========================================\n")


# ============================================================
# SETTINGS
# ============================================================

CORPUS_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "EMG-UKA-Trial-Corpus"
)

OUTPUT_FILE = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "proto5_dataset.npz"
)

FS = 600

N_FFT = 64
HOP_LENGTH = 32
N_MFCC = 13
N_MELS = 20

FMIN = 20
FMAX = 250


# ============================================================
# TARGET WORDS
# ============================================================

TARGET_WORDS = {
    "THE",
    "TO",
    "OF",
    "IN",
    "ARE",
    "AND",
    "IS",
    "THEY",
    "THAT",
    "THIS",
    "HAS",
    "HE",
    "I",
    "ON",
    "WERE",
    "WE",
    "IT",
    "YOU",
    "STATE",
    "NEW"
}


# ============================================================
# SUBJECT SPLIT
# ============================================================

TRAIN_SUBJECTS = ["002", "008"]
VALIDATION_SUBJECTS = ["004"]
TEST_SUBJECTS = ["006"]


# ============================================================
# READ ADC FILE
# ============================================================

def read_adc(file_path):

    data = np.fromfile(
        file_path,
        dtype=np.int16
    )

    if data.size == 0:
        raise ValueError(
            "ADC file contains no data"
        )

    # EMG-UKA recording format:
    # 6 EMG channels + 1 additional value
    if data.size % 7 != 0:
        raise ValueError(
            f"Data size {data.size} "
            "is not divisible by 7"
        )

    data = data.reshape(-1, 7)

    # Keep the six EMG channels
    data = data[:, :6]

    # channels × samples
    emg = data.T

    if emg.shape[0] != 6:
        raise ValueError(
            f"Expected 6 channels, "
            f"got {emg.shape[0]}"
        )

    if not np.all(np.isfinite(emg)):
        raise ValueError(
            "EMG contains NaN or infinite values"
        )

    return emg.astype(np.float32)


# ============================================================
# READ WORD ALIGNMENT
# ============================================================

def read_word_alignment(word_file):

    words = []

    with open(
        word_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 3:
                continue

            try:

                start = int(parts[0])
                end = int(parts[1])

            except ValueError:

                continue

            word = parts[2].upper()

            words.append(
                (start, end, word)
            )

    return words


# ============================================================
# FIND TARGET WORDS
# ============================================================

def find_target_words(words):

    targets = []

    for start, end, word in words:

        if word in TARGET_WORDS:

            targets.append(
                (start, end, word)
            )

    return targets


# ============================================================
# FIND RECORDINGS
# ============================================================

def find_recordings():

    recordings = []

    alignment_root = os.path.join(
        CORPUS_PATH,
        "Alignments"
    )

    if not os.path.isdir(alignment_root):

        raise FileNotFoundError(
            f"Alignment folder not found:\n"
            f"{alignment_root}"
        )

    for root, dirs, files in os.walk(
        alignment_root
    ):

        for filename in files:

            if not (
                filename.startswith("words_")
                and filename.endswith(".txt")
            ):
                continue

            word_file = os.path.join(
                root,
                filename
            )

            # ------------------------------------------------
            # READ ALIGNMENT
            # ------------------------------------------------

            try:

                words = read_word_alignment(
                    word_file
                )

            except Exception:

                continue

            targets = find_target_words(
                words
            )

            if not targets:
                continue


            # ------------------------------------------------
            # SUBJECT / SESSION
            # ------------------------------------------------

            session = os.path.basename(root)

            subject = os.path.basename(
                os.path.dirname(root)
            )


            # ------------------------------------------------
            # RECORDING IDENTIFIER
            # ------------------------------------------------

            identifier = (
                filename
                .replace("words_", "")
                .replace(".txt", "")
            )


            # ------------------------------------------------
            # EMG FILE
            # ------------------------------------------------

            emg_folder = os.path.join(
                CORPUS_PATH,
                "emg",
                subject,
                session
            )

            emg_filename = (
                "e07_"
                + identifier
                + ".adc"
            )

            emg_file = os.path.join(
                emg_folder,
                emg_filename
            )


            # ------------------------------------------------
            # CHECK FILE EXISTS
            # ------------------------------------------------

            if not os.path.isfile(
                emg_file
            ):
                continue


            recordings.append({

                "emg_file": emg_file,

                "word_file": word_file,

                "subject": subject,

                "session": session,

                "recording": identifier,

                "targets": targets

            })

    return recordings


# ============================================================
# VALIDATE RECORDINGS
# ============================================================

def validate_recordings(recordings):

    valid_recordings = []

    failed_count = 0

    failure_reasons = {}

    print("\n==========================================")
    print("FILTERING RECORDINGS")
    print("==========================================")

    print(
        "\nCandidate recordings:",
        len(recordings)
    )

    for i, recording in enumerate(
        recordings
    ):

        try:

            # Actually read the complete ADC file.
            # This is the filtering step.
            emg = read_adc(
                recording["emg_file"]
            )

            if emg.shape[0] != 6:

                raise ValueError(
                    "Incorrect channel count"
                )

            if emg.shape[1] < 32:

                raise ValueError(
                    "Recording too short"
                )

            recording["emg"] = emg

            valid_recordings.append(
                recording
            )

        except Exception as e:

            failed_count += 1

            reason = str(e)

            if reason not in failure_reasons:
                failure_reasons[reason] = 0

            failure_reasons[reason] += 1

        if (i + 1) % 100 == 0:

            print(
                f"Checked "
                f"{i + 1}/{len(recordings)}"
            )


    # --------------------------------------------------------
    # FILTERING SUMMARY
    # --------------------------------------------------------

    print("\n------------------------------------------")
    print("FILTERING RESULTS")
    print("------------------------------------------")

    print(
        "Candidate recordings:",
        len(recordings)
    )

    print(
        "Valid recordings:",
        len(valid_recordings)
    )

    print(
        "Failed recordings:",
        failed_count
    )

    if failure_reasons:

        print(
            "\nFailure reasons:"
        )

        for reason, count in sorted(
            failure_reasons.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            print(
                f"{count:5d} : {reason}"
            )

    return valid_recordings


# ============================================================
# EXTRACT WORD SEGMENTS
# ============================================================

def extract_word_segments(
    valid_recordings
):

    segments = []
    labels = []
    subjects = []

    print("\n==========================================")
    print("EXTRACTING WORD SEGMENTS")
    print("==========================================")

    total_targets = sum(
        len(r["targets"])
        for r in valid_recordings
    )

    processed = 0

    for recording in valid_recordings:

        emg = recording["emg"]

        num_samples = emg.shape[1]

        for start, end, word in recording[
            "targets"
        ]:

            processed += 1

            # ------------------------------------------------
            # Alignment frame conversion
            #
            # EMG-UKA alignment times are represented
            # in 10 ms frames.
            #
            # 600 Hz × 0.010 s = 6 samples/frame
            # ------------------------------------------------

            start_sample = int(
                start * FS * 0.010
            )

            end_sample = int(
                end * FS * 0.010
            )

            # Safety checks
            start_sample = max(
                0,
                start_sample
            )

            end_sample = min(
                num_samples,
                end_sample
            )

            if end_sample <= start_sample:

                continue

            segment = emg[
                :,
                start_sample:end_sample
            ]

            if segment.shape[1] < 32:

                continue

            if not np.all(
                np.isfinite(segment)
            ):

                continue

            segments.append(
                segment.astype(
                    np.float32
                )
            )

            labels.append(
                word
            )

            subjects.append(
                recording["subject"]
            )

        if processed % 100 == 0:

            print(
                f"Processed "
                f"{processed}/{total_targets}"
            )

    print("\n------------------------------------------")
    print("WORD SEGMENT RESULTS")
    print("------------------------------------------")

    print(
        "Segments extracted:",
        len(segments)
    )

    print(
        "Labels:",
        len(labels)
    )

    print(
        "Subjects:",
        len(subjects)
    )

    return (
        segments,
        labels,
        subjects
    )


# ============================================================
# MFCC
# ============================================================

def extract_mfcc(
    signal,
    fs=FS
):

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    mfcc = librosa.feature.mfcc(

        y=signal,

        sr=fs,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        n_mfcc=N_MFCC,

        n_mels=N_MELS,

        fmin=FMIN,

        fmax=FMAX

    )

    return mfcc


# ============================================================
# TIME FEATURES
# ============================================================

def extract_time_features(
    signal
):

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    mav = np.mean(
        np.abs(signal)
    )

    rms = np.sqrt(
        np.mean(signal ** 2)
    )

    waveform_length = np.sum(
        np.abs(
            np.diff(signal)
        )
    )

    zero_crossings = np.sum(
        np.diff(
            np.signbit(signal)
        )
    )

    zero_crossing_rate = (
        zero_crossings
        / max(len(signal), 1)
    )

    return np.array(
        [
            mav,
            rms,
            waveform_length,
            zero_crossing_rate
        ],
        dtype=np.float32
    )


# ============================================================
# FREQUENCY FEATURES
# ============================================================

def extract_frequency_features(
    signal,
    fs=FS
):

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    frequencies, power = welch(

        signal,

        fs=fs,

        nperseg=min(
            64,
            len(signal)
        )

    )

    total_power = np.sum(
        power
    )

    if (
        total_power <= 0
        or not np.isfinite(total_power)
    ):

        spectral_centroid = 0.0

        median_frequency = 0.0

        total_power = 0.0

    else:

        spectral_centroid = (

            np.sum(
                frequencies * power
            )
            / total_power

        )

        cumulative_power = np.cumsum(
            power
        )

        median_index = np.searchsorted(

            cumulative_power,

            total_power / 2

        )

        median_index = min(

            median_index,

            len(frequencies) - 1

        )

        median_frequency = (
            frequencies[median_index]
        )

    return np.array(
        [
            total_power,
            spectral_centroid,
            median_frequency
        ],
        dtype=np.float32
    )


# ============================================================
# PROTO 5 CHANNEL FEATURES
# ============================================================

def extract_channel_features(
    signal,
    fs=FS
):

    mfcc = extract_mfcc(
        signal,
        fs
    )

    time_features = (
        extract_time_features(
            signal
        )
    )

    frequency_features = (
        extract_frequency_features(
            signal,
            fs
        )
    )

    return (
        mfcc,
        time_features,
        frequency_features
    )


# ============================================================
# FIXED-SIZE FEATURE VECTOR
# ============================================================

def extract_window_features(
    segment,
    fs=FS
):

    feature_vector = []

    for channel in segment:

        mfcc, time_features, frequency_features = (
            extract_channel_features(
                channel,
                fs
            )
        )

        # ----------------------------------------------------
        # MFCC FIXED-SIZE REPRESENTATION
        #
        # Mean + standard deviation
        # for each of the 13 MFCC coefficients.
        #
        # 13 × 2 = 26
        #
        # Time = 4
        # Frequency = 3
        #
        # Total per channel = 33
        #
        # 6 channels × 33 = 198
        # ----------------------------------------------------

        mfcc_mean = np.mean(
            mfcc,
            axis=1
        )

        mfcc_std = np.std(
            mfcc,
            axis=1
        )

        feature_vector.extend(
            mfcc_mean
        )

        feature_vector.extend(
            mfcc_std
        )

        feature_vector.extend(
            time_features
        )

        feature_vector.extend(
            frequency_features
        )

    return np.asarray(
        feature_vector,
        dtype=np.float32
    )


# ============================================================
# EXTRACT ALL FEATURES
# ============================================================

def create_feature_dataset(
    segments
):

    X = []

    print("\n==========================================")
    print("EXTRACTING PROTO 5 FEATURES")
    print("==========================================")

    total = len(segments)

    for i, segment in enumerate(
        segments
    ):

        try:

            features = (
                extract_window_features(
                    segment
                )
            )

            if len(features) != 198:

                raise ValueError(
                    f"Expected 198 features, "
                    f"got {len(features)}"
                )

            X.append(
                features
            )

        except Exception as e:

            print(
                f"\nFeature extraction failed "
                f"for segment {i}:"
            )

            print(e)

            raise

        if (i + 1) % 100 == 0:

            print(
                f"Processed "
                f"{i + 1}/{total}"
            )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    print("\n------------------------------------------")
    print("FEATURE DATASET")
    print("------------------------------------------")

    print(
        "X shape:",
        X.shape
    )

    return X


# ============================================================
# CREATE LABEL ENCODING
# ============================================================

def encode_labels(
    labels
):

    classes = sorted(
        list(
            set(labels)
        )
    )

    class_to_index = {
        word: i
        for i, word in enumerate(classes)
    }

    y = np.asarray(
        [
            class_to_index[word]
            for word in labels
        ],
        dtype=np.int64
    )

    return (
        y,
        classes
    )


# ============================================================
# SUBJECT-AWARE SPLIT
# ============================================================

def subject_split(
    X,
    y,
    subjects
):

    subjects = np.asarray(
        subjects
    )

    train_mask = np.isin(
        subjects,
        TRAIN_SUBJECTS
    )

    validation_mask = np.isin(
        subjects,
        VALIDATION_SUBJECTS
    )

    test_mask = np.isin(
        subjects,
        TEST_SUBJECTS
    )

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_validation = X[
        validation_mask
    ]

    y_validation = y[
        validation_mask
    ]

    X_test = X[test_mask]
    y_test = y[test_mask]

    subjects_train = subjects[
        train_mask
    ]

    subjects_validation = subjects[
        validation_mask
    ]

    subjects_test = subjects[
        test_mask
    ]

    print("\n==========================================")
    print("SUBJECT-AWARE SPLIT")
    print("==========================================")

    print(
        "\nTraining subjects:",
        TRAIN_SUBJECTS
    )

    print(
        "Training shape:",
        X_train.shape
    )

    print(
        "\nValidation subject:",
        VALIDATION_SUBJECTS
    )

    print(
        "Validation shape:",
        X_validation.shape
    )

    print(
        "\nTesting subject:",
        TEST_SUBJECTS
    )

    print(
        "Testing shape:",
        X_test.shape
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        subjects_train,
        subjects_validation,
        subjects_test
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_dataset(
    X_train,
    X_validation,
    X_test
):

    print("\n==========================================")
    print("NORMALIZATION")
    print("==========================================")

    scaler = StandardScaler()

    # IMPORTANT:
    # Fit ONLY on training data.
    X_train = scaler.fit_transform(
        X_train
    )

    X_validation = scaler.transform(
        X_validation
    )

    X_test = scaler.transform(
        X_test
    )

    print(
        "Training mean:",
        np.mean(X_train)
    )

    print(
        "Training standard deviation:",
        np.std(X_train)
    )

    return (
        X_train.astype(np.float32),
        X_validation.astype(np.float32),
        X_test.astype(np.float32)
    )


# ============================================================
# FINAL DATA CHECK
# ============================================================

def check_dataset(
    X_train,
    X_validation,
    X_test,
    y_train,
    y_validation,
    y_test
):

    print("\n==========================================")
    print("FINAL DATA CHECK")
    print("==========================================")

    print(
        "Training finite:",
        np.all(
            np.isfinite(X_train)
        )
    )

    print(
        "Validation finite:",
        np.all(
            np.isfinite(X_validation)
        )
    )

    print(
        "Testing finite:",
        np.all(
            np.isfinite(X_test)
        )
    )

    if not np.all(
        np.isfinite(X_train)
    ):

        raise ValueError(
            "Training data contains "
            "NaN or infinite values"
        )

    if not np.all(
        np.isfinite(X_validation)
    ):

        raise ValueError(
            "Validation data contains "
            "NaN or infinite values"
        )

    if not np.all(
        np.isfinite(X_test)
    ):

        raise ValueError(
            "Test data contains "
            "NaN or infinite values"
        )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_distribution(
    y,
    classes,
    name
):

    print(
        f"\n{name}"
    )

    for i, word in enumerate(
        classes
    ):

        count = np.sum(
            y == i
        )

        print(
            f"{word:8s}: {count:5d}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. FIND CANDIDATE RECORDINGS
    # --------------------------------------------------------

    recordings = find_recordings()


    # --------------------------------------------------------
    # 2. FILTER / VALIDATE RECORDINGS
    # --------------------------------------------------------

    valid_recordings = (
        validate_recordings(
            recordings
        )
    )


    # --------------------------------------------------------
    # 3. EXTRACT WORD SEGMENTS
    # --------------------------------------------------------

    (
        segments,
        labels,
        subjects
    ) = extract_word_segments(
        valid_recordings
    )


    if len(segments) == 0:

        raise RuntimeError(
            "No valid word segments were extracted."
        )


    # --------------------------------------------------------
    # 4. EXTRACT PROTO 5 FEATURES
    # --------------------------------------------------------

    X = create_feature_dataset(
        segments
    )


    # --------------------------------------------------------
    # 5. ENCODE LABELS
    # --------------------------------------------------------

    (
        y,
        classes
    ) = encode_labels(
        labels
    )


    print("\n==========================================")
    print("WORD CLASSES")
    print("==========================================")

    for i, word in enumerate(
        classes
    ):

        print(
            f"{i:2d} -> {word}"
        )


    # --------------------------------------------------------
    # 6. SUBJECT SPLIT
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        subjects_train,
        subjects_validation,
        subjects_test
    ) = subject_split(
        X,
        y,
        subjects
    )


    # --------------------------------------------------------
    # 7. NORMALIZATION
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test
    ) = normalize_dataset(
        X_train,
        X_validation,
        X_test
    )


    # --------------------------------------------------------
    # 8. FINAL CHECK
    # --------------------------------------------------------

    check_dataset(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )


    # --------------------------------------------------------
    # 9. DISTRIBUTIONS
    # --------------------------------------------------------

    print("\n==========================================")
    print("CLASS DISTRIBUTIONS")
    print("==========================================")

    print_distribution(
        y_train,
        classes,
        "TRAINING"
    )

    print_distribution(
        y_validation,
        classes,
        "VALIDATION"
    )

    print_distribution(
        y_test,
        classes,
        "TEST"
    )


    # --------------------------------------------------------
    # 10. SAVE
    # --------------------------------------------------------

    print("\n==========================================")
    print("SAVING PROTO 5 DATASET")
    print("==========================================")

    np.savez_compressed(

        OUTPUT_FILE,

        X_train=X_train,

        X_validation=X_validation,

        X_test=X_test,

        y_train=y_train,

        y_validation=y_validation,

        y_test=y_test,

        subjects_train=subjects_train,

        subjects_validation=subjects_validation,

        subjects_test=subjects_test,

        classes=np.asarray(
            classes
        )

    )

    print(
        "\nDataset saved successfully!"
    )

    print(
        "File:",
        OUTPUT_FILE
    )


    # --------------------------------------------------------
    # 11. SUMMARY
    # --------------------------------------------------------

    print("\n==========================================")
    print("PROTO 5 DATASET SUMMARY")
    print("==========================================")

    print(
        "Number of classes:",
        len(classes)
    )

    print(
        "Number of input features:",
        X_train.shape[1]
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_validation)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    print(
        "\nProto 5 dataset generation complete!"
    )