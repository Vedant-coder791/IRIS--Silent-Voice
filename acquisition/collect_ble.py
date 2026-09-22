import asyncio
import csv
import struct
import sys
import termios
import tty
import select
from pathlib import Path

from bleak import BleakScanner, BleakClient


# ==========================================
# BLE SETTINGS
# ==========================================

DEVICE_NAME = "IRIS-EMG"

CHARACTERISTIC_UUID = (
    "beb5483e-36e1-4688-b7f5-ea07361b26a8"
)

SAMPLE_RATE = 600

SAMPLES_PER_PACKET = 10

# Packet:
# 4 bytes  = packet sequence
# 4 bytes  = timestamp of first sample
# 80 bytes = 10 samples × 4 channels × 2 bytes
PACKET_SIZE = 88

SAMPLE_INTERVAL_US = 1_000_000 / SAMPLE_RATE


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_ROOT = PROJECT_ROOT / "Data_original" / "raw"

METADATA_FILE = RAW_ROOT / "recordings_metadata.csv"


# ==========================================
# EXPERIMENT INFORMATION
# ==========================================

participant = input(
    "Participant ID (e.g. P01): "
).strip()

session = input(
    "Session number (e.g. 01): "
).strip()

word = input(
    "Word (e.g. HELP): "
).strip().upper()

condition = input(
    "Condition (SILENT/VOICED/WHISPERED/REST): "
).strip().upper()

trial = input(
    "Trial number (e.g. 001): "
).strip()


# ==========================================
# CREATE FOLDER
# ==========================================

folder = (
    RAW_ROOT
    / participant
    / f"session_{session}"
    / condition
    / word
)

folder.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# CREATE FILE
# ==========================================

filename = (
    f"{participant}_S{session}_"
    f"{condition}_{word}_T{trial}.csv"
)

filepath = folder / filename


# ==========================================
# PREVENT OVERWRITE
# ==========================================

if filepath.exists():

    print("\nERROR: This recording already exists!")

    print(f"File: {filepath}")

    print(
        "Please use a different trial number."
    )

    raise SystemExit


# ==========================================
# CREATE RAW DIRECTORY
# ==========================================

RAW_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# RECORDING STATE
# ==========================================

state = {
    "samples": [],
    "packet_count": 0,
    "invalid_packets": 0,
    "first_timestamp": None,
    "last_packet_sequence": None,
    "recording_enabled": False,
    "recording_finished": False
}


# ==========================================
# BLE CALLBACK
# ==========================================

def handle_data(sender, data):

    # --------------------------------------
    # Ignore data before recording starts
    # --------------------------------------

    if not state["recording_enabled"]:
        return

    # --------------------------------------
    # Check packet size
    # --------------------------------------

    if len(data) != PACKET_SIZE:

        state["invalid_packets"] += 1

        print(
            f"\nWARNING: Received "
            f"{len(data)} bytes "
            f"instead of {PACKET_SIZE}"
        )

        return

    # --------------------------------------
    # Packet header
    # --------------------------------------

    packet_sequence = struct.unpack_from(
        "<I",
        data,
        0
    )[0]

    first_sample_timestamp = struct.unpack_from(
        "<I",
        data,
        4
    )[0]

    # --------------------------------------
    # Check packet sequence
    # --------------------------------------

    previous_sequence = state[
        "last_packet_sequence"
    ]

    if (
        previous_sequence is not None
        and packet_sequence != previous_sequence + 1
    ):

        print(
            f"\nWARNING: BLE packet gap "
            f"{previous_sequence} -> "
            f"{packet_sequence}"
        )

    state["last_packet_sequence"] = (
        packet_sequence
    )

    # --------------------------------------
    # First packet
    # --------------------------------------

    if state["first_timestamp"] is None:

        state["first_timestamp"] = (
            first_sample_timestamp
        )

        print(
            "\nRecording started!"
        )

    # --------------------------------------
    # Decode 10 samples
    # --------------------------------------

    for i in range(SAMPLES_PER_PACKET):

        offset = 8 + (i * 8)

        ch1, ch2, ch3, ch4 = struct.unpack_from(
            "<HHHH",
            data,
            offset
        )

        timestamp = int(
            first_sample_timestamp
            + (
                i * SAMPLE_INTERVAL_US
            )
        )

        state["samples"].append([
            timestamp,
            ch1,
            ch2,
            ch3,
            ch4
        ])

    state["packet_count"] += 1


# ==========================================
# TERMINAL KEY READING
# ==========================================

def key_pressed():

    """
    Check whether a key is waiting in stdin.

    This is used for macOS terminal control.
    """

    readable, _, _ = select.select(
        [sys.stdin],
        [],
        [],
        0
    )

    return bool(readable)


def read_key():

    """
    Read one terminal character without waiting.
    """

    if key_pressed():

        return sys.stdin.read(1)

    return None


# ==========================================
# HOLD-ENTER RECORDING
# ==========================================

async def wait_for_enter_press():

    """
    Wait until ENTER is pressed.

    Returns when ENTER is detected.
    """

    print(
        "\n================================"
    )

    print(
        "HOLD ENTER TO RECORD"
    )

    print(
        "================================"
    )

    print(
        "\nPress and HOLD ENTER..."
    )

    # Save current terminal settings
    old_settings = termios.tcgetattr(
        sys.stdin
    )

    try:

        tty.setcbreak(
            sys.stdin.fileno()
        )

        while True:

            key = read_key()

            if key == "\n" or key == "\r":

                return

            await asyncio.sleep(
                0.01
            )

    finally:

        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            old_settings
        )


# ==========================================
# WAIT FOR ENTER RELEASE
# ==========================================

async def wait_for_enter_release():

    """
    Wait for ENTER to be released.

    NOTE:
    Terminal input does not expose a true
    physical key-release event like a GUI.

    We therefore wait until the terminal
    stops reporting the Enter key.

    """

    # Give the terminal time to process
    # the original Enter press.

    await asyncio.sleep(
        0.15
    )

    old_settings = termios.tcgetattr(
        sys.stdin
    )

    try:

        tty.setcbreak(
            sys.stdin.fileno()
        )

        # Drain remaining ENTER characters

        while key_pressed():

            key = sys.stdin.read(1)

            if key not in ("\n", "\r"):

                break

            await asyncio.sleep(
                0.01
            )

        # The terminal has now processed
        # the Enter press.

        # Wait for the next Enter event,
        # which acts as the release/stop
        # control in terminal mode.

        while True:

            key = read_key()

            if key == "\n" or key == "\r":

                return

            await asyncio.sleep(
                0.01
            )

    finally:

        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            old_settings
        )


# ==========================================
# SIMPLE HOLD-TO-RECORD MODE
# ==========================================

async def hold_to_record():

    """
    Terminal-safe recording control.

    Press ENTER to start.
    Press ENTER again to stop.

    Because terminals do not provide true
    key-release events, this behaves as:

        ENTER → START
        ENTER → STOP
    """

    print(
        "\n================================"
    )

    print(
        "PRESS ENTER TO START"
    )

    print(
        "PRESS ENTER AGAIN TO STOP"
    )

    print(
        "================================"
    )

    input(
        "\nPress ENTER to start..."
    )

    return


# ==========================================
# BLE ACQUISITION
# ==========================================

async def record_ble():

    print("\n================================")
    print("IRIS BLE ACQUISITION")
    print("================================")

    print("\nScanning for IRIS-EMG...")

    device = await BleakScanner.find_device_by_name(
        DEVICE_NAME,
        timeout=10
    )

    if device is None:

        print(
            "\nERROR: IRIS-EMG not found."
        )

        print(
            "Make sure the ESP32 is powered "
            "and advertising."
        )

        return False

    print("\nFound IRIS-EMG.")

    print(
        f"Address: {device.address}"
    )

    print("\nConnecting...")

    async with BleakClient(device) as client:

        print("Connected!")

        print(
            "\nPreparing BLE acquisition..."
        )

        await client.start_notify(
            CHARACTERISTIC_UUID,
            handle_data
        )

        print(
            "BLE notifications enabled."
        )

        print(
            f"\nRecording: {word}"
        )

        print(
            f"Condition: {condition}"
        )

        print(
            "Duration: HOLD ENTER"
        )

        print(
            f"Saving to: {filepath}"
        )

        # ----------------------------------
        # Reset state
        # ----------------------------------

        state["samples"].clear()

        state["packet_count"] = 0

        state["invalid_packets"] = 0

        state["first_timestamp"] = None

        state["last_packet_sequence"] = None

        state["recording_finished"] = False

        # ----------------------------------
        # START
        # ----------------------------------

        await hold_to_record()

        state["recording_enabled"] = True

        print(
            "\nWaiting for BLE data..."
        )

        # ----------------------------------
        # Wait for first sample
        # ----------------------------------

        while state["first_timestamp"] is None:

            await asyncio.sleep(
                0.01
            )

        # ----------------------------------
        # RECORD UNTIL USER STOPS
        # ----------------------------------

        print(
            "\nRecording..."
        )

        print(
            "Press ENTER when finished."
        )

        # Run the stop input in a thread
        # so BLE notifications continue.

        stop_task = asyncio.create_task(
            asyncio.to_thread(
                input,
                ""
            )
        )

        while not stop_task.done():

            await asyncio.sleep(
                0.01
            )

        # ----------------------------------
        # STOP
        # ----------------------------------

        state["recording_enabled"] = False

        state["recording_finished"] = True

        print(
            "\nENTER RELEASED / STOPPING..."
        )

        await client.stop_notify(
            CHARACTERISTIC_UUID
        )

    # ======================================
    # CHECK DATA
    # ======================================

    if len(state["samples"]) == 0:

        print(
            "\nERROR: No samples were recorded."
        )

        return False

    # ======================================
    # SAVE
    # ======================================

    print(
        "\nSaving recording..."
    )

    with open(
        filepath,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp_us",
            "CH1",
            "CH2",
            "CH3",
            "CH4"
        ])

        writer.writerows(
            state["samples"]
        )

    # ======================================
    # VERIFY
    # ======================================

    if not filepath.exists():

        print(
            "\nERROR: CSV file was not created."
        )

        return False

    print(
        "\nRecording saved successfully:"
    )

    print(
        filepath
    )

    return True


# ==========================================
# QUALITY CHECK
# ==========================================

def quality_check():

    print("\n================================")
    print("DATA QUALITY CHECK")
    print("================================")

    if not filepath.exists():

        print(
            "\nERROR: Recording file not found."
        )

        print(
            filepath
        )

        return False

    timestamps = []

    channel_data = {
        "CH1": [],
        "CH2": [],
        "CH3": [],
        "CH4": []
    }

    missing_values = False

    with open(
        filepath,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            try:

                timestamp = int(
                    row["timestamp_us"]
                )

                timestamps.append(
                    timestamp
                )

                for channel in channel_data:

                    value = row[channel]

                    if value == "":

                        missing_values = True

                    else:

                        channel_data[channel].append(
                            int(value)
                        )

            except (
                ValueError,
                KeyError,
                TypeError
            ):

                missing_values = True

    # ======================================
    # SAMPLE COUNT
    # ======================================

    sample_count = len(timestamps)

    print(
        f"Sample count     : {sample_count}"
    )

    # ======================================
    # SAMPLING RATE
    # ======================================

    recording_duration = 0

    sampling_rate = 0

    if len(timestamps) > 1:

        recording_duration = (
            timestamps[-1]
            - timestamps[0]
        ) / 1_000_000

        if recording_duration > 0:

            sampling_rate = (
                (len(timestamps) - 1)
                / recording_duration
            )

    sampling_rate_ok = (
        590
        <= sampling_rate
        <= 610
    )

    print(
        f"Sampling rate    : "
        f"{'PASS' if sampling_rate_ok else 'WARNING'} "
        f"({sampling_rate:.2f} Hz)"
    )

    # ======================================
    # MISSING VALUES
    # ======================================

    print(
        f"Missing values   : "
        f"{'WARNING' if missing_values else 'PASS'}"
    )

    # ======================================
    # CHANNEL CHECKS
    # ======================================

    channel_status = {}

    for channel, values in channel_data.items():

        if len(values) == 0:

            channel_status[channel] = False

            print(
                f"{channel:<17}: WARNING"
            )

            continue

        minimum = min(values)

        maximum = max(values)

        constant = (
            minimum == maximum
        )

        saturated = (
            minimum <= 5
            or maximum >= 4090
        )

        channel_status[channel] = not (
            constant
            or saturated
        )

        print(
            f"{channel:<17}: "
            f"{'PASS' if channel_status[channel] else 'WARNING'}"
        )

    # ======================================
    # BLE PACKETS
    # ======================================

    packet_status = (
        state["invalid_packets"] == 0
    )

    print(
        f"BLE packets      : "
        f"{'PASS' if packet_status else 'WARNING'} "
        f"(invalid: {state['invalid_packets']})"
    )

    # ======================================
    # OVERALL
    # ======================================

    all_channels_ok = all(
        channel_status.values()
    )

    overall_quality = (
        len(timestamps) > 0
        and sampling_rate_ok
        and not missing_values
        and all_channels_ok
        and packet_status
    )

    print("--------------------------------")

    print(
        f"Overall quality  : "
        f"{'PASS' if overall_quality else 'WARNING'}"
    )

    print("================================")

    # ======================================
    # METADATA
    # ======================================

    metadata_exists = (
        METADATA_FILE.exists()
    )

    with open(
        METADATA_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        if not metadata_exists:

            writer.writerow([
                "recording_id",
                "participant",
                "session",
                "condition",
                "word",
                "trial",
                "samples",
                "duration_seconds",
                "sampling_rate_hz",
                "quality_status",
                "file_path"
            ])

        recording_id = (
            f"{participant}_S{session}_"
            f"{condition}_{word}_T{trial}"
        )

        relative_filepath = filepath.relative_to(
            PROJECT_ROOT
        )

        writer.writerow([
            recording_id,
            participant,
            session,
            condition,
            word,
            trial,
            len(timestamps),
            f"{recording_duration:.3f}",
            f"{sampling_rate:.2f}",
            (
                "PASS"
                if overall_quality
                else "WARNING"
            ),
            str(relative_filepath)
        ])

    # ======================================
    # FINAL
    # ======================================

    print("\n================================")
    print("Recording complete!")
    print("================================")

    print(
        f"Samples recorded : {len(timestamps)}"
    )

    print(
        f"Duration         : "
        f"{recording_duration:.3f} s"
    )

    print(
        f"Sampling rate    : "
        f"{sampling_rate:.2f} Hz"
    )

    print(
        f"Quality          : "
        f"{'PASS' if overall_quality else 'WARNING'}"
    )

    print(
        f"Data file        : {filepath}"
    )

    print(
        f"Metadata         : {METADATA_FILE}"
    )

    return overall_quality


# ==========================================
# MAIN
# ==========================================

async def main():

    recording_success = await record_ble()

    if not recording_success:

        print(
            "\nRecording failed. "
            "Quality check was not run."
        )

        return

    quality_check()


# ==========================================
# RUN
# ==========================================

try:

    asyncio.run(main())

except KeyboardInterrupt:

    state["recording_enabled"] = False

    print(
        "\n\nRecording stopped by user."
    )
