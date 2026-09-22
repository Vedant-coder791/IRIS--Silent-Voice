
import asyncio
import csv
import os
import struct
import time
import threading

from bleak import BleakScanner, BleakClient
from pynput import keyboard


# ============================================================
# BLE SETTINGS
# ============================================================

DEVICE_NAME = "IRIS-EMG"

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

TARGET_SAMPLE_RATE = 600
MIN_SAMPLE_RATE = 590
MAX_SAMPLE_RATE = 610

CHANNELS = 4
SAMPLES_PER_PACKET = 10

# Packet:
# 4 bytes  = packet sequence
# 4 bytes  = timestamp of first sample
# 80 bytes = 10 samples × 4 channels × 2 bytes
PACKET_SIZE = 88


# ============================================================
# EXPERIMENT INFORMATION
# ============================================================

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


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW_ROOT = os.path.join(
    PROJECT_ROOT,
    "Data_original",
    "raw"
)


# ============================================================
# CREATE FOLDER
# ============================================================

folder = os.path.join(
    RAW_ROOT,
    participant,
    f"session_{session}",
    condition,
    word
)

os.makedirs(folder, exist_ok=True)


# ============================================================
# CREATE FILENAME
# ============================================================

filename = (
    f"{participant}_S{session}_"
    f"{condition}_{word}_T{trial}.csv"
)

filepath = os.path.join(
    folder,
    filename
)


# ============================================================
# PREVENT OVERWRITING
# ============================================================

if os.path.exists(filepath):

    print("\nERROR: This recording already exists!")
    print(f"File: {filepath}")
    print("Please use a different trial number.")

    raise SystemExit


# ============================================================
# METADATA FILE
# ============================================================

metadata_file = os.path.join(
    RAW_ROOT,
    "recordings_metadata.csv"
)

os.makedirs(RAW_ROOT, exist_ok=True)


# ============================================================
# SHARED RECORDING STATE
# ============================================================

recording_started = threading.Event()
recording_stopped = threading.Event()

recording_active = False


# ============================================================
# KEYBOARD CONTROL
# ============================================================

def on_press(key):

    global recording_active

    if key == keyboard.Key.enter:

        # Start only once
        if not recording_active:

            recording_active = True
            recording_started.set()

            print("\n")
            print("================================")
            print("RECORDING")
            print("================================")
            print("ENTER pressed")
            print("Recording now...")
            print("Release ENTER to stop.")
            print("================================")


def on_release(key):

    global recording_active

    if key == keyboard.Key.enter:

        if recording_active:

            recording_active = False
            recording_stopped.set()

            print("\n")
            print("ENTER released")
            print("Stopping recording...")

            return False


# ============================================================
# BLE DATA STORAGE
# ============================================================

samples = []

first_timestamp = None
last_timestamp = None

packet_count = 0


# ============================================================
# BLE NOTIFICATION CALLBACK
# ============================================================

def notification_handler(sender, data):

    global first_timestamp
    global last_timestamp
    global packet_count

    if len(data) != PACKET_SIZE:

        print(
            f"\nWARNING: Unexpected packet size: "
            f"{len(data)} bytes"
        )

        return

    # --------------------------------------------------------
    # Packet header
    # --------------------------------------------------------

    packet_sequence = struct.unpack_from(
        "<I",
        data,
        0
    )[0]

    packet_timestamp = struct.unpack_from(
        "<I",
        data,
        4
    )[0]

    # --------------------------------------------------------
    # Only collect while recording
    # --------------------------------------------------------

    if not recording_active:
        return

    packet_count += 1

    # --------------------------------------------------------
    # Extract 10 samples
    # --------------------------------------------------------

    offset = 8

    for i in range(SAMPLES_PER_PACKET):

        values = struct.unpack_from(
            "<4H",
            data,
            offset
        )

        offset += 8

        # ESP32 timestamps are based on the first
        # sample timestamp in the packet.
        #
        # Samples are approximately 600 Hz,
        # therefore interval ≈ 1666.67 us.

        timestamp = (
            packet_timestamp
            + round(i * (1_000_000 / TARGET_SAMPLE_RATE))
        )

        samples.append([
            timestamp,
            values[0],
            values[1],
            values[2],
            values[3]
        ])

        if first_timestamp is None:
            first_timestamp = timestamp

        last_timestamp = timestamp


# ============================================================
# MAIN
# ============================================================

async def main():

    global recording_active

    print("\n================================")
    print("IRIS EMG BLE COLLECTOR")
    print("================================")

    print(f"Device: {DEVICE_NAME}")

    print("\nScanning for ESP32...")

    device = await BleakScanner.find_device_by_name(
        DEVICE_NAME,
        timeout=10
    )

    if device is None:

        print("\nERROR: IRIS-EMG was not found.")

        print(
            "Make sure the ESP32 is powered on "
            "and advertising."
        )

        return

    print(f"\nFound: {device.name}")
    print(f"Address: {device.address}")

    # --------------------------------------------------------
    # Connect
    # --------------------------------------------------------

    print("\nConnecting...")

    async with BleakClient(device) as client:

        print("Connected!")

        # ----------------------------------------------------
        # Subscribe to notifications
        # ----------------------------------------------------

        await client.start_notify(
            CHARACTERISTIC_UUID,
            notification_handler
        )

        print("BLE notifications enabled.")

        print("\n================================")
        print(f"Recording: {word}")
        print(f"Condition: {condition}")
        print("Duration: UNLIMITED")
        print(f"Saving to:")
        print(filepath)
        print("================================")

        print(
            "\nHOLD ENTER to record."
        )

        print(
            "Release ENTER to stop."
        )

        # ----------------------------------------------------
        # Start keyboard listener
        # ----------------------------------------------------

        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )

        listener.start()

        # ----------------------------------------------------
        # Wait for Enter press
        # ----------------------------------------------------

        await asyncio.to_thread(
            recording_started.wait
        )

        # ----------------------------------------------------
        # Wait for Enter release
        # ----------------------------------------------------

        await asyncio.to_thread(
            recording_stopped.wait
        )

        # ----------------------------------------------------
        # Stop notifications
        # ----------------------------------------------------

        await client.stop_notify(
            CHARACTERISTIC_UUID
        )

        listener.stop()

    # ========================================================
    # SAVE DATA
    # ========================================================

    print("\nSaving recording...")

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

        for sample in samples:

            writer.writerow(sample)

    # ========================================================
    # TIMING CALCULATION
    # ========================================================

    sample_count = len(samples)

    recording_duration = 0
    sampling_rate = 0

    if sample_count > 1:

        elapsed_microseconds = (
            samples[-1][0]
            - samples[0][0]
        )

        recording_duration = (
            elapsed_microseconds
            / 1_000_000
        )

        sampling_rate = (
            (sample_count - 1)
            / recording_duration
        )

    # ========================================================
    # QUALITY CHECK
    # ========================================================

    print("\n================================")
    print("DATA QUALITY CHECK")
    print("================================")

    # --------------------------------------------------------
    # Sample count
    # --------------------------------------------------------

    # No fixed duration now, so sample count is reported
    # rather than judged against a 10-second target.

    sample_count_ok = sample_count > 0

    print(
        f"Sample count     : "
        f"{'PASS' if sample_count_ok else 'WARNING'} "
        f"({sample_count})"
    )

    # --------------------------------------------------------
    # Sampling rate
    # --------------------------------------------------------

    sampling_rate_ok = (
        MIN_SAMPLE_RATE
        <= sampling_rate
        <= MAX_SAMPLE_RATE
    )

    print(
        f"Sampling rate    : "
        f"{'PASS' if sampling_rate_ok else 'WARNING'} "
        f"({sampling_rate:.2f} Hz)"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = False

    with open(filepath, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            for channel in [
                "CH1",
                "CH2",
                "CH3",
                "CH4"
            ]:

                if row[channel] == "":

                    missing_values = True
                    break

            if missing_values:
                break

    print(
        f"Missing values   : "
        f"{'WARNING' if missing_values else 'PASS'}"
    )

    # --------------------------------------------------------
    # Channel checks
    # --------------------------------------------------------

    channel_status = {}

    with open(filepath, "r") as file:

        reader = csv.DictReader(file)

        channel_data = {
            "CH1": [],
            "CH2": [],
            "CH3": [],
            "CH4": []
        }

        for row in reader:

            for channel in channel_data:

                channel_data[channel].append(
                    int(row[channel])
                )

    for channel, values in channel_data.items():

        if not values:

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

        channel_status[channel] = (
            not constant
            and not saturated
        )

        print(
            f"{channel:<17}: "
            f"{'PASS' if channel_status[channel] else 'WARNING'}"
        )

    # ========================================================
    # OVERALL QUALITY
    # ========================================================

    all_channels_ok = (
        len(channel_status) == 4
        and all(channel_status.values())
    )

    overall_quality = (
        sample_count_ok
        and sampling_rate_ok
        and not missing_values
        and all_channels_ok
    )

    print("--------------------------------")

    if overall_quality:

        print("Overall quality  : PASS")

    else:

        print("Overall quality  : WARNING")

    print("================================")

    # ========================================================
    # UPDATE METADATA
    # ========================================================

    file_exists = os.path.exists(
        metadata_file
    )

    with open(
        metadata_file,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

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

        writer.writerow([
            recording_id,
            participant,
            session,
            condition,
            word,
            trial,
            sample_count,
            f"{recording_duration:.3f}",
            f"{sampling_rate:.2f}",
            (
                "PASS"
                if overall_quality
                else "WARNING"
            ),
            filepath
        ])

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n================================")
    print("Recording complete!")
    print("================================")

    print(
        f"Samples recorded : "
        f"{sample_count}"
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
        f"Data file        : "
        f"{filepath}"
    )

    print(
        f"Metadata         : "
        f"{metadata_file}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())

