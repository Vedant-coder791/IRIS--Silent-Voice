

import asyncio
import csv
import struct
from pathlib import Path
from threading import Event, Lock

from bleak import BleakClient, BleakScanner
from pynput import keyboard


# ============================================================
# IRIS SILENT SPEECH RECOGNITION
# FINAL BLE EMG DATA COLLECTOR
# ============================================================

DEVICE_NAME = "IRIS-EMG"

SERVICE_UUID = (
    "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
)

CHARACTERISTIC_UUID = (
    "beb5483e-36e1-4688-b7f5-ea07361b26a8"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

RAW_ROOT = (
    PROJECT_ROOT
    / "Data_original"
    / "raw"
)

METADATA_FILE = (
    RAW_ROOT
    / "recordings_metadata.csv"
)


# ============================================================
# ACQUISITION SETTINGS
# ============================================================

SAMPLE_RATE = 600

SAMPLES_PER_PACKET = 10

PACKET_SIZE = 88

BYTES_PER_SAMPLE = 8


# ============================================================
# STATE
# ============================================================

recording_active = False

recording_started = Event()

recording_stopped = Event()

state_lock = Lock()


# ============================================================
# DATA
# ============================================================

samples = []

received_packets = 0

lost_packets = 0

first_packet_sequence = None

last_packet_sequence = None

packet_sequence_errors = []

unexpected_packet_sizes = 0


# ============================================================
# RECORDING INFORMATION
# ============================================================

participant = ""

session = ""

condition = ""

word = ""

trial = ""


# ============================================================
# BLE NOTIFICATION HANDLER
# ============================================================

def notification_handler(
    sender,
    data
):

    global samples
    global received_packets
    global lost_packets
    global first_packet_sequence
    global last_packet_sequence
    global unexpected_packet_sizes


    # ========================================================
    # ONLY RECORD WHEN ENTER IS HELD
    # ========================================================

    with state_lock:

        if not recording_active:

            return


    # ========================================================
    # VERIFY PACKET SIZE
    # ========================================================

    if len(data) != PACKET_SIZE:

        unexpected_packet_sizes += 1

        print(
            f"WARNING: Unexpected packet size: "
            f"{len(data)} bytes"
        )

        return


    # ========================================================
    # READ PACKET HEADER
    # ========================================================

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


    # ========================================================
    # CHECK PACKET SEQUENCE
    # ========================================================

    if first_packet_sequence is None:

        first_packet_sequence = (
            packet_sequence
        )

    else:

        expected_sequence = (
            (last_packet_sequence + 1)
            & 0xFFFFFFFF
        )


        if packet_sequence != expected_sequence:

            difference = (
                packet_sequence
                - expected_sequence
            ) & 0xFFFFFFFF


            if difference > 0:

                lost_packets += difference

                packet_sequence_errors.append(
                    (
                        expected_sequence,
                        packet_sequence,
                        difference
                    )
                )


                print(
                    "WARNING: BLE packet gap detected: "
                    f"expected {expected_sequence}, "
                    f"received {packet_sequence}, "
                    f"missing {difference}"
                )


    last_packet_sequence = packet_sequence

    received_packets += 1


    # ========================================================
    # EXTRACT SAMPLES
    # ========================================================

    for i in range(
        SAMPLES_PER_PACKET
    ):

        offset = (
            8
            + i * BYTES_PER_SAMPLE
        )


        ch1 = struct.unpack_from(
            "<H",
            data,
            offset
        )[0]


        ch2 = struct.unpack_from(
            "<H",
            data,
            offset + 2
        )[0]


        ch3 = struct.unpack_from(
            "<H",
            data,
            offset + 4
        )[0]


        ch4 = struct.unpack_from(
            "<H",
            data,
            offset + 6
        )[0]


        # ----------------------------------------------------
        # Reconstruct sample timestamp
        # ----------------------------------------------------
        #
        # The ESP32 timestamp is the timestamp of sample 0.
        #
        # The acquisition rate is 600 Hz.
        #

        timestamp = (
            packet_timestamp
            + round(
                i * 1_000_000 / SAMPLE_RATE
            )
        )


        samples.append(
            (
                timestamp,
                packet_sequence,
                ch1,
                ch2,
                ch3,
                ch4
            )
        )


# ============================================================
# FIND BLE DEVICE
# ============================================================

async def find_device():

    print()

    print(
        "Scanning for IRIS-EMG..."
    )


    devices = await BleakScanner.discover(
        timeout=5
    )


    for device in devices:

        if device.name == DEVICE_NAME:

            print(
                f"Found {DEVICE_NAME}"
            )

            return device


    print()

    print(
        "ERROR: IRIS-EMG was not found."
    )

    print(
        "Check that the ESP32 is powered on "
        "and advertising."
    )

    return None


# ============================================================
# KEYBOARD PRESS
# ============================================================

def on_press(key):

    global recording_active


    try:

        if key == keyboard.Key.enter:

            with state_lock:

                if not recording_active:

                    recording_active = True

                    samples.clear()


                    print()

                    print(
                        "================================"
                    )

                    print(
                        "RECORDING STARTED"
                    )

                    print(
                        "================================"
                    )


                    recording_started.set()


    except Exception as error:

        print(
            f"Keyboard error: {error}"
        )


# ============================================================
# KEYBOARD RELEASE
# ============================================================

def on_release(key):

    global recording_active


    try:

        if key == keyboard.Key.enter:

            with state_lock:

                if recording_active:

                    recording_active = False


                    print()

                    print(
                        "================================"
                    )

                    print(
                        "RECORDING STOPPED"
                    )

                    print(
                        "================================"
                    )


                    recording_stopped.set()


                    return False


    except Exception as error:

        print(
            f"Keyboard error: {error}"
        )


# ============================================================
# CREATE OUTPUT PATH
# ============================================================

def create_output_path():

    folder = (
        RAW_ROOT
        / participant
        / f"session_{int(session):02d}"
        / condition
        / word
    )


    folder.mkdir(
        parents=True,
        exist_ok=True
    )


    filename = (
        f"{participant}"
        f"_S{int(session):02d}"
        f"_{condition}"
        f"_{word}"
        f"_T{int(trial):03d}.csv"
    )


    return folder / filename


# ============================================================
# QUALITY CHECK
# ============================================================

def quality_check(
    output_file,
    sample_count,
    duration
):

    print()

    print(
        "================================="
    )

    print(
        "DATA QUALITY CHECK"
    )

    print(
        "================================="
    )


    quality_status = "PASS"


    # ========================================================
    # SAMPLE COUNT
    # ========================================================

    if sample_count > 0:

        print(
            f"Sample count     : PASS ({sample_count})"
        )

    else:

        print(
            "Sample count     : FAIL (0)"
        )

        quality_status = "WARNING"


    # ========================================================
    # SAMPLING RATE
    # ========================================================

    if duration > 0:

        sampling_rate = (
            (sample_count - 1)
            / duration
        )

    else:

        sampling_rate = 0


    if (
        590 <= sampling_rate <= 610
    ):

        print(
            f"Sampling rate    : PASS "
            f"({sampling_rate:.2f} Hz)"
        )

    else:

        print(
            f"Sampling rate    : WARNING "
            f"({sampling_rate:.2f} Hz)"
        )

        quality_status = "WARNING"


    # ========================================================
    # READ CSV
    # ========================================================

    missing_values = False


    channel_values = {

        "CH1": [],
        "CH2": [],
        "CH3": [],
        "CH4": []
    }


    with open(
        output_file,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file
        )


        for row in reader:

            for channel in channel_values:

                value = row[channel]


                if value == "":

                    missing_values = True

                else:

                    channel_values[channel].append(
                        int(value)
                    )


    # ========================================================
    # MISSING VALUES
    # ========================================================

    if not missing_values:

        print(
            "Missing values   : PASS"
        )

    else:

        print(
            "Missing values   : FAIL"
        )

        quality_status = "WARNING"


    # ========================================================
    # CHANNEL CHECKS
    # ========================================================

    for channel, values in channel_values.items():

        if len(values) == 0:

            print(
                f"{channel:<17}: FAIL"
            )

            quality_status = "WARNING"

            continue


        minimum = min(values)

        maximum = max(values)


        # ----------------------------------------------------
        # Constant signal
        # ----------------------------------------------------

        if minimum == maximum:

            print(
                f"{channel:<17}: WARNING "
                f"(constant: {minimum})"
            )

            quality_status = "WARNING"


        # ----------------------------------------------------
        # Full-range saturation
        # ----------------------------------------------------

        elif (
            minimum <= 0
            and maximum >= 4095
        ):

            print(
                f"{channel:<17}: WARNING "
                f"(possible saturation)"
            )

            quality_status = "WARNING"


        else:

            print(
                f"{channel:<17}: PASS"
            )


    # ========================================================
    # PACKET INTEGRITY
    # ========================================================

    print()

    print(
        f"BLE packets      : {received_packets}"
    )

    print(
        f"Lost packets     : {lost_packets}"
    )

    print(
        f"Bad packet sizes : {unexpected_packet_sizes}"
    )


    if lost_packets == 0:

        print(
            "Packet integrity : PASS"
        )

    else:

        print(
            "Packet integrity : FAIL"
        )

        quality_status = "WARNING"


    if unexpected_packet_sizes == 0:

        print(
            "Packet size      : PASS"
        )

    else:

        print(
            "Packet size      : FAIL"
        )

        quality_status = "WARNING"


    # ========================================================
    # FINAL STATUS
    # ========================================================

    print(
        "---------------------------------"
    )


    print(
        f"Overall quality  : {quality_status}"
    )


    print(
        "================================="
    )


    return (
        quality_status,
        sampling_rate
    )


# ============================================================
# SAVE RECORDING
# ============================================================

def save_recording():

    if len(samples) == 0:

        print()

        print(
            "ERROR: No samples were recorded."
        )

        return


    # ========================================================
    # OUTPUT PATH
    # ========================================================

    output_file = (
        create_output_path()
    )


    # ========================================================
    # NEVER OVERWRITE
    # ========================================================

    if output_file.exists():

        print()

        print(
            "ERROR: Recording already exists:"
        )

        print(
            output_file
        )

        print()

        print(
            "Use a different trial number."
        )

        return


    # ========================================================
    # TIMESTAMP RANGE
    # ========================================================

    first_timestamp = (
        samples[0][0]
    )

    last_timestamp = (
        samples[-1][0]
    )


    duration = (
        last_timestamp
        - first_timestamp
    ) / 1_000_000


    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        output_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow([
            "timestamp_us",
            "packet_sequence",
            "CH1",
            "CH2",
            "CH3",
            "CH4"
        ])


        writer.writerows(
            samples
        )


    # ========================================================
    # QUALITY CHECK
    # ========================================================

    (
        quality_status,
        sampling_rate
    ) = quality_check(
        output_file,
        len(samples),
        duration
    )


    # ========================================================
    # CREATE METADATA DIRECTORY
    # ========================================================

    RAW_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # METADATA FILE
    # ========================================================

    metadata_exists = (
        METADATA_FILE.exists()
    )


    with open(
        METADATA_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )


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
                "received_packets",
                "lost_packets",
                "quality_status",
                "file_path"
            ])


        recording_id = (
            f"{participant}"
            f"_S{int(session):02d}"
            f"_{condition}"
            f"_{word}"
            f"_T{int(trial):03d}"
        )


        relative_path = (
            output_file.relative_to(
                PROJECT_ROOT
            )
        )


        writer.writerow([
            recording_id,
            participant,
            session,
            condition,
            word,
            trial,
            len(samples),
            f"{duration:.6f}",
            f"{sampling_rate:.2f}",
            received_packets,
            lost_packets,
            quality_status,
            str(relative_path)
        ])


    # ========================================================
    # FINAL RECORDING SUMMARY
    # ========================================================

    print()

    print(
        "================================="
    )

    print(
        "RECORDING COMPLETE"
    )

    print(
        "================================="
    )


    print(
        f"Samples recorded : {len(samples)}"
    )

    print(
        f"Duration         : {duration:.3f} s"
    )

    print(
        f"Sampling rate    : {sampling_rate:.2f} Hz"
    )

    print(
        f"BLE packets      : {received_packets}"
    )

    print(
        f"Lost packets     : {lost_packets}"
    )

    print(
        f"Quality           : {quality_status}"
    )

    print()

    print(
        "Saved to:"
    )

    print(
        output_file
    )

    print()


# ============================================================
# MAIN
# ============================================================

async def run():

    global participant
    global session
    global condition
    global word
    global trial

    # ========================================================
    # RESET STATE
    # ========================================================

    samples.clear()

    recording_started.clear()

    recording_stopped.clear()


    # ========================================================
    # RECORDING INFORMATION
    # ========================================================

    print()

    print(
        "================================="
    )

    print(
        "IRIS BLE EMG DATA COLLECTION"
    )

    print(
        "================================="
    )

    print()


    participant = input(
        "Participant ID: "
    ).strip().upper()


    session = input(
        "Session number: "
    ).strip()


    condition = input(
        "Condition "
        "(SILENT/VOICED/WHISPERED/REST): "
    ).strip().upper()


    word = input(
        "Word: "
    ).strip().upper()


    trial = input(
        "Trial number: "
    ).strip()


    # ========================================================
    # FIND DEVICE
    # ========================================================

    device = await find_device()


    if device is None:

        return


    # ========================================================
    # CONNECT
    # ========================================================

    print()

    print(
        "Connecting to IRIS-EMG..."
    )


    async with BleakClient(
        device
    ) as client:

        print(
            "Connected!"
        )


        # ====================================================
        # START BLE NOTIFICATIONS
        # ====================================================

        await client.start_notify(
            CHARACTERISTIC_UUID,
            notification_handler
        )


        # ====================================================
        # READY
        # ====================================================

        print()

        print(
            "================================="
        )

        print(
            "READY TO RECORD"
        )

        print(
            "================================="
        )

        print()

        print(
            "HOLD ENTER to record."
        )

        print(
            "RELEASE ENTER to stop."
        )

        print()


        # ====================================================
        # KEYBOARD LISTENER
        # ====================================================

        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )


        listener.start()


        # ====================================================
        # WAIT FOR RECORDING START
        # ====================================================

        await asyncio.to_thread(
            recording_started.wait
        )


        # ====================================================
        # WAIT FOR RECORDING STOP
        # ====================================================

        await asyncio.to_thread(
            recording_stopped.wait
        )


        # ====================================================
        # STOP BLE NOTIFICATIONS
        # ====================================================

        await client.stop_notify(
            CHARACTERISTIC_UUID
        )


        listener.stop()


        # ====================================================
        # SAVE RECORDING
        # ====================================================

        save_recording()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            run()
        )

    except KeyboardInterrupt:

        print()

        print(
            "Recording cancelled."
        )
