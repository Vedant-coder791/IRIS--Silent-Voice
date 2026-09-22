import asyncio
import struct
import time

from bleak import BleakScanner, BleakClient


# ==========================================
# BLE SETTINGS
# ==========================================

DEVICE_NAME = "IRIS-EMG"

CHARACTERISTIC_UUID = (
    "beb5483e-36e1-4688-b7f5-ea07361b26a8"
)

SAMPLES_PER_PACKET = 10
BYTES_PER_SAMPLE = 8
EXPECTED_PACKET_SIZE = (
    SAMPLES_PER_PACKET * BYTES_PER_SAMPLE
)


# ==========================================
# GLOBAL COUNTERS
# ==========================================

total_samples = 0
total_packets = 0

start_time = None


# ==========================================
# BLE DATA CALLBACK
# ==========================================

def handle_data(sender, data):

    global total_samples
    global total_packets
    global start_time

    # Start timing when first packet arrives
    if start_time is None:
        start_time = time.time()

    # Check packet size
    if len(data) != EXPECTED_PACKET_SIZE:
        print(
            f"WARNING: Unexpected packet size: "
            f"{len(data)} bytes"
        )
        return

    # Decode 10 samples
    for i in range(SAMPLES_PER_PACKET):

        offset = i * BYTES_PER_SAMPLE

        ch1, ch2, ch3, ch4 = struct.unpack_from(
            "<HHHH",
            data,
            offset
        )

        total_samples += 1

    total_packets += 1


# ==========================================
# MAIN BLE FUNCTION
# ==========================================

async def main():

    global start_time

    print()
    print("================================")
    print("IRIS BLE RECEIVER")
    print("================================")

    print("\nScanning for IRIS-EMG...")

    device = await BleakScanner.find_device_by_name(
        DEVICE_NAME,
        timeout=10
    )

    if device is None:

        print("\nERROR: IRIS-EMG not found.")
        print("Make sure the ESP32 is powered and advertising.")

        return

    print("\nFound device!")
    print(f"Address: {device.address}")

    print("\nConnecting...")

    async with BleakClient(device) as client:

        print("Connected!")

        print(
            f"Subscribed to:\n"
            f"{CHARACTERISTIC_UUID}"
        )

        await client.start_notify(
            CHARACTERISTIC_UUID,
            handle_data
        )

        print("\nReceiving BLE data...")
        print("Press Ctrl+C to stop.\n")

        last_print = time.time()

        while True:

            await asyncio.sleep(0.1)

            current_time = time.time()

            if current_time - last_print >= 2:

                last_print = current_time

                if start_time is not None:

                    elapsed = (
                        current_time - start_time
                    )

                    rate = (
                        total_samples / elapsed
                    )

                    print(
                        f"Packets received: "
                        f"{total_packets:6d} | "
                        f"Samples received: "
                        f"{total_samples:7d} | "
                        f"Rate: {rate:7.2f} Hz"
                    )


# ==========================================
# START PROGRAM
# ==========================================

try:

    asyncio.run(main())

except KeyboardInterrupt:

    print("\n\nStopping BLE receiver...")

    if start_time is not None:

        elapsed = time.time() - start_time

        if elapsed > 0:

            rate = total_samples / elapsed

            print(
                f"\nTotal packets : {total_packets}"
            )

            print(
                f"Total samples : {total_samples}"
            )

            print(
                f"Average rate  : {rate:.2f} Hz"
            )

    print("\nDone.")