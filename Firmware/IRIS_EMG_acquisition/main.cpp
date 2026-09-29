
#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ============================================================
// IRIS SILENT SPEECH RECOGNITION
// FINAL ESP32-S3 BLE EMG ACQUISITION FIRMWARE
// ============================================================
//
// DATA PATH:
//
// EMG AFE
//    ↓
// ESP32-S3 ADC
//    ↓
// 4 channels @ 600 Hz
//    ↓
// 10 samples / BLE packet
//    ↓
// 88-byte BLE notification
//    ↓
// Python collector
//
// PACKET FORMAT:
//
// Bytes 0-3    : uint32 packet sequence
// Bytes 4-7    : uint32 timestamp of first sample (µs)
// Bytes 8-87   : 10 × [CH1, CH2, CH3, CH4]
//                 each channel = uint16
//
// Total packet size = 88 bytes
// ============================================================


// ============================================================
// BLE SETTINGS
// ============================================================

#define DEVICE_NAME "IRIS-EMG"

#define SERVICE_UUID \
    "4fafc201-1fb5-459e-8fcc-c5c9c331914b"

#define CHARACTERISTIC_UUID \
    "beb5483e-36e1-4688-b7f5-ea07361b26a8"


// ============================================================
// ADC CHANNELS
// ============================================================
//
// ESP32-S3 ADC1:
//
// GPIO4 = ADC1_CH3
// GPIO5 = ADC1_CH4
// GPIO6 = ADC1_CH5
// GPIO7 = ADC1_CH6
//
// These are the four acquisition channels.
//
// IMPORTANT:
// The analog voltage presented to these pins must remain
// within the ESP32-S3 ADC input range.
//

#define CH1_PIN 4
#define CH2_PIN 5
#define CH3_PIN 6
#define CH4_PIN 7


// ============================================================
// SAMPLING
// ============================================================

constexpr uint32_t SAMPLE_RATE_HZ = 600;

constexpr uint8_t SAMPLES_PER_PACKET = 10;

constexpr uint32_t PACKET_SIZE = 88;


// ============================================================
// TIMING
// ============================================================
//
// 1,000,000 / 600 = 1666.666... µs
//
// Therefore we alternate between 1666 and 1667 µs.
// This keeps the long-term average very close to 600 Hz.
//

constexpr uint32_t BASE_INTERVAL_US =
    1000000UL / SAMPLE_RATE_HZ;

constexpr uint32_t TIMING_REMAINDER_US =
    1000000UL % SAMPLE_RATE_HZ;


// ============================================================
// PACKET BUFFER
// ============================================================

uint8_t packet[PACKET_SIZE];


// ============================================================
// COUNTERS
// ============================================================

uint32_t packetSequence = 0;

uint32_t samplesGenerated = 0;

uint32_t packetsSent = 0;


// ============================================================
// TIMING STATE
// ============================================================

uint32_t nextSampleTime = 0;

uint32_t timingAccumulator = 0;


// ============================================================
// BLE STATE
// ============================================================

BLECharacteristic *pCharacteristic = nullptr;

volatile bool deviceConnected = false;


// ============================================================
// BLE SERVER CALLBACKS
// ============================================================

class ServerCallbacks : public BLEServerCallbacks {

    void onConnect(BLEServer *server) override {

        deviceConnected = true;

        Serial.println();
        Serial.println("================================");
        Serial.println("BLE CLIENT CONNECTED");
        Serial.println("================================");
    }


    void onDisconnect(BLEServer *server) override {

        deviceConnected = false;

        Serial.println();
        Serial.println("================================");
        Serial.println("BLE CLIENT DISCONNECTED");
        Serial.println("================================");

        delay(200);

        Serial.println(
            "Restarting BLE advertising..."
        );

        BLEDevice::startAdvertising();

        Serial.println(
            "BLE advertising restarted."
        );
    }
};


// ============================================================
// SETUP
// ============================================================

void setup() {

    Serial.begin(115200);

    delay(1000);


    // ========================================================
    // STARTUP MESSAGE
    // ========================================================

    Serial.println();
    Serial.println("================================");
    Serial.println("IRIS EMG BLE");
    Serial.println("FINAL ACQUISITION FIRMWARE");
    Serial.println("================================");


    // ========================================================
    // ADC CONFIGURATION
    // ========================================================

    pinMode(CH1_PIN, INPUT);
    pinMode(CH2_PIN, INPUT);
    pinMode(CH3_PIN, INPUT);
    pinMode(CH4_PIN, INPUT);

    // 12-bit ADC:
    // range = 0 to 4095

    analogReadResolution(12);


    Serial.println();
    Serial.println("ADC configured.");

    Serial.print("CH1 GPIO: ");
    Serial.println(CH1_PIN);

    Serial.print("CH2 GPIO: ");
    Serial.println(CH2_PIN);

    Serial.print("CH3 GPIO: ");
    Serial.println(CH3_PIN);

    Serial.print("CH4 GPIO: ");
    Serial.println(CH4_PIN);


    // ========================================================
    // BLE INITIALIZATION
    // ========================================================

    Serial.println();
    Serial.println("Initializing BLE...");

    BLEDevice::init(
        DEVICE_NAME
    );


    // ========================================================
    // BLE SERVER
    // ========================================================

    BLEServer *server =
        BLEDevice::createServer();

    server->setCallbacks(
        new ServerCallbacks()
    );


    // ========================================================
    // BLE SERVICE
    // ========================================================

    BLEService *service =
        server->createService(
            SERVICE_UUID
        );


    // ========================================================
    // BLE CHARACTERISTIC
    // ========================================================

    pCharacteristic =
        service->createCharacteristic(
            CHARACTERISTIC_UUID,
            BLECharacteristic::PROPERTY_NOTIFY
        );


    // ========================================================
    // BLE NOTIFICATION DESCRIPTOR
    // ========================================================

    pCharacteristic->addDescriptor(
        new BLE2902()
    );


    // ========================================================
    // START SERVICE
    // ========================================================

    service->start();


    // ========================================================
    // CONFIGURE ADVERTISING
    // ========================================================

    BLEAdvertising *advertising =
        BLEDevice::getAdvertising();


    advertising->addServiceUUID(
        SERVICE_UUID
    );


    advertising->setScanResponse(
        true
    );


    advertising->setMinPreferred(
        0x06
    );


    advertising->setMinPreferred(
        0x12
    );


    // ========================================================
    // START ADVERTISING
    // ========================================================

    Serial.println();
    Serial.println(
        "Starting BLE advertising..."
    );

    BLEDevice::startAdvertising();


    // ========================================================
    // SYSTEM INFORMATION
    // ========================================================

    Serial.println();
    Serial.println("================================");
    Serial.println("BLE READY");
    Serial.println("================================");

    Serial.print(
        "Device name: "
    );

    Serial.println(
        DEVICE_NAME
    );

    Serial.print(
        "Sample rate: "
    );

    Serial.print(
        SAMPLE_RATE_HZ
    );

    Serial.println(
        " Hz"
    );

    Serial.print(
        "Samples/packet: "
    );

    Serial.println(
        SAMPLES_PER_PACKET
    );

    Serial.print(
        "Packet size: "
    );

    Serial.println(
        PACKET_SIZE
    );

    Serial.println();
    Serial.println(
        "Advertising as IRIS-EMG"
    );

    Serial.println(
        "Waiting for BLE connection..."
    );

    Serial.println();


    // ========================================================
    // INITIALIZE SAMPLING CLOCK
    // ========================================================

    nextSampleTime = micros();
}


// ============================================================
// LOOP
// ============================================================

void loop() {

    // ========================================================
    // WAIT FOR NEXT SAMPLE TIME
    // ========================================================

    while (
        (int32_t)(
            micros() - nextSampleTime
        ) < 0
    ) {

        // Precise timing wait.
    }


    // ========================================================
    // SCHEDULE NEXT SAMPLE
    // ========================================================

    nextSampleTime += BASE_INTERVAL_US;

    timingAccumulator += TIMING_REMAINDER_US;


    if (
        timingAccumulator >= SAMPLE_RATE_HZ
    ) {

        nextSampleTime += 1;

        timingAccumulator -= SAMPLE_RATE_HZ;
    }


    // ========================================================
    // READ FOUR ADC CHANNELS
    // ========================================================

    uint16_t ch1 =
        analogRead(CH1_PIN);

    uint16_t ch2 =
        analogRead(CH2_PIN);

    uint16_t ch3 =
        analogRead(CH3_PIN);

    uint16_t ch4 =
        analogRead(CH4_PIN);


    // ========================================================
    // DETERMINE POSITION INSIDE PACKET
    // ========================================================

    uint8_t sampleIndex =
        samplesGenerated %
        SAMPLES_PER_PACKET;


    // ========================================================
    // START NEW PACKET
    // ========================================================

    if (sampleIndex == 0) {

        // --------------------------------------------
        // Packet sequence number
        // --------------------------------------------

        uint32_t sequence =
            packetSequence;

        memcpy(
            packet,
            &sequence,
            sizeof(sequence)
        );


        // --------------------------------------------
        // Timestamp of first sample
        // --------------------------------------------

        uint32_t timestamp =
            micros();

        memcpy(
            packet + 4,
            &timestamp,
            sizeof(timestamp)
        );
    }


    // ========================================================
    // STORE SAMPLE IN PACKET
    // ========================================================

    uint16_t offset =
        8 +
        (sampleIndex * 8);


    // CH1

    memcpy(
        packet + offset,
        &ch1,
        sizeof(ch1)
    );


    // CH2

    memcpy(
        packet + offset + 2,
        &ch2,
        sizeof(ch2)
    );


    // CH3

    memcpy(
        packet + offset + 4,
        &ch3,
        sizeof(ch3)
    );


    // CH4

    memcpy(
        packet + offset + 6,
        &ch4,
        sizeof(ch4)
    );


    samplesGenerated++;


    // ========================================================
    // SEND COMPLETE PACKET
    // ========================================================

    if (
        sampleIndex ==
        SAMPLES_PER_PACKET - 1
    ) {

        if (deviceConnected) {

            pCharacteristic->setValue(
                packet,
                PACKET_SIZE
            );

            pCharacteristic->notify();

            packetsSent++;
        }


        // --------------------------------------------
        // Advance sequence number
        // --------------------------------------------

        packetSequence++;


        // --------------------------------------------
        // Status every 2 seconds
        // --------------------------------------------

        if (
            samplesGenerated % 1200 == 0
        ) {

            Serial.print(
                "Samples: "
            );

            Serial.print(
                samplesGenerated
            );

            Serial.print(
                " | Packets sent: "
            );

            Serial.print(
                packetsSent
            );

            Serial.print(
                " | BLE: "
            );


            if (deviceConnected) {

                Serial.println(
                    "CONNECTED"
                );

            } else {

                Serial.println(
                    "WAITING"
                );
            }
        }
    }
}
