import cv2
import time
import snap7

from bottlecap_inspection.inspect import inspect_cap
from bottlecap_inspection.io import load_image
from bottlecap_inspection.config import CFG


# =========================
# PLC CONFIG
# =========================
PLC_IP = "192.168.1.1"   # change
RACK = 0
SLOT = 1

DB_NUMBER = 1
DB_SIZE = 12
AREA_DB = 0x84  # DB area


# =========================
# PLC HELPERS
# =========================
def set_bit(data: bytearray, byte: int, bit: int, value: bool):
    mask = 1 << bit
    if value:
        data[byte] |= mask
    else:
        data[byte] &= ~mask
        

def write_plc_result(plc, passed: bool, pulse_width_s: float = 0.1):
    """
    PLC bits:
      DBX0.0 -> w_Result_Pulse (momentary trigger)
      DBX0.1 -> w_Result       (PASS / FAIL level)
    """

    data = bytearray(2)  # only need DBB0

    # --- Set result level ---
    set_bit(data, 0, 1, passed)   # w_Result

    # --- Raise pulse ---
    set_bit(data, 0, 0, True)     # w_Result_Pulse = 1
    plc.db_write(DB_NUMBER, 0, data)
    print(f"📤 Result written ({'PASS' if passed else 'FAIL'})")

    # --- Pulse width ---
    time.sleep(pulse_width_s)

    # --- Reset pulse ---
    set_bit(data, 0, 0, False)    # w_Result_Pulse = 0
    plc.db_write(DB_NUMBER, 0, data)
    print("📤 Pulse reset")


# =========================
# CAMERA SETUP
# =========================
def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened")
    return cap


# =========================
# MAIN LOOP
# =========================
def main():
    # --- PLC connect ---
    plc = snap7.client.Client()
    print("Connecting to PLC...")
    plc.connect(PLC_IP, RACK, SLOT)
    print("PLC connected")

    # --- Camera ---
    cap = open_camera(0)
    print("Press SPACE to inspect | ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Live", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key == 32:  # SPACE = trigger
            t0 = time.perf_counter()

            # --- Vision ---
            image = cv2.resize(frame, CFG["resize_width"])
            result = inspect_cap(image, early_exit=False)

            # status = result["status"]
            # plc_word = result["plc_word"]
            # passed = (result["status"] == "pass")
            plc_word = False


            # --- PLC write ---
            write_plc_result(plc, plc_word)

            dt = (time.perf_counter() - t0) * 1000
            print(f"[{status.upper()}] PLC_WORD={bin(plc_word)} | {dt:.1f} ms")

    cap.release()
    cv2.destroyAllWindows()
    plc.disconnect()
    print("Shutdown clean")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    main()
