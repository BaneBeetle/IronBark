"""
pi/test_lidar.py — Quick test: is the RPLidar C1 working?
Run on Pi: python3 test_lidar.py

Checks:
  1. Serial port exists
  2. Can connect to LiDAR
  3. Reads 3 full scans
  4. Prints distance in each direction (forward/left/right/behind)
"""

import sys
import os
import asyncio

# ── Check serial port ─────────────────────────────────────────────
PORT = "/dev/ttyUSB0"
BAUD = 460800

print("=" * 50)
print("  RPLidar C1 — Quick Test")
print("=" * 50)

if not os.path.exists(PORT):
    for alt in ["/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyACM0"]:
        if os.path.exists(alt):
            PORT = alt
            break
    else:
        print(f"\n[FAIL] No serial port found.")
        print("  → Is the USB adapter plugged in?")
        print("  → Run: lsusb | grep -i silicon")
        sys.exit(1)

print(f"\n[OK] Serial port: {PORT}")

# ── Import library ────────────────────────────────────────────────
try:
    from rplidarc1 import RPLidar
    print("[OK] Library: rplidarc1")
except ImportError:
    print("[FAIL] rplidarc1 library not found")
    print("  → Run: pip3 install --break-system-packages rplidarc1")
    sys.exit(1)


# ── Scan test ─────────────────────────────────────────────────────
async def main():
    print(f"\nConnecting to {PORT} @ {BAUD} baud...")
    try:
        lidar = RPLidar(PORT, BAUD)
        print("[OK] Connected")
    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        sys.exit(1)

    print(f"\nReading 3 scans...")
    scan_task = asyncio.create_task(lidar.simple_scan())

    points = []
    last_angle = -1
    scan_count = 0
    queue = lidar.output_queue

    for _ in range(3000):
        try:
            data = await asyncio.wait_for(queue.get(), timeout=3.0)
        except asyncio.TimeoutError:
            print("[FAIL] Timeout — no scan data received")
            print("  → Motor might not be spinning. Check power.")
            break

        angle = data["a_deg"]
        distance = data["d_mm"]
        quality = data["q"]

        if angle < last_angle - 180:
            scan_count += 1
            if points:
                def arc_min(pts, start, end):
                    if start > end:
                        arc = [d for a, d in pts if a >= start or a <= end]
                    else:
                        arc = [d for a, d in pts if start <= a <= end]
                    return min(arc) / 10.0 if arc else -1

                fwd = arc_min(points, 330, 30)
                right = arc_min(points, 30, 150)
                behind = arc_min(points, 150, 210)
                left = arc_min(points, 210, 330)
                print(f"  Scan {scan_count}: {len(points)} pts | "
                      f"F={fwd:.0f}cm  R={right:.0f}cm  B={behind:.0f}cm  L={left:.0f}cm")

            if scan_count >= 3:
                break
            points = []

        if quality > 0 and distance > 0:
            points.append((angle, distance))
        last_angle = angle

    lidar.stop_event.set()
    try:
        lidar.shutdown()
    except Exception:
        pass

    print(f"\n{'=' * 50}")
    if scan_count >= 3:
        print("  PASS — LiDAR is working!")
    else:
        print(f"  FAIL — Only got {scan_count} scans")
    print(f"{'=' * 50}")

asyncio.run(main())
