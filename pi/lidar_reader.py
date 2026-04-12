"""
pi/lidar_reader.py — IronBark Phase 7: RPLidar C1 Scanner
Reads 360-degree LiDAR scans and publishes them to the Mac over ZMQ.

The C1 produces ~300-500 points per scan at 10Hz. Each point has an angle
(0-360 degrees) and distance (millimeters). We pack each scan as JSON
and PUSH to the Mac, which uses it for obstacle avoidance.

Hardware: Slamtec RPLidar C1 via USB-to-UART adapter → /dev/ttyUSB0
Install:  pip3 install --break-system-packages rplidarc1

Usage:
    python3 -u lidar_reader.py [--serial /dev/ttyUSB0] [--baud 460800]
"""

import argparse
import asyncio
import json
import sys
import time
import signal

import zmq

try:
    from rplidarc1 import RPLidar
except ImportError:
    print("ERROR: rplidarc1 library not found.")
    print("  Install: pip3 install --break-system-packages rplidarc1")
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────────
try:
    import config as _cfg
    DEFAULT_PC_IP = _cfg.PC_IP
    DEFAULT_ZMQ_PORT = getattr(_cfg, "ZMQ_LIDAR_PORT", 50507)
except ImportError:
    DEFAULT_PC_IP = "127.0.0.1"
    DEFAULT_ZMQ_PORT = 50507

DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 460800
STATS_INTERVAL = 5.0


async def scan_loop(serial_port, baud, zmq_sock):
    """Read scans from RPLidar C1 and publish over ZMQ."""
    print(f"[LiDAR] Connecting to {serial_port} @ {baud} baud...")
    lidar = RPLidar(serial_port, baud)
    print("[LiDAR] Connected. Starting scan...")

    scan_task = asyncio.create_task(lidar.simple_scan())
    queue = lidar.output_queue

    scan_buffer = []
    last_angle = -1
    scan_count = 0
    total_scans = 0
    last_stats = time.time()

    try:
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                print("[LiDAR] Timeout — no data for 2s")
                continue

            angle = data["a_deg"]
            distance = data["d_mm"]
            quality = data["q"]

            # Detect new scan (angle wraps around from ~360 to ~0)
            if angle < last_angle - 180:
                # Publish completed scan
                if len(scan_buffer) > 50:
                    msg = json.dumps({
                        "n": len(scan_buffer),
                        "ts": time.time(),
                        "pts": scan_buffer,
                    })
                    zmq_sock.send_string(msg)
                    scan_count += 1
                    total_scans += 1

                    now = time.time()
                    if now - last_stats >= STATS_INTERVAL:
                        hz = scan_count / (now - last_stats)
                        print(f"[LiDAR] {total_scans} total | {hz:.1f} Hz | "
                              f"{len(scan_buffer)} pts/scan")
                        scan_count = 0
                        last_stats = now

                scan_buffer = []

            if quality > 0 and distance > 0:
                scan_buffer.append({
                    "a": round(angle, 2),
                    "d": round(distance, 1),
                })
            last_angle = angle

    except asyncio.CancelledError:
        pass
    finally:
        lidar.stop_event.set()
        try:
            lidar.shutdown()
        except Exception:
            pass
        print("[LiDAR] Disconnected.")


def main():
    parser = argparse.ArgumentParser(description="IronBark LiDAR Reader")
    parser.add_argument("--serial", default=DEFAULT_SERIAL_PORT,
                        help=f"Serial port (default: {DEFAULT_SERIAL_PORT})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                        help=f"Baud rate (default: {DEFAULT_BAUD})")
    parser.add_argument("--pc-ip", default=DEFAULT_PC_IP,
                        help=f"Mac IP (default: {DEFAULT_PC_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_ZMQ_PORT,
                        help=f"ZMQ port (default: {DEFAULT_ZMQ_PORT})")
    args = parser.parse_args()

    # ZMQ PUSH socket → Mac
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(f"tcp://{args.pc_ip}:{args.port}")
    print(f"[LiDAR] ZMQ PUSH → {args.pc_ip}:{args.port}")

    # Graceful shutdown
    def shutdown(sig, frame):
        print("\n[LiDAR] Shutting down...")
        sys.exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    asyncio.run(scan_loop(args.serial, args.baud, sock))

    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
