"""
pi/lidar_reader.py — IronBark Phase 7: RPLidar C1 Scanner
Reads 360-degree LiDAR scans and publishes them to the Mac over ZMQ.

The C1 produces ~500 points per scan at 10Hz. Each point has an angle
(0-360 degrees) and distance (millimeters). We pack each scan as JSON
and PUSH to the Mac, which uses it for obstacle avoidance.

Hardware: Slamtec RPLidar C1 via USB-to-UART adapter → /dev/ttyUSB0
Install:  pip3 install rplidar-roboticia

Usage:
    python3 -u lidar_reader.py [--port /dev/ttyUSB0] [--baud 460800]
"""

import argparse
import json
import sys
import time
import signal

import zmq

try:
    from rplidar import RPLidar
    RPLIDAR_LIB = "roboticia"
except ImportError:
    try:
        from rplidarc1 import RPLidarC1
        RPLIDAR_LIB = "rplidarc1"
    except ImportError:
        print("ERROR: No RPLidar library found.")
        print("  Try: pip3 install rplidar-roboticia")
        print("  Or:  pip3 install rplidarc1")
        sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────────
try:
    import config as _cfg
    DEFAULT_PC_IP = _cfg.PC_IP
    DEFAULT_ZMQ_PORT = getattr(_cfg, "ZMQ_LIDAR_PORT", 50507)
except ImportError:
    DEFAULT_PC_IP = "127.0.0.1"
    DEFAULT_ZMQ_PORT = 50507

# RPLidar C1 uses 460800 baud (vs A1/A2 at 115200)
DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 460800

# How often to print stats
STATS_INTERVAL = 5.0


def run_roboticia(serial_port, baud, zmq_sock):
    """Read scans using the Roboticia rplidar library (sync iterator)."""
    lidar = RPLidar(serial_port, baudrate=baud)

    info = lidar.get_info()
    health = lidar.get_health()
    print(f"[LiDAR] Model: {info.get('model', '?')} "
          f"FW: {info.get('firmware', '?')} "
          f"HW: {info.get('hardware', '?')}")
    print(f"[LiDAR] Health: {health[0]} (code={health[1]})")

    scan_count = 0
    last_stats = time.time()

    try:
        for scan in lidar.iter_scans(max_buf_meas=5000, min_len=100):
            # scan = list of (quality, angle_deg, distance_mm) tuples
            points = []
            for quality, angle, distance in scan:
                if quality > 0 and distance > 0:
                    points.append({
                        "a": round(angle, 2),
                        "d": round(distance, 1),
                    })

            if points:
                msg = json.dumps({
                    "n": len(points),
                    "ts": time.time(),
                    "pts": points,
                })
                zmq_sock.send_string(msg)

            scan_count += 1
            now = time.time()
            if now - last_stats >= STATS_INTERVAL:
                hz = scan_count / (now - last_stats)
                print(f"[LiDAR] {scan_count} scans | {hz:.1f} Hz | "
                      f"{len(points)} pts/scan")
                scan_count = 0
                last_stats = now

    except KeyboardInterrupt:
        pass
    finally:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        print("[LiDAR] Disconnected.")


def run_rplidarc1(serial_port, baud, zmq_sock):
    """Read scans using the rplidarc1 library (async)."""
    import asyncio

    async def _scan_loop():
        lidar = RPLidarC1(serial_port, baud)
        queue = lidar.get_output_queue()

        scan_buffer = []
        last_angle = -1
        scan_count = 0
        last_stats = time.time()

        try:
            scan_task = asyncio.create_task(lidar.start_scan())

            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                angle = data["a_deg"]
                distance = data["d_mm"]
                quality = data["q"]

                # Detect new scan (angle wraps around from ~360 to ~0)
                if angle < last_angle - 180:
                    # Publish completed scan
                    if len(scan_buffer) > 100:
                        msg = json.dumps({
                            "n": len(scan_buffer),
                            "ts": time.time(),
                            "pts": scan_buffer,
                        })
                        zmq_sock.send_string(msg)

                        scan_count += 1
                        now = time.time()
                        if now - last_stats >= STATS_INTERVAL:
                            hz = scan_count / (now - last_stats)
                            print(f"[LiDAR] {scan_count} scans | {hz:.1f} Hz | "
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
            await lidar.stop_scan()
            lidar.disconnect()
            print("[LiDAR] Disconnected.")

    asyncio.run(_scan_loop())


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
    print(f"[LiDAR] ZMQ PUSH connected to {args.pc_ip}:{args.port}")
    print(f"[LiDAR] Serial: {args.serial} @ {args.baud} baud")
    print(f"[LiDAR] Library: {RPLIDAR_LIB}")

    # Graceful shutdown
    def shutdown(sig, frame):
        print("\n[LiDAR] Shutting down...")
        sys.exit(0)
    signal.signal(signal.SIGTERM, shutdown)

    if RPLIDAR_LIB == "roboticia":
        run_roboticia(args.serial, args.baud, sock)
    else:
        run_rplidarc1(args.serial, args.baud, sock)

    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
