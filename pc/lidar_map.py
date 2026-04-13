"""
pc/lidar_map.py — IronBark Phase 7: LiDAR Obstacle Map (Mac Side)
Receives 360-degree LiDAR scans from the Pi over ZMQ and provides
simple arc-distance queries for obstacle avoidance.

Usage:
    from lidar_map import LidarMap

    lidar = LidarMap(zmq_context)
    lidar.start()

    # In follow loop:
    fwd  = lidar.get_min_distance(330, 30)    # forward arc
    left = lidar.get_min_distance(30, 150)    # left arc
    right = lidar.get_min_distance(210, 330)  # right arc

    if fwd < 35:  # obstacle ahead (cm)
        if left > right:
            turn_left()
        else:
            turn_right()
"""

import json
import time
import threading
from typing import Optional, List, Tuple

import zmq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class LidarMap:
    """
    Receives LiDAR scans over ZMQ and provides arc-based distance queries.

    The scan is stored as a list of (angle_deg, distance_mm) tuples.
    Arc queries return the minimum distance within an angular range,
    converted to centimeters for consistency with the ultrasonic API.

    Thread-safe: the background receiver thread writes scans under a lock,
    and the query methods read under the same lock.
    """

    def __init__(self, ctx: zmq.Context, port: int = None):
        self.port = port or getattr(config, "ZMQ_LIDAR_PORT", 50507)
        self.ctx = ctx
        self.lock = threading.Lock()
        self.running = False

        # Latest scan: list of (angle_deg, distance_mm)
        self._scan: List[Tuple[float, float]] = []
        self._scan_time: float = 0.0
        self._scan_count: int = 0

        # Stale threshold — ignore scans older than this (seconds)
        self._stale_threshold = getattr(config, "LIDAR_STALE_S", 1.0)

    def start(self):
        """Start the background receiver thread."""
        if self.running:
            return
        self.running = True

        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.CONFLATE, 1)  # keep only latest scan
        self.sock.setsockopt(zmq.RCVTIMEO, 500)
        bind_ip = getattr(config, "PC_IP", "0.0.0.0")
        self.sock.bind(f"tcp://{bind_ip}:{self.port}")

        self.thread = threading.Thread(target=self._receiver, name="LiDAR-Recv",
                                       daemon=True)
        self.thread.start()
        print(f"[LidarMap] PULL socket bound on {bind_ip}:{self.port}")

    def stop(self):
        """Stop the receiver thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.sock.close()

    def _receiver(self):
        """Background thread: receive scans from Pi."""
        while self.running:
            try:
                raw = self.sock.recv_string()
                msg = json.loads(raw)
                pts = msg.get("pts", [])
                scan = [(p["a"], p["d"]) for p in pts]
                with self.lock:
                    self._scan = scan
                    self._scan_time = msg.get("ts", time.time())
                    self._scan_count += 1
            except zmq.Again:
                continue
            except Exception as e:
                print(f"[LidarMap] Error: {e}")

    def _is_stale(self) -> bool:
        """Check if the latest scan is too old to trust."""
        return time.time() - self._scan_time > self._stale_threshold

    def get_min_distance(self, start_angle: float, end_angle: float,
                          max_range_cm: float = 500) -> float:
        """
        Get the minimum distance (cm) of any point in the angular arc
        from start_angle to end_angle (degrees, clockwise).

        Angles are 0=forward, 90=right, 180=behind, 270=left.
        The arc wraps around 360 if start > end (e.g., 330 to 30 = forward).

        Returns max_range_cm if no points are in the arc or scan is stale.
        """
        with self.lock:
            if not self._scan or self._is_stale():
                return max_range_cm
            scan = list(self._scan)

        min_dist = max_range_cm
        for angle, dist_mm in scan:
            if self._angle_in_arc(angle, start_angle, end_angle):
                dist_cm = dist_mm / 10.0
                if 0.5 < dist_cm < min_dist:  # filter out <5mm (noise)
                    min_dist = dist_cm
        return min_dist

    def get_forward_distance(self) -> float:
        """Min distance in the forward arc (±30° = 330° to 30°)."""
        fwd_half = getattr(config, "LIDAR_FORWARD_ARC_HALF", 30)
        return self.get_min_distance(360 - fwd_half, fwd_half)

    def get_left_distance(self) -> float:
        """Min distance in the left arc (30° to 150°)."""
        return self.get_min_distance(30, 150)

    def get_right_distance(self) -> float:
        """Min distance in the right arc (210° to 330°)."""
        return self.get_min_distance(210, 330)

    def get_rear_distance(self) -> float:
        """Min distance in the rear arc (150° to 210°)."""
        return self.get_min_distance(150, 210)

    def get_scan_age_ms(self) -> float:
        """How old the latest scan is (milliseconds)."""
        with self.lock:
            return (time.time() - self._scan_time) * 1000

    def get_scan_count(self) -> int:
        """Total scans received since start."""
        with self.lock:
            return self._scan_count

    def has_data(self) -> bool:
        """True if at least one scan has been received and it's not stale."""
        with self.lock:
            return self._scan_count > 0 and not self._is_stale()

    def get_scan(self):
        """Get a copy of the latest scan for visualization. Returns list of (angle_deg, distance_mm)."""
        with self.lock:
            return list(self._scan)

    def find_best_direction(self, obstacle_cm: float = 50) -> tuple:
        """
        Find the direction with the most free space by scanning all angles.
        Returns (best_angle_deg, best_distance_cm).

        Divides the 360° scan into 12 sectors (30° each), finds the minimum
        distance in each sector, then picks the sector with the largest
        minimum distance. This is a simple "gap finder" — it always points
        toward the most open space.

        Returns (best_angle, best_dist_cm) where best_angle is the center
        of the best sector (0=forward, 90=right, 180=behind, 270=left).
        """
        with self.lock:
            if not self._scan or self._is_stale():
                return (0, 999)  # default forward
            scan = list(self._scan)

        # 12 sectors of 30° each
        num_sectors = 12
        sector_size = 360.0 / num_sectors
        sector_mins = [999.0] * num_sectors

        for angle, dist_mm in scan:
            dist_cm = dist_mm / 10.0
            if dist_cm < 0.5:
                continue
            sector = int(angle / sector_size) % num_sectors
            if dist_cm < sector_mins[sector]:
                sector_mins[sector] = dist_cm

        # Find sector with largest minimum distance (most open)
        best_sector = 0
        best_dist = 0
        for i, d in enumerate(sector_mins):
            if d > best_dist:
                best_dist = d
                best_sector = i

        best_angle = best_sector * sector_size + sector_size / 2
        return (best_angle, best_dist)

    def get_direction_to_go(self, obstacle_cm: float = 50) -> str:
        """
        High-level: which way should the dog go?
        Returns "forward", "turn_left", "turn_right", "backward", or "stop".

        Uses find_best_direction() to find the most open space,
        then maps the angle to a motor command. Includes backward
        because turns also move the dog forward (2-DOF legs).
        """
        fwd = self.get_forward_distance()
        rear = self.get_rear_distance()

        # If very close to a wall ahead, MUST back up first.
        # Turning won't help because turns move forward too.
        if fwd < 20:
            if rear > 30:
                return "backward"
            else:
                return "stop"  # boxed in

        best_angle, best_dist = self.find_best_direction(obstacle_cm)

        if best_dist < obstacle_cm:
            # All forward/side directions blocked — back up if possible
            if rear > 30:
                return "backward"
            return "stop"

        # Map angle to action (0=forward, 90=right, 180=behind, 270=left)
        if best_angle <= 45 or best_angle >= 315:
            return "forward"
        elif 135 < best_angle <= 225:
            # Best direction is behind — back up instead of 180° turn
            if rear > 30:
                return "backward"
            return "turn_left"  # fallback to 180° turn
        elif 45 < best_angle <= 135:
            return "turn_right"
        else:
            return "turn_left"

    @staticmethod
    def _angle_in_arc(angle: float, start: float, end: float) -> bool:
        """Check if angle is within the arc from start to end (clockwise)."""
        angle = angle % 360
        start = start % 360
        end = end % 360
        if start <= end:
            return start <= angle <= end
        else:
            # Arc wraps around 360 (e.g., 330 to 30)
            return angle >= start or angle <= end
