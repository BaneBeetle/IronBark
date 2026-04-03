"""
pc/follower.py — Project IronBark Phase 3
The Follow-Me Brain. Only "forward" moves legs. "stop" = body_stop().
Turning is disabled. Head tracks owner with dead zone + smoothing.
"""

import sys
import time
import json
import struct
from enum import Enum
from pathlib import Path

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class State(Enum):
    IDLE = "IDLE"
    FOLLOW = "FOLLOW"
    SEARCH = "SEARCH"
    EXPLORE = "EXPLORE"


class Command:
    def __init__(self, action="stop", speed=80, head_yaw=0.0,
                 head_pitch=None, step_count=2, bark=False):
        self.action = action
        self.speed = speed
        self.head_yaw = head_yaw
        self.head_pitch = head_pitch if head_pitch is not None else config.HEAD_DEFAULT_PITCH
        self.step_count = step_count
        self.bark = bark

    def to_json(self):
        return json.dumps({
            "action": self.action, "speed": self.speed,
            "head_yaw": self.head_yaw, "head_pitch": self.head_pitch,
            "step_count": self.step_count, "bark": self.bark,
        }).encode("utf-8")


class Follower:
    def __init__(self):
        self.state = State.IDLE
        self.last_owner_seen = 0.0
        self.search_start_time = 0.0
        self._unconfirmed_count = 0
        self._should_bark = False
        self._last_cmd_time = 0.0
        self._cmd_interval = 0.3
        self._smoothed_head_yaw = 0.0

        self.ctx = zmq.Context()
        self.cmd_sock = self.ctx.socket(zmq.PUB)
        self.cmd_sock.bind(f"tcp://*:{config.CMD_PORT}")
        print(f"[Follower] Command PUB bound on port {config.CMD_PORT}")

    def update(self, perception_result):
        now = time.time()
        owner = self._find_owner(perception_result)

        bark = False
        if self._should_bark:
            bark = True
            self._should_bark = False
            self._unconfirmed_count = 0

        if owner is not None:
            self.last_owner_seen = now
            self._transition(State.FOLLOW)
            cmd = self._follow(owner, perception_result)
        elif bark:
            cmd = Command("stop", head_yaw=self._smoothed_head_yaw, bark=True)
        elif now - self.last_owner_seen < 2.0 and self.state == State.FOLLOW:
            cmd = Command("forward", speed=98, head_yaw=self._smoothed_head_yaw, step_count=2)
        elif now - self.last_owner_seen < config.SEARCH_TIMEOUT_S:
            if self.state != State.SEARCH:
                self._transition(State.SEARCH)
                self.search_start_time = now
            cmd = self._search(now)
        elif now - self.last_owner_seen < config.SEARCH_TIMEOUT_S + config.EXPLORE_TIMEOUT_S:
            if self.state != State.EXPLORE:
                self._transition(State.EXPLORE)
            cmd = self._explore(now)
        else:
            self._transition(State.IDLE)
            cmd = Command("stop", head_yaw=0.0)

        if now - self._last_cmd_time >= self._cmd_interval:
            self.send_command(cmd)
            self._last_cmd_time = now
        return cmd

    def _find_owner(self, result):
        if not hasattr(result, 'detections') or not result.detections:
            return None
        if hasattr(result, 'face_matches') and result.face_matches:
            for i, match in enumerate(result.face_matches):
                if match.is_owner and i < len(result.detections):
                    self._unconfirmed_count = 0
                    return result.detections[i]
        if len(result.detections) > 0 and self.state == State.FOLLOW:
            self._unconfirmed_count += 1
            if self._unconfirmed_count > 10:
                self._should_bark = True
        return None

    def _follow(self, owner_detection, result):
        cx, cy = owner_detection.center
        area = owner_detection.area
        frame_area = 1280 * 720
        area_ratio = area / frame_area
        offset_x = cx - config.FRAME_CENTER_X

        target_yaw = max(-90, min(90, -offset_x * 90 / config.FRAME_CENTER_X))
        if abs(target_yaw - self._smoothed_head_yaw) < 8.0:
            head_yaw = self._smoothed_head_yaw
        else:
            self._smoothed_head_yaw = self._smoothed_head_yaw * 0.8 + target_yaw * 0.2
            head_yaw = self._smoothed_head_yaw

        head_pitch = -45 + (area_ratio / config.TARGET_AREA_RATIO) * 30
        head_pitch = max(-45, min(-15, head_pitch))

        if area_ratio > config.TARGET_AREA_RATIO:
            return Command("stop", head_yaw=head_yaw, head_pitch=head_pitch)

        return Command("forward", speed=98, head_yaw=head_yaw, head_pitch=head_pitch, step_count=2)

    def _search(self, now):
        elapsed = now - self.search_start_time
        cycle = int(elapsed / 3) % 3
        head_yaw = [45.0, -45.0, 0.0][cycle]
        self._smoothed_head_yaw = head_yaw
        return Command("stop", head_yaw=head_yaw)

    def _explore(self, now):
        elapsed = now - self.search_start_time
        cycle = int(elapsed / 4) % 3
        head_yaw = [60.0, -60.0, 0.0][cycle]
        self._smoothed_head_yaw = head_yaw
        return Command("stop", head_yaw=head_yaw)

    def _transition(self, new_state):
        if self.state != new_state:
            print(f"[Follower] {self.state.value} → {new_state.value}")
            self.state = new_state

    def send_command(self, cmd):
        self.cmd_sock.send(cmd.to_json())

    def close(self):
        self.cmd_sock.close()
        self.ctx.term()


if __name__ == "__main__":
    import cv2
    sys.path.insert(0, str(Path(__file__).parent))
    from perception_pipeline import PerceptionPipeline

    print("=" * 60)
    print("IronBark Phase 3 — Follow-Me (Forward Only)")
    print("=" * 60)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.bind(f"tcp://*:{config.ZMQ_PORT}")
    print(f"[Main] PULL socket bound on port {config.ZMQ_PORT}")

    pipeline = PerceptionPipeline(config)
    pipeline.start()
    follower = Follower()

    window = "IronBark Follow-Me"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    def get_latest_frame(socket):
        msg = None
        while True:
            try:
                msg = socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
        return msg

    def decode_frame(raw):
        header_size = struct.calcsize("<qI")
        if len(raw) < header_size:
            return None
        timestamp, payload_len = struct.unpack_from("<qI", raw, 0)
        jpeg_data = raw[header_size:header_size + payload_len]
        return cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    try:
        while True:
            raw = get_latest_frame(sock)
            if raw is None:
                try:
                    raw = sock.recv()
                except zmq.Again:
                    waiting = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(waiting, "Waiting for Pi stream...",
                                (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 255), 2)
                    cv2.imshow(window, waiting)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                    continue

            frame = decode_frame(raw)
            if frame is None:
                continue

            result = pipeline.process_frame(frame)
            cmd = follower.update(result)

            display = pipeline.draw_overlay(frame, result)
            state_text = f"State: {follower.state.value} | Cmd: {cmd.action} spd={cmd.speed}"
            cv2.putText(display, state_text, (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow(window, display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        follower.close()
        pipeline.stop()
        sock.close()
        cv2.destroyAllWindows()
