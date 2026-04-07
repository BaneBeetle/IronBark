from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import requests

from .yolo_detector import PersonDetection


@dataclass
class VLMResponse:
    """Legacy obstacle avoidance response (Phase 4). Kept for compat."""
    description:       str
    navigation_hint:   str
    action:            str = "forward"
    action_confidence: float = 0.0
    latency_ms:        float = 0.0
    model:             str = ""


@dataclass
class SituationResponse:
    """Phase 6A: VLM scene reading → behavior mode."""
    mode:        str             # ACTIVE / GENTLE / CALM / PLAYFUL / SOCIAL
    description: str = ""        # scene description from VLM
    latency_ms:  float = 0.0
    model:       str = ""


@dataclass
class ExploreResponse:
    """Phase 6B: VLM navigation guidance for EXPLORE state."""
    direction:  str             # FORWARD / LEFT / RIGHT / BACK
    reasoning:  str = ""        # why VLM chose this direction
    latency_ms: float = 0.0
    model:      str = ""


VALID_MODES = {"ACTIVE", "GENTLE", "CALM", "PLAYFUL", "SOCIAL"}
VALID_DIRECTIONS = {"FORWARD", "LEFT", "RIGHT", "BACK"}

SITUATION_PROMPT = """You are the vision system of a small robot dog following its owner.
Analyze the scene and choose the most appropriate behavior mode.

Respond with EXACTLY two lines:
LINE 1 - SCENE: Brief description of the environment and what the owner is doing.
LINE 2 - MODE: Exactly one of: ACTIVE, GENTLE, CALM, PLAYFUL, SOCIAL

Rules:
- ACTIVE: Owner is standing, walking, moving around, or outdoors.
- GENTLE: Owner is sitting on a couch, chair, or floor, or crouching down.
- CALM: Owner is at a desk working, eating, reading, or appears to be resting.
- PLAYFUL: Children present, energetic activity, owner reaching toward camera.
- SOCIAL: Multiple people visible, owner talking to someone, gathering.
- Default to ACTIVE if uncertain.

Be concise. Respond with exactly two lines."""

EXPLORE_PROMPT = """You are a small robot dog that has lost its owner and is searching for them in a home.
Analyze this image and decide which direction to explore to find them.

Respond with EXACTLY two lines:
LINE 1 - SCENE: Brief description of what you see (doorways, hallways, open spaces, walls).
LINE 2 - DIRECTION: Exactly one of: FORWARD, LEFT, RIGHT, BACK

Rules:
- FORWARD: Open space, doorway, or hallway ahead worth exploring.
- LEFT: Doorway, opening, or unexplored space to the left.
- RIGHT: Doorway, opening, or unexplored space to the right.
- BACK: Dead end, wall ahead, or nothing useful — turn around.
- Prefer doorways and open spaces over walls and furniture.

Be concise. Respond with exactly two lines."""

NAVIGATION_PROMPT_TEMPLATE = """You are the vision system of a small robot dog following its owner.
Analyze this image and respond with EXACTLY two lines:

LINE 1 - SCENE: Brief description of obstacles, people, and layout.
LINE 2 - ACTION: Exactly one of: FORWARD, TURN_LEFT, TURN_RIGHT, STOP

Rules for choosing ACTION:
- FORWARD: Path ahead is clear, safe to continue toward the person.
- TURN_LEFT: Obstacle on the right or ahead-right; turn left to go around it.
- TURN_RIGHT: Obstacle on the left or ahead-left; turn right to go around it.
- STOP: Danger directly ahead, no safe direction, or very close to a wall/object.

{person_context}

Be concise. Respond with exactly two lines."""


class VLMReasoner:
    # Map free-text phrases to valid motor actions (checked longest-first)
    _ACTION_KEYWORDS = {
        "move forward":  "forward",
        "go forward":    "forward",
        "go straight":   "forward",
        "forward":       "forward",
        "proceed":       "forward",
        "continue":      "forward",
        "turn_left":     "turn_left",
        "turn left":     "turn_left",
        "go left":       "turn_left",
        "veer left":     "turn_left",
        "turn_right":    "turn_right",
        "turn right":    "turn_right",
        "go right":      "turn_right",
        "veer right":    "turn_right",
        "stop":          "stop",
        "halt":          "stop",
        "wait":          "stop",
    }

    def __init__(self, model="llava:7b", host="http://localhost:11434", timeout_s=60):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self._generate_url = f"{self.host}/api/generate"
        self._tags_url = f"{self.host}/api/tags"
        # Sort keywords longest-first for greedy matching
        self._sorted_keywords = sorted(self._ACTION_KEYWORDS.keys(),
                                       key=len, reverse=True)
        print(f"[VLMReasoner] Model: {model} @ {host}")

    def _encode_frame(self, frame, jpeg_quality=75):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        success, jpeg_bytes = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")
        return base64.b64encode(jpeg_bytes.tobytes()).decode("utf-8")

    def _build_prompt(self, detections):
        if detections:
            n = len(detections)
            confs = [f"{d.confidence:.0%}" for d in detections]
            person_context = f"Note: {n} person(s) already detected by YOLO with confidences: {', '.join(confs)}."
        else:
            person_context = "No persons detected by YOLO in this frame."
        return NAVIGATION_PROMPT_TEMPLATE.format(person_context=person_context)

    def reason(self, frame, detections=None, jpeg_quality=85):
        t_start = time.perf_counter()
        image_b64 = self._encode_frame(frame, jpeg_quality)
        prompt = self._build_prompt(detections)

        payload = {
            "model": self.model, "prompt": prompt,
            "images": [image_b64], "stream": False,
            "options": {"temperature": 0.1, "num_predict": 150},
        }

        try:
            response = requests.post(self._generate_url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            raw_text = response.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raw_text = "ERROR: Cannot connect to Ollama. Is it running? Try: ollama serve"
        except requests.exceptions.Timeout:
            raw_text = f"ERROR: VLM timed out (>{self.timeout_s}s)."
        except Exception as e:
            raw_text = f"ERROR: {e}"

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        description, navigation_hint = self._parse_response(raw_text)
        action, action_confidence = self._extract_action(raw_text, navigation_hint)
        return VLMResponse(description=description, navigation_hint=navigation_hint,
                           action=action, action_confidence=action_confidence,
                           latency_ms=latency_ms, model=self.model)

    def _parse_response(self, raw_text):
        description = raw_text
        navigation_hint = ""
        lines = raw_text.split("\n")
        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()
            cleaned = stripped
            for prefix in ["LINE 1 -", "LINE 2 -", "LINE 1:", "LINE 2:",
                           "LINE1:", "LINE2:", "1.", "2.", "1)", "2)"]:
                if upper.startswith(prefix):
                    cleaned = stripped[len(prefix):].strip()
                    upper = cleaned.upper()
                    break
            if upper.startswith("SCENE:") or upper.startswith("SCENE -"):
                sep = ":" if ":" in cleaned else "-"
                description = cleaned.split(sep, 1)[1].strip() if sep in cleaned else cleaned
            elif upper.startswith("NAVIGATE:") or upper.startswith("NAVIGATE -"):
                sep = ":" if ":" in cleaned else "-"
                navigation_hint = cleaned.split(sep, 1)[1].strip() if sep in cleaned else cleaned
            elif upper.startswith("NAVIGATION:") or upper.startswith("NAVIGATION -"):
                sep = ":" if ":" in cleaned else "-"
                navigation_hint = cleaned.split(sep, 1)[1].strip() if sep in cleaned else cleaned
            elif upper.startswith("ACTION:") or upper.startswith("ACTION -"):
                sep = ":" if ":" in cleaned else "-"
                if not navigation_hint:
                    navigation_hint = cleaned.split(sep, 1)[1].strip() if sep in cleaned else cleaned
            elif any(kw in stripped.lower() for kw in ["move forward", "turn left", "turn right",
                                                        "stop", "go straight", "proceed"]):
                if not navigation_hint:
                    navigation_hint = stripped.split(":", 1)[1].strip() if ":" in stripped else stripped

        if not navigation_hint:
            sentences = [s.strip() for s in raw_text.replace("\n", " ").split(".") if s.strip()]
            navigation_hint = (sentences[-1] + ".") if sentences else "No navigation advice available."
        if not description:
            description = raw_text
        return description, navigation_hint

    def _extract_action(self, raw_text: str, navigation_hint: str):
        """Parse structured action from VLM output. Returns (action, confidence)."""
        text_upper = raw_text.upper()

        # Priority 1: Exact ACTION line (e.g., "ACTION: FORWARD")
        for line in raw_text.split("\n"):
            stripped = line.strip().upper()
            for prefix in ("ACTION:", "LINE 2 - ACTION:", "LINE 2:", "2."):
                if stripped.startswith(prefix):
                    token = stripped[len(prefix):].strip().replace(" ", "_")
                    if token in ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"):
                        return token.lower(), 1.0

        # Priority 2: Keyword matching in navigation_hint (longest-first)
        hint_lower = navigation_hint.lower()
        for keyword in self._sorted_keywords:
            if keyword in hint_lower:
                return self._ACTION_KEYWORDS[keyword], 0.7

        # Priority 3: Keyword matching in full raw text
        raw_lower = raw_text.lower()
        for keyword in self._sorted_keywords:
            if keyword in raw_lower:
                return self._ACTION_KEYWORDS[keyword], 0.5

        # Default: forward (safe fallback)
        return "forward", 0.0

    # ── Phase 6A: Situation query ──────────────────────────────────────

    def situation_query(self, frame, jpeg_quality=75):
        """Classify the scene into a behavior mode. Returns SituationResponse."""
        t_start = time.perf_counter()
        image_b64 = self._encode_frame(frame, jpeg_quality)

        payload = {
            "model": self.model, "prompt": SITUATION_PROMPT,
            "images": [image_b64], "stream": False,
            "options": {"temperature": 0.1, "num_predict": 80},
        }

        raw_text = ""
        try:
            response = requests.post(self._generate_url, json=payload,
                                     timeout=self.timeout_s)
            response.raise_for_status()
            raw_text = response.json().get("response", "").strip()
        except Exception as e:
            raw_text = f"ERROR: {e}"

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        mode, description = self._parse_situation(raw_text)
        return SituationResponse(mode=mode, description=description,
                                 latency_ms=latency_ms, model=self.model)

    def _parse_situation(self, raw_text):
        """Extract MODE and SCENE from VLM situation response."""
        description = ""
        mode = "ACTIVE"  # safe default
        for line in raw_text.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()
            # Strip common line prefixes
            for prefix in ["LINE 1 -", "LINE 2 -", "LINE 1:", "LINE 2:",
                           "1.", "2.", "1)", "2)"]:
                if upper.startswith(prefix):
                    stripped = stripped[len(prefix):].strip()
                    upper = stripped.upper()
                    break
            if upper.startswith("SCENE:"):
                description = stripped.split(":", 1)[1].strip()
            elif upper.startswith("MODE:"):
                token = stripped.split(":", 1)[1].strip().upper()
                if token in VALID_MODES:
                    mode = token
        return mode, description

    # ── Phase 6B: Explore query ────────────────────────────────────────

    def explore_query(self, frame, jpeg_quality=75):
        """Ask VLM which direction to explore. Returns ExploreResponse."""
        t_start = time.perf_counter()
        image_b64 = self._encode_frame(frame, jpeg_quality)

        payload = {
            "model": self.model, "prompt": EXPLORE_PROMPT,
            "images": [image_b64], "stream": False,
            "options": {"temperature": 0.1, "num_predict": 80},
        }

        raw_text = ""
        try:
            response = requests.post(self._generate_url, json=payload,
                                     timeout=self.timeout_s)
            response.raise_for_status()
            raw_text = response.json().get("response", "").strip()
        except Exception as e:
            raw_text = f"ERROR: {e}"

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        direction, reasoning = self._parse_explore(raw_text)
        return ExploreResponse(direction=direction, reasoning=reasoning,
                               latency_ms=latency_ms, model=self.model)

    def _parse_explore(self, raw_text):
        """Extract DIRECTION and SCENE from VLM explore response."""
        reasoning = ""
        direction = "FORWARD"  # safe default
        for line in raw_text.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()
            for prefix in ["LINE 1 -", "LINE 2 -", "LINE 1:", "LINE 2:",
                           "1.", "2.", "1)", "2)"]:
                if upper.startswith(prefix):
                    stripped = stripped[len(prefix):].strip()
                    upper = stripped.upper()
                    break
            if upper.startswith("SCENE:"):
                reasoning = stripped.split(":", 1)[1].strip()
            elif upper.startswith("DIRECTION:"):
                token = stripped.split(":", 1)[1].strip().upper()
                if token in VALID_DIRECTIONS:
                    direction = token
        return direction, reasoning

    # ── Utilities ──────────────────────────────────────────────────────

    def health_check(self):
        try:
            return requests.get(f"{self.host}/api/tags", timeout=3).status_code == 200
        except Exception:
            return False

    def list_models(self):
        try:
            response = requests.get(self._tags_url, timeout=5)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except Exception:
            return []
