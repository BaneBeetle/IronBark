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
    description:     str
    navigation_hint: str
    latency_ms:      float
    model:           str


NAVIGATION_PROMPT_TEMPLATE = """You are the vision system of an autonomous robot dog.
Analyze this image and respond with EXACTLY two lines:
LINE 1 - SCENE: A brief description of what you see (obstacles, people, layout).
LINE 2 - NAVIGATE: Your recommended navigation action (move forward, turn left/right, stop, etc.).

{person_context}

Be concise. Focus on: obstacles blocking the path, safe directions to move, people present."""


class VLMReasoner:
    def __init__(self, model="llava:7b", host="http://localhost:11434", timeout_s=60):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self._generate_url = f"{self.host}/api/generate"
        self._tags_url = f"{self.host}/api/tags"
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
        return VLMResponse(description=description, navigation_hint=navigation_hint,
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
