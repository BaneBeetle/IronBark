# Security Audit Report

**Project:** IronBark  
**Date:** 2026-04-10  
**Scanned:** 33 files across 8 directories (excluding `.git/`, `.venv/`)

## Summary

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 1 |
| Medium | 2 |
| Low | 2 |

**Overall posture: Good.** No leaked secrets, no hardcoded credentials, no passwords in source or git history. The `.env` pattern was implemented correctly from the start. The findings below are architectural risks in the ZMQ communication layer, not credential leaks.

---

## High Findings

### [HIGH-001] ZeroMQ sockets bound to all interfaces with no authentication

- **Files:**
  - `pc/follower.py:162` — `tcp://*:5556` (follow-me commands)
  - `pc/follower.py:509` — `tcp://*:50505` (webcam stream)
  - `pc/follower.py:521` — `tcp://*:50506` (ribbon cam stream)
  - `pi/motor_controller.py:250` — `tcp://*:5557` (teleop commands)
  - `pi/motor_controller.py:256` — `tcp://*:5558` (telemetry)
  - `pc/enroll_owner.py:43` — `tcp://*:50505`
  - `pc/bbox_calibrate.py:58` — `tcp://*:50505`
  - `pi/remote_control.py:89,95` — `tcp://*:5557,5558`
- **Pattern:** All ZMQ sockets bind to `0.0.0.0` (all network interfaces) with no authentication, encryption, or access control.
- **Risk:** Any device on the same network can:
  - **Inject motor commands** to ports 5556/5557 — make the dog walk off a table, spin indefinitely, or bark at 3 AM.
  - **Read the camera streams** on ports 50505/50506 — live video surveillance of wherever the dog is.
  - **Read telemetry** on port 5558 — battery level, ultrasonic distance, current action.
- **Mitigating factor:** Tailscale VPN. If both nodes communicate over Tailscale, the `100.x.x.x` IPs are only reachable within your Tailscale network. However, the sockets bind `0.0.0.0`, meaning they're *also* exposed on the local LAN interface. An attacker on the same WiFi could connect directly via the `192.168.x.x` address.
- **Fix (short-term):** Bind to specific interface instead of all:
  ```python
  # Before (INSECURE — listens on all interfaces)
  sock.bind(f"tcp://*:{port}")

  # After (binds only to Tailscale interface)
  sock.bind(f"tcp://{config.PC_IP}:{port}")   # Mac-side
  sock.bind(f"tcp://{config.PI_IP}:{port}")   # Pi-side (for Pi-bound sockets)
  ```
- **Fix (long-term):** Enable ZMQ CurveZMQ encryption:
  ```python
  import zmq.auth
  # Generate keypairs: zmq.auth.create_certificates("/path/to/certs", "server")
  server_public, server_secret = zmq.auth.load_certificate("server.key_secret")
  sock.curve_secretkey = server_secret
  sock.curve_publickey = server_public
  sock.curve_server = True  # Server side
  ```
  This adds encryption + mutual authentication. Only nodes with matching keypairs can connect.

---

## Medium Findings

### [MED-001] Motor commands accepted without validation or rate limiting

- **File:** `pi/motor_controller.py:365-415` (`_receive_command()`)
- **Pattern:** The motor controller deserializes JSON from ZMQ and passes `action`, `speed`, `step_count` directly to the PiDog SDK with no input validation beyond what the SDK itself enforces.
- **Risk:** A malformed or malicious command (e.g., `speed: 99999`, `step_count: 1000`) could cause unexpected servo behavior. Combined with HIGH-001 (no ZMQ auth), any device on the network can send these.
- **Fix:** Add input validation before execution:
  ```python
  VALID_ACTIONS = {"forward", "backward", "turn_left", "turn_right", "stop"}
  speed = max(0, min(100, msg.get("speed", 80)))
  step_count = max(1, min(10, msg.get("step_count", 2)))
  action = msg.get("action", "stop")
  if action not in VALID_ACTIONS:
      action = "stop"
  ```

### [MED-002] SIGKILL used for process cleanup without graceful shutdown attempt

- **Files:**
  - `pi/motor_controller.py:58` — `os.kill(int(pid), signal.SIGKILL)` on orphaned PiDog processes
  - `pi/motor_controller.py:588` — `os.kill(sp.pid, signal.SIGKILL)` on sensor subprocess
  - `pi/remote_control.py:35,282` — Same pattern
  - `pi/read_distance.py:24,44` — Same pattern
- **Pattern:** SIGKILL is used as a first resort for cleaning up PiDog sensor subprocesses. SIGKILL cannot be caught, so the target process gets no chance to release GPIO pins, close file descriptors, or flush buffers.
- **Risk:** GPIO pins left in an indeterminate state. Not a security vulnerability per se, but a reliability issue that causes "GPIO busy" errors on next startup. In a worst case, a held GPIO pin could keep a servo energized, drawing current and generating heat.
- **Mitigating factor:** The cleanup functions try SIGTERM in the `_shutdown()` path before escalating to SIGKILL. The SIGKILL in `cleanup_orphaned_pidog()` is for processes that survived a previous unclean exit (where SIGTERM already failed).
- **Fix:** Add a SIGTERM attempt with timeout before SIGKILL in the orphan cleanup:
  ```python
  os.kill(int(pid), signal.SIGTERM)
  time.sleep(0.5)
  # Only SIGKILL if still alive
  try:
      os.kill(int(pid), 0)  # Check if still running
      os.kill(int(pid), signal.SIGKILL)
  except ProcessLookupError:
      pass  # Already exited
  ```

---

## Low Findings

### [LOW-001] Ollama VLM endpoint is unauthenticated HTTP on localhost

- **File:** `config.py:54` — `VLM_HOST = "http://localhost:11434"`
- **Pattern:** The Ollama API is accessed over plaintext HTTP. No API key, no authentication.
- **Risk:** Minimal in practice — Ollama binds to `127.0.0.1` by default, so only local processes can access it. However, if Ollama is reconfigured to bind `0.0.0.0` (common for remote access), anyone on the network can query your VLM.
- **Fix:** No code change needed. Ensure Ollama stays on `localhost`. If remote access is needed, use an SSH tunnel or reverse proxy with auth.

### [LOW-002] Owner face embedding stored as unencrypted .npy file

- **File:** `data/owner_embedding.npy` (gitignored, not committed)
- **Pattern:** The 512-dimensional ArcFace face embedding is saved as a raw NumPy file. It's biometric data — a mathematical representation of the owner's face.
- **Risk:** Low. The embedding cannot be reverse-engineered back into a face image. However, it could theoretically be used for face matching by someone who obtains the file. It's gitignored and never committed.
- **Mitigating factor:** Already in `.gitignore`. The `data/` directory with `*.npy` is excluded from version control.
- **Fix (optional):** Encrypt at rest with a passphrase if the device is at risk of physical theft:
  ```python
  from cryptography.fernet import Fernet
  key = Fernet.generate_key()  # Store in .env
  f = Fernet(key)
  encrypted = f.encrypt(embedding.tobytes())
  ```

---

## Clean Areas (no findings)

| Category | Result |
|----------|--------|
| Hardcoded API keys / tokens | None found |
| Hardcoded passwords | None found |
| SSH private keys in repo | None found |
| `.env` committed to git | No — properly gitignored, never committed |
| `.env.example` leaking real values | Clean — uses `<placeholder>` format |
| Tailscale IPs in source code | None in `.py` or `.sh` files (only in `.env` which is gitignored) |
| Tailscale IPs in docs | Sanitized to placeholders (done this session) |
| SQL injection | No SQL in project |
| Command injection (`shell=True`) | None found |
| `eval()` / `exec()` | None found (one `__import__` in `pc_receiver.py` is benign path setup) |
| Disabled SSL verification | None found |
| Secrets in git history | None found — clean history |
| WiFi credentials | None found |
| `.pem` / `.key` files | Only pip's CA bundle in `.venv/` (standard, not a secret) |

---

## Recommendations

### Immediate Actions (do today)
Nothing critical. No rotations needed — there are no leaked secrets.

### Short-term Improvements (this sprint)

1. **Bind ZMQ to specific interfaces** instead of `0.0.0.0`. This is a one-line change per socket and eliminates the LAN exposure risk while keeping Tailscale working.

2. **Add input validation** to `motor_controller.py` command parsing. Clamp `speed` to 0-100, `step_count` to 1-10, whitelist `action` values.

### Long-term Hardening (future)

3. **ZMQ CurveZMQ encryption** for mutual authentication between Pi and Mac. Prevents any unauthorized device from sending commands or reading video even on the Tailscale network.

4. **SIGTERM-first cleanup** in the orphan PiDog process killer.

5. **Consider encrypted storage** for the face embedding file if the Pi is deployed in an untrusted physical environment.

### Pre-commit Hook Setup

To prevent future secret leaks if the project grows:

```bash
pip install pre-commit
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
EOF
pre-commit install
```

This scans every commit for accidental secret inclusion before it reaches the repo.
