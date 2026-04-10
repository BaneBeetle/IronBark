# Behavior State Machine

The follower state machine (`pc/follower.py`) controls what the dog does at any moment. It reads perception results every frame and decides whether to follow, search, explore, or idle.

## States

```
                    owner detected
         ┌──────────────────────────────────┐
         │                                  │
         ▼                                  │
    ┌─────────┐   owner lost    ┌─────────┐ │    owner lost     ┌──────────┐
    │  IDLE   │ ◀──── 35s ──── │ EXPLORE │ │ ◀──── 5s ──────  │  SEARCH  │
    └─────────┘                 └─────────┘ │                   └──────────┘
         │                          ▲       │                        ▲
         │   owner                  │       │                        │
         │   detected               │ 5s    │                        │ 2s
         │                          │       │                        │
         │                     ┌────┴───────┴──┐                     │
         └────────────────────▶│    FOLLOW     │─────────────────────┘
                               └───────────────┘
                                  owner lost
```

| State | Entry Condition | Exit Condition | Duration |
|-------|----------------|----------------|----------|
| **IDLE** | Owner lost for 35+ seconds, or system startup | Owner detected | Indefinite |
| **FOLLOW** | Owner detected (face match confirmed) | Owner lost for 2+ seconds | While owner visible |
| **SEARCH** | Owner lost for 2-5 seconds | Owner found (-> FOLLOW) or 5s elapsed (-> EXPLORE) | Up to 5 seconds |
| **EXPLORE** | Owner lost for 5-35 seconds | Owner found (-> FOLLOW) or 35s elapsed (-> IDLE) | Up to 30 seconds |

> **Note:** The 2-second gap between losing the owner and entering SEARCH is a "coast" period. The dog continues forward in FOLLOW state with the last known heading, giving the owner time to reappear before the dog starts searching.

## FOLLOW State

The dog actively tracks and approaches the owner.

### Safety Hierarchy

Actions are evaluated top to bottom. The first matching condition wins:

| Priority | Condition | Action |
|----------|-----------|--------|
| 1. Ultrasonic stop | `0 < distance < 35cm` | Redirect turn toward owner (or stop if straight ahead) |
| 2. Bark hold | `bark_hold_until > now` | Hold head up, optional repeat bark every 3s |
| 3. Arrival | `area_ratio > arrival_ratio` | Re-center if needed, then stop + bark + idle pose |
| 4. Body centering | `abs(offset_x) > 120px` | Turn left/right toward owner |
| 5. Forward cruise | Default | Walk forward at behavior mode speed |

### Steering

```python
offset_x = owner_bbox_center_x - frame_center_x  # (640/2 = 320)

if offset_x < -BODY_TURN_THRESHOLD:    # Owner is left
    action = "turn_left"
elif offset_x > BODY_TURN_THRESHOLD:   # Owner is right
    action = "turn_right"
else:
    action = "forward"                  # Owner is centered
```

`BODY_TURN_THRESHOLD` is 120 pixels. The dead zone prevents oscillation when the owner is roughly centered.

### Distance Estimation

No depth sensor needed. The bounding box area ratio serves as a distance proxy:

```python
area_ratio = (bbox_width * bbox_height) / (frame_width * frame_height)
```

| area_ratio | Approximate Distance | Dog Action |
|------------|---------------------|------------|
| < 0.02 | > 15 feet | Ignore (too far / detection noise) |
| 0.02 - 0.12 | 5-15 feet | Forward at behavior speed |
| 0.12 - 0.40 | 3-5 feet | Forward, slowing |
| > 0.40 | < 3 feet | **Arrived** — stop, bark, sit |

The arrival threshold (`0.40`) comes from the behavior mode. GENTLE mode uses `0.30` (stops farther away), PLAYFUL uses `0.45` (gets closer).

### Head Tracking

During FOLLOW, `head_mode="local"`. The Mac sends the raw owner bounding box in the command JSON. The Pi's `LocalHeadTracker` computes head servo angles locally:

- **Yaw:** Horizontal offset from frame center, scaled to +-45 degrees.
- **Pitch:** Area ratio compared to target ratio — far away = pitch up slightly, close = pitch down.
- **Smoothing:** Exponential filter (alpha=0.7), converges in ~250ms at 20 Hz.
- **Dead zone:** 10 degrees — prevents micro-jitter from detection noise.

This eliminates a ~50ms network round-trip compared to the Mac computing and sending angles.

## SEARCH State

Owner disappeared less than 5 seconds ago. The dog stands still and sweeps its head looking for the owner.

### Head Sweep

```python
elapsed = now - search_start_time
head_yaw = 35.0 * sin(2 * pi * elapsed / 4.0)  # +-35 degrees, 4s period
head_pitch = EXPLORE_HEAD_PITCH                   # -10 degrees (looking down)
```

- **Body:** Stopped.
- **Head mode:** `remote` — Mac controls angles directly.
- **Head pitch:** -10 degrees (down toward floor). During FOLLOW the head was pitched up to see faces; SEARCH pitches down to scan the room for the owner at floor level.
- **VLM:** Continues situation queries. If the owner walks back into frame, the fast path (YOLO + ArcFace) transitions immediately to FOLLOW.

## EXPLORE State

Owner gone for 5-30 seconds. The dog navigates the room using VLM-guided decisions from the ribbon camera.

### Explore Loop

```
┌──▶ Send "thinking" command (purple RGB, tail wag, head oscillation)
│         │
│         ▼
│    VLM receives ribbon cam frame
│    VLM returns direction: FORWARD / LEFT / RIGHT / BACK
│         │
│         ▼
│    Execute move for ~2 seconds
│         │
│         ▼
│    Check: owner found? ──yes──▶ FOLLOW
│         │ no
└─────────┘
```

### Direction Mapping

| VLM Direction | Motor Command | Duration | Description |
|--------------|---------------|----------|-------------|
| FORWARD | `forward` speed=70, step_count=3 | 2s | Open space or doorway ahead |
| LEFT | `turn_left` speed=80, step_count=4 | 2s | Opening to the left |
| RIGHT | `turn_right` speed=80, step_count=4 | 2s | Opening to the right |
| BACK | `turn_left` speed=90, step_count=8 | 2s | Dead end — 180-degree turn |

### VLM Query Routing

When the state machine enters EXPLORE, it calls `pipeline.set_vlm_query_type("explore")`. The VLM worker thread switches from situation queries (webcam) to explore queries (ribbon camera). The ribbon camera is used because:

1. The webcam is mounted on the head servo, which pitches up during FOLLOW to see faces.
2. During EXPLORE, the head pitches down (-10 degrees) but still doesn't see straight ahead.
3. The ribbon camera is fixed on the nose, level with the floor — ideal for seeing doorways, hallways, and obstacles.

If the ribbon camera is unavailable (`USE_RIBBON_CAM=False` or the stream isn't running), the pipeline falls back to the webcam.

### Thinking Animation

While waiting for the VLM response (~245ms per query), the dog plays a "thinking" animation:

- Purple RGB breathing pattern (bps=1.5)
- Tail wag (3 steps, slow)
- Gentle head oscillation: `yaw = 15 * sin(2 * pi * t / 2.5)`
- Head pitch stays at -10 degrees

This makes the dog look alive rather than frozen between VLM queries.

## IDLE State

Owner gone for 35+ seconds (SEARCH_TIMEOUT + EXPLORE_TIMEOUT). The dog gives up searching and rests.

- **Body:** Stopped.
- **Head:** Centered (yaw=0, default pitch).
- **RGB:** Blue breathing (idle pattern).
- **VLM:** Continues situation queries (in case someone walks by).

## Behavior Modes (VLM-Driven)

Every 2.5 seconds during FOLLOW, the VLM classifies the scene and returns a behavior mode. The mode modifies follow parameters:

| Mode | Speed | Arrival Ratio | Bark | Volume | Idle Pose | When |
|------|-------|--------------|------|--------|-----------|------|
| ACTIVE | 98 | 0.40 | Yes | 80 | stand | Owner standing, walking, outdoors |
| GENTLE | 60 | 0.30 | Yes | 40 | sit | Owner sitting on couch, crouching |
| CALM | 50 | 0.25 | No | 0 | lie | Owner at desk, eating, resting |
| PLAYFUL | 98 | 0.45 | Yes | 80 | stand | Children present, energetic activity |
| SOCIAL | 70 | 0.35 | No | 0 | stand | Multiple people, owner talking |

### Hysteresis

Mode changes use **2-consecutive-same** hysteresis. The VLM must return the same mode twice in a row before the dog switches:

```python
if new_mode == pending_mode:
    pending_mode_count += 1
else:
    pending_mode = new_mode
    pending_mode_count = 1

if pending_mode_count >= 2:
    behavior_mode = new_mode   # Commit the switch
```

At 2.5-second query intervals, the fastest possible mode switch is ~5 seconds. This prevents the dog from flickering between ACTIVE and GENTLE when the VLM is uncertain.

## Transition Diagram with VLM Query Routing

| From State | To State | VLM Query Type |
|-----------|----------|---------------|
| Any | FOLLOW | `situation` (webcam, every 2.5s) |
| Any | SEARCH | `situation` (webcam, every 2.5s) |
| Any | EXPLORE | `explore` (ribbon cam, as fast as VLM responds) |
| Any | IDLE | `situation` (webcam, every 2.5s) |

The `pipeline.set_vlm_query_type()` call happens inside `_transition()`. The VLM worker thread checks the current query type each cycle and routes to the appropriate VLM function (`situation_query` or `explore_query`).
