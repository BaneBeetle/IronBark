# ROS2 Integration: IronBark Follow-Me Pipeline

> **Status:** Future work. IronBark currently uses raw ZeroMQ for all inter-node communication. Nothing in this document exists in the codebase yet. This is a design reference and implementation roadmap for bridging IronBark into a ROS2 ecosystem.

## What You Would Need

### Hardware (no changes)
The existing IronBark hardware works as-is. No new sensors or compute nodes required.

| Component | Already Have | Notes |
|-----------|-------------|-------|
| Raspberry Pi 5 | Yes | Runs pi_sender + motor_controller unchanged |
| USB webcam (head-mounted) | Yes | Streams on ZMQ :50505 |
| OV5647 ribbon cam (nose) | Yes | Streams on ZMQ :50506 |
| Mac with Apple Silicon | Yes | Runs perception + ROS2 bridge nodes |

### Software to Install

| Software | Version | Purpose | Install |
|----------|---------|---------|---------|
| **ROS2 Humble** | Humble Hawksbill (LTS) | Core framework, message passing, tooling | [docs.ros.org install guide](https://docs.ros.org/en/humble/Installation.html) |
| **colcon** | Latest | ROS2 build tool | `sudo apt install python3-colcon-common-extensions` |
| **cv_bridge** | Humble | Convert OpenCV frames to `sensor_msgs/Image` | `sudo apt install ros-humble-cv-bridge` |
| **rosbag2** | Humble | Session recording and replay | Ships with `ros-humble-desktop` |
| **Nav2** | Humble | Path planning and autonomous navigation (optional) | `sudo apt install ros-humble-navigation2` |
| **RViz2** | Humble | 3D visualization of topics, TF, markers | Ships with `ros-humble-desktop` |
| **tf2_ros** | Humble | Transform tree (camera frames relative to body) | Ships with `ros-humble-desktop` |
| **pyzmq** | 25+ | ZMQ bindings (already installed for IronBark) | `pip install pyzmq` |

### Code to Write

| Component | Estimated Effort | Description |
|-----------|-----------------|-------------|
| `video_bridge_node.py` | ~100 lines | ZMQ PULL -> `sensor_msgs/Image` publisher for both cameras |
| `telemetry_bridge_node.py` | ~60 lines | ZMQ SUB -> custom `DogTelemetry` publisher |
| `cmd_bridge_node.py` | ~80 lines | `geometry_msgs/Twist` subscriber -> ZMQ PUB motor commands |
| `perception_bridge_node.py` | ~150 lines | Wrap `PerceptionPipeline` output as custom `PerceptionResult` messages |
| Custom `.msg` definitions | 4 files | `PersonDetection`, `PerceptionResult`, `BehaviorState`, `DogTelemetry` |
| Launch file | ~30 lines | Parameterized launch for all bridge nodes |
| Static TF publisher config | ~10 lines | Camera frame positions relative to `base_link` |
| Dynamic TF broadcaster | ~50 lines | Webcam frame that updates with head servo angles |

**Total new code:** ~500-600 lines of Python + 4 message definitions.

### Architectural Decision: Bridge vs Native Port

| Approach | Pros | Cons |
|----------|------|------|
| **Bridge (recommended)** | Zero changes to working IronBark code; incremental; can run with or without ROS2 | Extra serialization hop; ZMQ port conflicts with follower.py |
| **Native ROS2 port** | Clean ROS2-native architecture; no ZMQ dependency | Rewrite all inter-node communication; lose working system during port; months of work |

The bridge approach is recommended. It preserves the working system and adds ROS2 as a parallel interface.

### What This Unlocks

| Capability | Without ROS2 | With ROS2 Bridge |
|-----------|-------------|-----------------|
| Visualization | OpenCV window only | RViz2 with 3D markers, TF tree, multi-topic |
| Session recording | Manual log parsing | `rosbag2 record` / `rosbag2 play` for full replay |
| Path planning | VLM-only reactive explore | Nav2 costmap + global/local planners |
| SLAM | None | RTAB-Map, Cartographer, or ORB-SLAM3 with camera feed |
| Obstacle mapping | Ultrasonic only (1D, 15cm) | 2D/3D costmap from camera depth estimation |
| Interoperability | Custom ZMQ protocol | Standard ROS2 topics any node can subscribe to |

---

## Overview

This integration wraps IronBark's vision data (YOLO detections, face recognition, VLM behavior modes, and motor commands) as ROS2 topics. This lets you:

- Visualize detections and behavior state in RViz2
- Feed IronBark's perception into Nav2 for path planning
- Log sessions with `rosbag2` for replay and analysis
- Combine IronBark's owner tracking with SLAM or obstacle mapping

The approach is a **bridge pattern** — ROS2 nodes subscribe to IronBark's ZMQ sockets and republish as standard ROS2 messages. The existing ZMQ pipeline continues to run unmodified.

## Prerequisites

- ROS2 Humble or Iron (Ubuntu 22.04 or 24.04)
- IronBark codebase running (see [quickstart-setup.md](quickstart-setup.md))
- Both ZMQ streams active (webcam on port 50505, ribbon cam on port 50506)
- `motor_controller.py` running on the Pi

```bash
# ROS2 dependencies
sudo apt install ros-humble-desktop python3-colcon-common-extensions
pip install pyzmq numpy opencv-python
```

## Package Structure

```
ironbark_ros/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/
│   └── ironbark_ros
├── ironbark_ros/
│   ├── __init__.py
│   ├── video_bridge_node.py      # ZMQ video -> sensor_msgs/Image
│   ├── perception_bridge_node.py # Perception results -> custom msgs
│   ├── cmd_bridge_node.py        # geometry_msgs/Twist -> ZMQ commands
│   └── telemetry_bridge_node.py  # ZMQ telemetry -> custom msgs
├── msg/
│   ├── PersonDetection.msg
│   ├── PerceptionResult.msg
│   ├── BehaviorState.msg
│   └── DogTelemetry.msg
└── launch/
    └── ironbark_bridge.launch.py
```

## Custom Message Definitions

### PersonDetection.msg

```
# One detected person with face recognition result
int32 x1
int32 y1
int32 x2
int32 y2
float32 confidence          # YOLO detection confidence
float32 face_similarity     # ArcFace cosine similarity (0 if no face)
bool is_owner               # True if face_similarity >= threshold
int32 center_x
int32 center_y
int32 area
```

### PerceptionResult.msg

```
# Full perception output for one frame
std_msgs/Header header
ironbark_ros/PersonDetection[] detections
string behavior_mode        # ACTIVE, GENTLE, CALM, PLAYFUL, SOCIAL
string behavior_description # VLM scene description
string explore_direction    # FORWARD, LEFT, RIGHT, BACK (empty if not exploring)
string explore_reasoning    # VLM reasoning for direction
string follower_state       # IDLE, FOLLOW, SEARCH, EXPLORE
float32 fast_path_latency_ms
int32 frame_id
```

### BehaviorState.msg

```
# Current behavior mode and follow-me state
std_msgs/Header header
string follower_state       # IDLE, FOLLOW, SEARCH, EXPLORE
string behavior_mode        # ACTIVE, GENTLE, CALM, PLAYFUL, SOCIAL
float32 area_ratio          # Owner bbox area ratio (distance proxy)
float32 ultrasonic_cm       # Ultrasonic distance reading
bool danger                 # True if ultrasonic < 15cm
```

### DogTelemetry.msg

```
# Pi-side telemetry
std_msgs/Header header
float32 distance_cm
float32 battery_v
string current_action       # forward, stop, turn_left, etc.
bool danger
string command_source       # follow, teleop
```

## Published Topics

| Topic | Type | Rate | Source | Description |
|-------|------|------|--------|-------------|
| `/ironbark/camera/webcam/image_raw` | `sensor_msgs/Image` | 30 Hz | ZMQ :50505 | Head-mounted webcam (BGR8) |
| `/ironbark/camera/ribbon/image_raw` | `sensor_msgs/Image` | 30 Hz | ZMQ :50506 | Nose-mounted ribbon cam (BGR8) |
| `/ironbark/perception` | `PerceptionResult` | 10 Hz | Perception pipeline | Detections, face matches, VLM results |
| `/ironbark/behavior_state` | `BehaviorState` | 10 Hz | Follower FSM | Current state, mode, distance proxy |
| `/ironbark/telemetry` | `DogTelemetry` | 5 Hz | ZMQ :5558 | Ultrasonic, battery, motor state |
| `/ironbark/detections/markers` | `visualization_msgs/MarkerArray` | 10 Hz | Bridge | RViz2 bounding box markers |

## Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/ironbark/cmd_vel` | `geometry_msgs/Twist` | External velocity commands (e.g., from Nav2). Bridged to ZMQ :5556. |

## Bridge Nodes

### Video Bridge Node

Receives JPEG frames from ZMQ PULL sockets and publishes as `sensor_msgs/Image` for RViz2 and downstream ROS2 nodes.

```python
#!/usr/bin/env python3
"""ironbark_ros/video_bridge_node.py — ZMQ video stream -> ROS2 Image topics."""

import struct
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import zmq
import cv2
import numpy as np


class VideoBridgeNode(Node):
    def __init__(self):
        super().__init__('ironbark_video_bridge')

        self.declare_parameter('zmq_webcam_port', 50505)
        self.declare_parameter('zmq_ribbon_port', 50506)
        self.declare_parameter('zmq_bind_ip', '0.0.0.0')

        webcam_port = self.get_parameter('zmq_webcam_port').value
        ribbon_port = self.get_parameter('zmq_ribbon_port').value
        bind_ip = self.get_parameter('zmq_bind_ip').value

        self.bridge = CvBridge()

        # ZMQ PULL sockets (same config as follower.py)
        self.ctx = zmq.Context()

        self.webcam_sock = self.ctx.socket(zmq.PULL)
        self.webcam_sock.setsockopt(zmq.CONFLATE, 1)
        self.webcam_sock.setsockopt(zmq.RCVTIMEO, 100)
        self.webcam_sock.bind(f'tcp://{bind_ip}:{webcam_port}')

        self.ribbon_sock = self.ctx.socket(zmq.PULL)
        self.ribbon_sock.setsockopt(zmq.CONFLATE, 1)
        self.ribbon_sock.setsockopt(zmq.RCVTIMEO, 100)
        self.ribbon_sock.bind(f'tcp://{bind_ip}:{ribbon_port}')

        # Publishers
        self.webcam_pub = self.create_publisher(Image, '/ironbark/camera/webcam/image_raw', 10)
        self.ribbon_pub = self.create_publisher(Image, '/ironbark/camera/ribbon/image_raw', 10)

        # 30 Hz timer to match camera frame rate
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info(
            f'Video bridge: webcam={webcam_port}, ribbon={ribbon_port}')

    def decode_zmq_frame(self, raw: bytes):
        """Decode IronBark's binary frame format: [timestamp_us | jpeg_len | jpeg]."""
        header_size = struct.calcsize('<qI')
        if len(raw) < header_size:
            return None, 0
        timestamp_us, payload_len = struct.unpack_from('<qI', raw, 0)
        jpeg_data = raw[header_size:header_size + payload_len]
        frame = cv2.imdecode(
            np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame, timestamp_us

    def publish_frame(self, sock, publisher, frame_id):
        """Try to receive a ZMQ frame and publish as ROS2 Image."""
        try:
            raw = sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            return
        frame, timestamp_us = self.decode_zmq_frame(raw)
        if frame is None:
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        publisher.publish(msg)

    def timer_callback(self):
        self.publish_frame(self.webcam_sock, self.webcam_pub, 'webcam_link')
        self.publish_frame(self.ribbon_sock, self.ribbon_pub, 'ribbon_cam_link')

    def destroy_node(self):
        self.webcam_sock.close()
        self.ribbon_sock.close()
        self.ctx.term()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

> **Warning:** The video bridge and `follower.py` both bind ZMQ PULL sockets. You cannot run both simultaneously on the same ports. Either run the ROS2 bridge *instead of* follower.py, or modify follower.py to publish to a ROS2 topic rather than binding ZMQ directly.

### Telemetry Bridge Node

Subscribes to the Pi's telemetry PUB socket and republishes as a ROS2 topic.

```python
#!/usr/bin/env python3
"""ironbark_ros/telemetry_bridge_node.py — ZMQ telemetry -> ROS2 DogTelemetry."""

import json
import rclpy
from rclpy.node import Node
from ironbark_ros.msg import DogTelemetry
import zmq


class TelemetryBridgeNode(Node):
    def __init__(self):
        super().__init__('ironbark_telemetry_bridge')

        self.declare_parameter('pi_ip', '100.0.0.2')  # Replace with your Pi's Tailscale IP
        self.declare_parameter('telemetry_port', 5558)

        pi_ip = self.get_parameter('pi_ip').value
        port = self.get_parameter('telemetry_port').value

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.RCVTIMEO, 200)
        self.sock.connect(f'tcp://{pi_ip}:{port}')
        self.sock.subscribe(b'')

        self.pub = self.create_publisher(DogTelemetry, '/ironbark/telemetry', 10)
        self.timer = self.create_timer(0.2, self.timer_callback)  # 5 Hz
        self.get_logger().info(f'Telemetry bridge: {pi_ip}:{port}')

    def timer_callback(self):
        try:
            raw = self.sock.recv_string(zmq.NOBLOCK)
            data = json.loads(raw)
        except zmq.Again:
            return

        msg = DogTelemetry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.distance_cm = float(data.get('distance_cm', -1))
        msg.battery_v = float(data.get('battery_v', 0.0))
        msg.current_action = data.get('action', 'unknown')
        msg.danger = data.get('danger', False)
        msg.command_source = data.get('source', '')
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Command Velocity Bridge Node

Translates standard `geometry_msgs/Twist` messages (e.g., from Nav2 or teleop_twist_keyboard) into IronBark's ZMQ command format.

```python
#!/usr/bin/env python3
"""ironbark_ros/cmd_bridge_node.py — ROS2 Twist -> ZMQ follow-me commands."""

import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import zmq


class CmdBridgeNode(Node):
    def __init__(self):
        super().__init__('ironbark_cmd_bridge')

        self.declare_parameter('cmd_port', 5556)

        port = self.get_parameter('cmd_port').value
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f'tcp://*:{port}')

        self.sub = self.create_subscription(
            Twist, '/ironbark/cmd_vel', self.twist_callback, 10)
        self.get_logger().info(f'Cmd bridge: publishing on ZMQ :{port}')

    def twist_callback(self, msg: Twist):
        """Map Twist linear.x/angular.z to IronBark motor commands."""
        linear = msg.linear.x
        angular = msg.angular.z

        # Map to IronBark's discrete action space
        if abs(angular) > 0.3:
            action = 'turn_left' if angular > 0 else 'turn_right'
            speed = min(98, int(abs(angular) * 100))
        elif linear > 0.1:
            action = 'forward'
            speed = min(98, int(linear * 100))
        elif linear < -0.1:
            action = 'backward'
            speed = min(98, int(abs(linear) * 100))
        else:
            action = 'stop'
            speed = 0

        cmd = {
            'action': action,
            'speed': speed,
            'step_count': 2,
            'head_mode': 'remote',
            'head_yaw': 0.0,
            'head_pitch': 15,
            'bark': False,
        }
        self.sock.send(json.dumps(cmd).encode('utf-8'))


def main(args=None):
    rclpy.init(args=args)
    node = CmdBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch File

```python
# launch/ironbark_bridge.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('pi_ip', default_value='100.0.0.2'),  # Your Pi's Tailscale IP

        Node(
            package='ironbark_ros',
            executable='video_bridge_node',
            name='video_bridge',
            parameters=[{
                'zmq_webcam_port': 50505,
                'zmq_ribbon_port': 50506,
            }],
        ),
        Node(
            package='ironbark_ros',
            executable='telemetry_bridge_node',
            name='telemetry_bridge',
            parameters=[{
                'pi_ip': LaunchConfiguration('pi_ip'),
                'telemetry_port': 5558,
            }],
        ),
        # Uncomment to enable external velocity control:
        # Node(
        #     package='ironbark_ros',
        #     executable='cmd_bridge_node',
        #     name='cmd_bridge',
        #     parameters=[{'cmd_port': 5556}],
        # ),
    ])
```

## Quick Test

### Build the Package

```bash
cd ~/ros2_ws/src
# Copy or symlink ironbark_ros package here
cd ~/ros2_ws
colcon build --packages-select ironbark_ros
source install/setup.bash
```

### Start the Bridge

Ensure IronBark's Pi-side processes are running (pi_sender on ports 50505/50506, motor_controller). Then:

```bash
ros2 launch ironbark_ros ironbark_bridge.launch.py pi_ip:=<your-pi-tailscale-ip>
```

### Verify Topics

```bash
ros2 topic list
# Expected:
# /ironbark/camera/webcam/image_raw
# /ironbark/camera/ribbon/image_raw
# /ironbark/telemetry

ros2 topic hz /ironbark/camera/webcam/image_raw
# Expected: ~30 Hz

ros2 topic echo /ironbark/telemetry
# Expected: distance_cm, battery_v, current_action fields updating at 5 Hz
```

### View in RViz2

```bash
rviz2
```

Add an Image display subscribed to `/ironbark/camera/webcam/image_raw`. You should see the live webcam feed from the dog's head.

## TF Frames

IronBark does not currently publish a TF tree. For Nav2 integration, you would define these frames:

```
base_link             # Dog body center (on the ground)
├── webcam_link       # Head-mounted webcam (moves with head servo)
├── ribbon_cam_link   # Nose-mounted camera (fixed relative to base)
└── ultrasonic_link   # Ultrasonic sensor (forward-facing)
```

A static transform publisher handles the fixed transforms:

```bash
ros2 run tf2_ros static_transform_publisher \
    0.05 0.0 0.03 0 0 0 base_link ribbon_cam_link

ros2 run tf2_ros static_transform_publisher \
    0.0 0.0 0.08 0 0 0 base_link ultrasonic_link
```

The `webcam_link` transform changes with head servo angles. A dynamic TF broadcaster that reads head angles from the telemetry topic would be needed for accurate webcam-to-base transforms.

## Integration Examples

### Recording a Session with rosbag2

```bash
ros2 bag record /ironbark/camera/webcam/image_raw \
                /ironbark/telemetry \
                -o ironbark_session_001
```

Replay for offline analysis:

```bash
ros2 bag play ironbark_session_001
```

### Nav2 Velocity Control

Replace IronBark's follower state machine with Nav2's navigation stack:

1. Run the cmd_bridge_node (uncomment in launch file).
2. Stop `follower.py` on the Mac (it conflicts with the cmd_bridge on port 5556).
3. Configure Nav2 to publish to `/ironbark/cmd_vel`.
4. Nav2's `Twist` messages are translated to IronBark's discrete motor commands by the bridge.

> **Note:** IronBark's motor commands are discrete (forward/turn_left/turn_right/stop), not continuous velocities. The cmd_bridge maps Twist values to the closest discrete action. Fine-grained velocity control would require PID loops on the Pi side, which IronBark does not currently implement.

### Custom Perception Subscriber

Process IronBark's camera feed with your own ROS2 node:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class MyProcessorNode(Node):
    def __init__(self):
        super().__init__('my_processor')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, '/ironbark/camera/webcam/image_raw',
            self.image_callback, 10)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # Your processing here — SLAM, obstacle mapping, etc.
        self.get_logger().info(f'Frame: {frame.shape}')
```

## Architectural Considerations

### ZMQ Port Conflicts

The video bridge binds the same ZMQ PULL ports as `follower.py`. You have two options:

1. **Bridge-only mode:** Run the ROS2 bridge instead of follower.py. Use ROS2 nodes for all perception and control.
2. **Sidecar mode:** Modify `pi_sender.py` to use ZMQ PUB/SUB instead of PUSH/PULL, allowing multiple subscribers on the same port. This requires a one-line change on the Pi side but changes the delivery semantics (PUB/SUB drops messages if subscribers are slow; PUSH/PULL queues them).

### Latency

The ZMQ-to-ROS2 bridge adds ~1-2ms of overhead per message (serialization + publish). This is negligible compared to the ~30ms YOLO inference and ~245ms VLM query.

### QoS Settings

For camera topics, use `BEST_EFFORT` reliability to avoid buffering stale frames:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
self.webcam_pub = self.create_publisher(Image, '/ironbark/camera/webcam/image_raw', qos)
```

This mirrors the `ZMQ_CONFLATE=1` behavior in the native pipeline — always process the latest frame.

## Troubleshooting

### "No messages on /ironbark/camera/webcam/image_raw"

**Cause:** ZMQ PULL socket didn't receive frames. Either pi_sender isn't running, the IP is wrong, or another process (follower.py) already bound the port.

**Solution:** Check that pi_sender is running on the Pi (`ps aux | grep pi_sender`). Verify no other process is binding port 50505 on the Mac. Kill follower.py if it's running.

### "QoS incompatible" warnings in subscriber

**Cause:** Publisher and subscriber have mismatched QoS reliability settings.

**Solution:** Match QoS profiles. If the bridge publishes with `BEST_EFFORT`, subscribers must also use `BEST_EFFORT`. RViz2 defaults to `RELIABLE` — change it in the display settings.

### Telemetry topic shows distance_cm: -1.0

**Cause:** The Pi's ultrasonic sensor isn't initialized, or motor_controller.py isn't running with `sudo`.

**Solution:** Restart motor_controller with `sudo python3 motor_controller.py`. The ultrasonic sensor subprocess requires root for GPIO access.
