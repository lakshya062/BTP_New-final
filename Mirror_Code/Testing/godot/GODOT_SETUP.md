# Godot Setup (Edge-Ready Path)

Use Godot 4.x. This is a good renderer choice for your edge plan because it runs on Linux ARM and supports OpenGL/Vulkan paths depending on board driver support.

## 1) Import your avatar

1. Create/open a Godot 4 project.
2. Copy `human_body_autorig.glb` into project assets.
3. Drag the GLB into a scene so it instantiates nodes.
4. Find the `Skeleton3D` node in the imported hierarchy.

## 2) Attach the live receiver

1. Copy this script into your Godot project:
   - `Mirror_Code/Testing/godot/AvatarPoseReceiver.gd`
2. Attach it to the avatar `Skeleton3D` node.
3. Keep default UDP port `7000` unless you change sender args.

## 3) Start live stream from Testing

From host machine:

```bash
cd "/Users/lakshyasharan/Desktop/Transferred/Transfer/Gym Startup/Modular_code/Modular_final/FINAL FINAL_BTP/BTP_New-final/Mirror_Code/Testing"
./run_avatar_bridge.sh human_body_autorig.retarget.json
```

## 4) Play the Godot scene

1. Press Play in Godot.
2. You should see runtime bone updates from UDP packets.

## 5) Edge deployment note

For Orange Pi 5:

1. Run the same sender script on edge with `rtmpose_env`.
2. Run a Godot export on the same edge and keep UDP on localhost (`127.0.0.1`) to avoid network jitter.

