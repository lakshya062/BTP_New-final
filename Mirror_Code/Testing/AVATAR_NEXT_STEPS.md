# Avatar Next Steps (Testing)

Use the local Testing virtual environment for all asset checks:

```bash
cd "/Users/lakshyasharan/Desktop/Transferred/Transfer/Gym Startup/Modular_code/Modular_final/FINAL FINAL_BTP/BTP_New-final/Mirror_Code/Testing"
./rtmpose_env/bin/python glb_rig_preflight.py --input human_body_autorig.glb --report human_body_autorig.preflight.json
./rtmpose_env/bin/python generate_retarget_config.py --input human_body_autorig.glb --output human_body_autorig.retarget.json
./rtmpose_env/bin/python retarget_smoke_test.py --config human_body_autorig.retarget.json --output human_body_autorig.retarget_smoke.json
```

Immediate live step (camera -> live bone UDP packets):

```bash
cd "/Users/lakshyasharan/Desktop/Transferred/Transfer/Gym Startup/Modular_code/Modular_final/FINAL FINAL_BTP/BTP_New-final/Mirror_Code/Testing"
./run_avatar_bridge.sh human_body_autorig.retarget.json
```

Expected outputs:

- `human_body_autorig.preflight.json`: pass/fail + rig metrics.
- `human_body_autorig.retarget.json`: rig slots and IK chain targets for runtime.
- `human_body_autorig.retarget_smoke.json`: sample quaternion output for chain sanity checks.

Recommended immediate integration order:

1. Use Godot 4 as the renderer (see `godot/GODOT_SETUP.md`).
2. Load `human_body_autorig.glb` in Godot and attach `godot/AvatarPoseReceiver.gd` to `Skeleton3D`.
3. Start `./run_avatar_bridge.sh human_body_autorig.retarget.json`.
4. Verify live arm/leg chain behavior, then add IK and constraint tuning in renderer.
