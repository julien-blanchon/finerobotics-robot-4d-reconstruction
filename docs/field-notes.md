# Field notes: from a dataset to cameras, kinematics and 3D

Raw tips and tricks collected while onboarding datasets (by people and agents), to be
consolidated later into `docs/dataset-onboarding.md`. Add to it whenever something non-obvious
is learnt: what happened, how it showed, what fixed it, which dataset. Per-embodiment details
stay in `priors/onboarding/<embodiment>.md` (section "Tips and tricks"); general lessons here.

## Checking results

- **Look at the overlay, always.** Residuals and group agreement are not enough. Seen:
  - sim-panda's static camera was 145 cm / 68° off at 0.83 px;
  - DROID's exteriors were placed below the table at 0.86 px;
  - InternData-M1's static cameras were 11.5 cm / 6.5° off at 0.93 px;
  - an RB-Y1 wrist mount was 25° off at 0.7 px.

  A static camera's own tracks fit any pose (all its rays leave one centre); only links to the
  moving cameras and the robot pin it.
- **Draw surfaces, not just axes.**
  - A URDF with visual meshes only drew nothing until visual meshes were kept for drawing (abc).
  - COLLADA visuals can't be read: fall back to collision meshes per link (HSR, Panda, RB-Y1).
  - Axes alone hid a wrong abc calibration.
- **Read the overlay by failure pattern:**
  - right place but lagging the motion: timing;
  - right at the base, wrong along the chain in every camera: kinematics;
  - wrong in one camera only: that camera.
- **Ground truth where it exists.** Simulators that ship camera poses (sim-panda, maniskill,
  molmobot, InternData) measure real error: always compare before trusting a method.

## Getting the inputs right (before any solve)

- **Recorded calibration first.** Look in the delivered metadata, then the upstream source:
  - abc: `foxglove.CameraCalibration` in each upstream `episode.mcap`;
  - InternData-M1: upstream annotations, exact to 0.000 px;
  - DROID: ZED factory intrinsics and DROID's own per-episode extrinsics;
  - molmobot: per-frame extrinsics and render intrinsics.
- **Check the calibration's resolution against the video's.** Crop vs resize differ:
  - abc: 640x480 is a centre crop of 848x480;
  - molmobot: 480x480 render shown as 512x288;
  - furniture-bench: 1280x720 squashed to 224x224, so non-square pixels;
  - robomind2: letterbox padding.
- **Joint mapping is often wrong in the metadata.** community-v3 maps the description's
  movable joints in file order, giving wrong joints and indices past the state width. Read the
  state `names`; check units (degrees, normalised percent, LeRobot-calibrated degrees with
  unknown offsets).
- **The description can be wrong:**
  - the wrong gripper (DROID and M1 have a Robotiq, not a Franka hand);
  - the wrong arm (Trossen AI data on a ViperX description);
  - single-arm descriptions for bimanual data;
  - meshes in unreadable formats.

  Fix it in a corrected copy under `priors/descriptions/<embodiment>/`, never in the delivery.
- **Gripper attachment angle:** check it against a view where the wrist mount is known. The
  Robotiq on the Panda flange was +90°, not the +45° chosen by eye at first.
- **Simulators randomise:** mounts per episode (molmobot: 1–5 cm, 4–10°), focals, robot base
  placement (InternData-M1: ±5 cm). Use episode scope and the recorded values when they exist;
  images alone don't pin a per-episode mount.
- **States may be commanded, not measured** (xrobotwin). Or absent: many Open X-Embodiment
  sets record only the end-effector pose. With no joint states there is no FK, so blocked.

## Timing

- **Measure the video/joint lag on every dataset** (`r4r fit --shifts`): abc's states lag
  ~2.3 frames, the main cause of its wrist-mount scatter. Confirm on a second station or
  setup; it can differ. Then free the sub-frame part (`kinematics.time_offset_frames`).
- **Solve joint offsets and the time offset together:** with the time offset held, joint
  offsets absorb the timing error and come out wrong.

## Cameras

- **Place cameras by eye first** (`r4r preview`), then calibrate: the solver refines good
  priors, it doesn't rescue bad ones.
- **Static camera starts:**
  - Don't start a dataset-scope camera from one episode's PnP: take the consensus over
    episodes (sim-panda episode 0 had few links and was 1.5 m off).
  - Start from the recorded or prior pose when there is one.
- **Head cameras on pan/tilt heads:** the mount search needs head motion while the base is
  parked. HSR's head barely moves then, so its mount comes from the description's optical
  frame.
- **Wrist mounts:**
  - From one camera's video, per-episode fits scatter: pool episodes, and fix timing first.
  - The focal is rarely the cause of failures: the true focal rescues only 2 of 12 blind
    failures. Unmodelled kinematics and orientation local minima are.
- **Fisheye:** declare `lens: fisheye`; recorded coefficients first (OpenCV/ROS/RealSense
  give Kannala-Brandt directly); a static fisheye needs a known focal (focal, distortion and
  distance trade off).

## Bimanual and mobile

- **Bimanual:**
  - Keep the first arm as the anchor and free the second arm's base.
  - It needs ~20 episodes to agree to millimetres (abc: 10 episodes 1.5 cm, 20 episodes 0.3 cm).
  - Arm placement differs per setup or station.
- **Mobile bases:**
  - Base motion comes from the recorded state/actions, never vision.
  - First check whether the base moves within episodes at all.
  - Validate odometry against an on-robot camera's rotation (HSR: commanded yaw useless,
    wheel encoders good).
  - molmobot records the base pose exactly.
- **Stations:** a dataset can mix physical setups (abc: 19 stations by camera calibration
  signature). Calibrate per station.

## Compute

- A solve with few unknowns and many observations: Levenberg–Marquardt, not Adam. Keep Adam
  where many independent starts are solved at once (mount search).
- Re-attaching tracks to links changes the objective: converge with the assignment fixed,
  then re-attach, in rounds.
- Track each video once and share it across stages; match cameras at a few dozen moments per
  episode, not every few frames.
- Shared login GPUs fill up without warning (other users take 96 GB): queue through a slot
  runner that falls back to the second node, and retry on out-of-memory at start.

## Tools and models

- **Dedicated estimators beat 4D models for one-shot focal guesses less than expected:**
  ARROW gen2 with a self-draft gave a 3% median focal error; MoGe-2 and GeoCalib were much
  worse on robot footage.
- **Robot meshes:** only for checking (overlays), never inside the solve. Meshes are often
  wrong or incomplete.
