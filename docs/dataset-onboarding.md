# Onboarding a dataset

What a person or an agent does, once per dataset, before r4r processes all of its episodes.
The goal is a priors file (`priors/<family>/<dataset>.json`) and a configuration that work
unattended at scale. Everything here is tried on a handful of episodes; mass processing then
reuses the result with no further choices.

Keep this guide current: when a tool, a default or a lesson changes, update it here.
Legend: ✅ tool exists · ⬜ planned (do it by hand or with `research/` scripts until then).

## The agent works first

Agents now have good physical grounding and image understanding: the first and main step for
a dataset is the agent itself looking at the data and the simulation, and setting as priors
everything that can be set that way, before any optimisation. Concretely:
- Look at the videos and at the robot simulated at the recorded joint states:
  `r4r preview <dataset> check.png` ✅ draws the description's meshes at the recorded joints over
  every camera, with the cameras where the priors put them (no solve). Place each camera by
  eye (which link, roughly where, which way it looks), check joint signs, units and timing,
  the focal (does the drawn robot have the right size?), the arm bases of a bimanual robot,
  then re-render and iterate until the drawn robot sits close to the real one.
- Write ad-hoc scripts freely for this (state plots, simulated motions, frame grids, quick
  fits): they live in the agent's scratchpad, not in the library. One shared tool cannot fit
  every dataset; the library provides the solver and the checks.
- Record what was found in the standard format: `priors/<family>/<dataset>.json` (the values)
  and `priors/onboarding/<embodiment>.md` (the record: what was checked, how, evidence;
  template `priors/onboarding/TEMPLATE.md`).

Only then calibrate (section 4): the solver refines good priors; it does not rescue bad ones.

## 0. Is it in scope?

- Robot data with joint states and a robot description. Handheld grippers (UMI-style) and
  egocentric human video are out of scope for now.
- `r4r inspect <family>/datasets/<name>` ✅ must run. If the description does not load, see
  `priors/ISSUES.md` and the loader (meshes and inertias are repaired automatically).

## 1. Kinematics: make forward kinematics right

1. `r4r inspect … --episode 0` ✅, and a few other episodes.
   - **State columns varying but unmapped** → a joint missing from the mapping: add it to
     `joints` in the priors (index, URDF joint, sign, optional `from`/`to` range).
   - **Joints outside their limits** → wrong sign, offset or unit: fix with `sign` or a range.
   - **Units**: degrees are converted automatically; `normalised_percent` needs a range per
     joint in the priors (the loader refuses otherwise).
   - **Body tree**: check that what moves is what you expect (waist, head, both arms).
2. `r4r sheet <dataset> out.png` ✅: camera frames next to the robot skeleton. The skeleton's
   motion must match the video.
3. Video/joint-state offset: `r4r fit <dataset> <camera> --shifts -3 -2 -1 0 1 2 3` ✅. If a
   shift other than 0 wins clearly, set `state_shift`. Check two or three episodes: it may
   vary.

## 2. Cameras: what is where, and what is known

For every camera decide, from the sheet, `r4r links` ✅ and the robot description:

- **Static or moving**, and for moving cameras **the link** it is rigid on (metadata is often
  wrong; `r4r links` gives evidence: still share, rotation residual per body group).
- **Focal length**: look for a recorded calibration before estimating anything, in this order:
  1. the delivered dataset (`info.json` camera fields, per-frame intrinsics columns, `meta/`);
  2. the upstream source in `provenance.json`: raw recordings often keep it when the
     conversion dropped it (abc-130k: `foxglove.CameraCalibration` on `/…-camera-info` in each
     `episode.mcap`, read from the first MBs with `research/abc_calibration.py`; ROS bags:
     `camera_info`; also the source's format docs and calibration files);
  3. the camera model's datasheet.

  Check the calibration's resolution against the video's (abc-130k: K at 848×480, videos
  640×480, so crop vs resize must be checked). A dropped calibration goes into
  `priors/ISSUES.md`. When intrinsics differ per episode (several recording stations,
  per-unit calibrations), write them to `priors/<family>/<dataset>.episodes.parquet` (one row
  per episode and camera: `episode, camera, fx, fy, cx, cy` at the video resolution, and
  `station` when known); the loader gives each stream its K ✅. A station is a setup: `r4r rig`
  refuses episodes from several stations. Only then is it estimated
  (⬜ a focal tool: ARROW pooled over many frames for static cameras; the moving-camera fit
  gives it for wrist/head cameras). Fixed by default (`focal_range: 0`); a range only for
  datasets that really vary it (randomising simulators, per-unit calibrations).
- **Scope** of each pose: `dataset` (bolted, never moves), `episode` (re-placed between
  episodes, e.g. DROID's exterior cameras), `frame` (rare).
- **Lens** ✅: `"lens": "pinhole"` (default) or `"fisheye"`, and its `distortion`
  coefficients, looked for like the focal (recorded calibration first):
  - pinhole: OpenCV radial `[k1, k2, k3]` (from `distCoeffs` `[k1, k2, p1, p2, k3]`: drop the
    tangential `p1, p2`); missing: an ideal pinhole;
  - fisheye: OpenCV fisheye / Kannala–Brandt `[k1, k2, k3, k4]` (ROS `equidistant`,
    RealSense `kannala_brandt4`); missing: unknown, refined from the equidistant model
    (`ρ = θ`) with a wide prior. The focal is then pixels per radian at the centre (a
    pinhole-like guess, 0.7 × width, is a fine start).

  `distortion_range` is the prior std of each coefficient in the bundle: 0 holds them (the
  default when given), a value lets the fit refine them (e.g. 0.05 for a datasheet guess).
  Coefficients do not depend on the video resolution. Other models (rational, Double Sphere,
  FOV) must be converted or refitted to one of these two. `r4r rig` / `r4r fit --write`
  write fitted coefficients back (`distortion_range: 0` afterwards, like the focal).
  A first draft for a fisheye without recorded coefficients can come from AnyCalib
  (⬜, see `docs/related-work.md`).

## 3. Other unknowns

- **Bimanual**: where the second arm is relative to the first is usually a guess in the
  description. Declare it free: `"bodies": {"right_base": {"position_range": 0.2,
  "orientation_range_deg": 20, "scope": "dataset"}}` (episode scope if it moves between
  episodes). Keep the first arm as the anchor.
- **Mobile bases**: declare the base free per frame (`"bodies": {"<root body>":
  {"position_range": 0.05, "orientation_range_deg": 3, "scope": "frame"}}`, a random walk) and
  say where its motion is recorded, as its starting trajectory ✅ (`base_odometry`):
  `pose` (planar x, y, θ columns: simulators, odometry), `velocity` (body-frame vx, vy, ω), or
  `caster` (wheel angles left, right[, steering] with radius, half track and steering offset
  from the description; offset 0 and no steering for a differential drive). Validate it: the
  translation against commanded base velocities, the turn against an on-robot camera's
  rotation over frames where the camera's joints are still (`research/hsr_yaw_check.py`);
  pick encoder signs that way. Commanded turn rates can be a poor reference (HSR). `r4r rig`
  ✅ then uses the record throughout: moving-camera mounts are searched on each episode's
  longest span where the base is parked (≥ 20 timesteps), static cameras are placed with the
  base moved, and the bundle's per-frame base starts from the record. The motion comes from
  what the dataset records, not from vision. Look for it in this order, in the state, the
  actions and the feature `names`: a recorded base/root pose or odometry; a measured base
  velocity; wheel encoders; a commanded base velocity (check it against measured motion);
  for legged robots, leg kinematics with the stance foot held (joint states, contacts, IMU
  orientation; ⬜ `base_odometry` kind not yet written). First check whether the base moves
  within episodes at all: if it never does, the fixed-base pipeline applies episode by
  episode. A base that moves with nothing recorded is a dataset issue (`priors/ISSUES.md`).
- Always keep **one anchor** (a fixed arm, a known camera): with everything free, the whole
  scene can shift with no change in the images.

## 4. Calibrate the setup on a sample, and tune

Solve in stages (the solver refuses a free camera mount together with a free body above it):

1. Moving-camera mounts, each from its own video, over a sample of episodes with consensus:
   `r4r fit <dataset> <camera> --episodes … --write` ✅. Check that most episodes agree; the
   disagreeing ones are usually poor views (zoomed, dark, little texture).
2. The whole rig: `r4r rig <dataset> --episodes … --groups 2 [--write]` ✅ runs every stage
   from the priors: free moving mounts (skipped when the priors know them), multi-camera
   tracks (KLT + RoMa links every `--every` timesteps), static cameras by PnP per episode,
   then one bundle over all episodes with the free bodies and static cameras free and the
   mounts frozen.
3. The same bundle on separate groups of episodes (`--groups`): it prints how much they
   disagree per free body. Dataset-scope values must agree within a few millimetres / tenths
   of a degree. Residuals alone do not reveal a poorly determined unknown. `--write` stores
   the fitted corrections of dataset-scope bodies in the priors (`position`, `orientation`
   of the correction, and the groups' disagreement as the remaining range); the loader and
   `r4r export` apply them.

**What to tune** (on the sample, then frozen for mass processing):

| setting | effect | start with |
|---|---|---|
| frames (stride) | more frames: longer, cleaner tracks; more time and memory | every frame for KLT, subsample for the search |
| episodes per setup fit | the main lever for dataset-scope unknowns | 5–20 |
| RoMa pairs (`--every`) | links between cameras: what pins arm bases and static cameras | every 0.3–1 s |
| agreement tolerance | which episodes count as agreeing | 2 cm / 2° |
| focal range | 0 when known | 0 |

Record what you chose and why in the priors' `notes`.

## 5. Freeze and export

- `r4r fit … --write` ✅ stores mounts, focal and ranges in the priors; `r4r export
  <dataset> robot.xml` ✅ writes the description with every camera placed and each range.
- Add any metadata bug you found to `priors/ISSUES.md` (for the dataset maintainers).

## 6. Check before mass processing

- **Look at the overlay: required.** `r4r rig … --save solution.pt` ✅, then
  `r4r overlay solution.pt check.png` ✅ (a grid: a few moments of every solved episode, every
  camera) and `r4r overlay solution.pt episode.mp4 --episode 0` ✅ (a video). The robot, drawn
  from the description's meshes with the solved kinematics and cameras, must sit on the real
  robot in every camera, in every frame, on several episodes. An agent reads the PNG; a person
  watches the video. Residuals and group agreement are not enough: on sim-panda a static
  camera 145 cm off still fitted at 0.83 px (its own tracks fit any pose).
  - Robot drawn in the right place but trailing the motion: a time offset (`state_shift`,
    `r4r fit --shifts`).
  - Right place at the base, wrong further along the chain, in every camera: kinematics
    (joint mapping, sign, offset, sag, a wrong description). Fix the model: this is the
    agent's work per dataset, not the solver's.
  - Right in some cameras, wrong in one: that camera's mount or pose.
  - Meshes can be wrong or missing (simplified collision meshes, absent grippers): judge
    the skeleton and the end-effector frames as well as the surfaces.
- ⬜ `r4r report`: per-camera residuals, inlier share, agreement across episodes, held-out
  episodes, with the overlay. Today: `research/validation/` scripts and the rerun viewer
  (`research/viz_rerun.py`).

## Lessons so far

- Joint states can be commanded rather than measured (xrobotwin): FK is then off while the
  arm moves. Check the state against the video (sheet, shift scores).
- Simulators may randomise mounts and focals per episode (molmobot, interndata): declare
  episode-scope ranges instead of fighting for one value.
- A blind focal guess makes the moving-camera search fail on hard datasets (wrong orientation
  with an inflated focal): give the focal.
- One episode rarely pins down an arm base: pool episodes.
- Most failed wrist/head mount searches on simulators come from unmodelled kinematics (a
  video/joint lag, body sag, missing priors), not the focal: even the true focal rescues few
  of them. Get the kinematics right first (overlay, `--shifts`).
