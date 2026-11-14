# How r4r works

r4r turns a robot dataset (videos, joint states, robot description) into camera poses and
intrinsics for every frame, in the robot's base frame and in metres, and optionally into
dense depth, 3D tracks and point clouds that line up with the robot model.

Legend: ✅ exists and is tested on several datasets · 🔶 exists as a prototype in `research/`
· ⬜ planned.

## The one idea: every unknown is a parameter with a scope and a range

Nothing about a dataset is hard-coded in the library. Everything a dataset can get wrong or
leave out is a parameter of the robot description with:

- a **value** (best known so far) and a **range** (how far fitting may move it);
- a **scope**: how often it can change —
  - *dataset*: one value for all episodes (a fixed lens's focal length, a wrist camera bolted
    to the hand, a joint offset);
  - *episode*: one value per episode, inside the dataset-wide range (a static camera re-placed
    between episodes, an arm base moved, a simulator randomising mounts);
  - *frame*: one value per frame (a mobile base driving, a humanoid walking).

| unknown | typical scope | status |
|---|---|---|
| joint mapping, signs, units, video/state offset | dataset | ✅ priors (`joints`, `state_shift`), units in the loader |
| focal length | dataset (episode for randomising sims) | ✅ prior + fit; ⬜ preprocessing tool |
| moving-camera mount (link, pose) | dataset | ✅ search + bundle + episode consensus |
| static-camera pose | dataset or episode | ✅ PnP on moving-camera points + bundle (`r4r rig`) |
| arm-base placement (bimanual) | dataset or episode | ✅ free bodies, staged (`r4r rig`), written back to the priors |
| robot base pose | frame | ✅ free body at frame scope, started from recorded odometry (`base_odometry`), wired through every `r4r rig` stage (mounts on parked spans); ⬜ visual odometry when none, end-to-end validation |
| lens distortion | dataset | ⬜ (fisheye: later, with correction tables) |

Values and ranges live in `priors/<family>/<dataset>.json`; `r4r export` writes them into the
robot description (MJCF) with each camera placed and each range as an `r4r:` numeric, so the
description itself says what is known and what still varies. One solver (the kinematic
bundle) fits whatever is free, at whatever scope.

## Solving coupled unknowns in stages

Two unknowns on the same kinematic chain can trade off: a wrist camera's mount and the base of
its arm, or a static camera and a free body, can absorb each other's error and still fit the
pixels. The solver refuses to free a camera mount together with a body above it, and setup
calibration solves in stages:

1. Each moving camera's mount from its own video. It does not depend on any body upstream: a
   base offset moves the camera's whole trajectory rigidly and the static scene with it.
   Frozen afterwards.
2. Unknowns that only exist relative to others (an arm base, a static camera) jointly, over
   many episodes, from multi-camera tracks (KLT tracks merged through RoMa links between
   cameras), with the stage-1 mounts fixed.
3. The same solve on separate groups of episodes: their agreement is the check, their spread
   the uncertainty. Residuals alone do not reveal a poorly determined unknown.

## Stage 1 — dataset preparation (once per dataset, a person or an agent with tools)

The step-by-step guide is `docs/dataset-onboarding.md` (kept current). Settings (frames,
episodes, RoMa pairs) are tuned there on a few episodes, then frozen for mass processing, which
must be fast because it runs on every episode.


Goal: a priors file good enough that the rest runs unattended.

1. `r4r inspect`: joint mapping (state columns that vary but drive nothing, joints beyond
   their limits, units), body tree and how far each body moves. ✅
2. `r4r sheet` (frames next to the robot skeleton) and `r4r links` (motion evidence for which
   body a camera rides on, static vs moving): decide each camera's link. ✅
3. `r4r fit --shifts …`: the video/joint-state offset, if any. ✅
4. Focal lengths: from the dataset's intrinsics, the camera model, or estimated (ARROW pooled
   over many frames for static cameras, the motion fit for moving ones). Fixed by default. ⬜ tool
5. Declare the unknowns that remain, with ranges and scopes (above).

## Stage 2 — setup calibration (once per setup: episodes sharing their constants)

Goal: the dataset-scope values, and the range of the episode-scope ones.

1. **Pick data.** 20–50 episodes spread over the dataset; inside each, the windows that carry
   information: many surviving tracks, enough light and texture, motion with parallax, the
   static scene in view rather than the gripper. ⬜ (today: whole episodes)
2. **Per-episode fits.** Moving cameras: KLT tracks on every frame (CPU, ~2 s per camera per
   minute of video), a batched global search over candidate bodies × 24 orientations
   (× focal factors if the focal is unknown), then the bundle. ✅
3. **Consensus.** The value most episodes agree with; disagreeing episodes are set aside
   (bad views, wrong local minimum). ✅
4. **Pooled fit** over the agreeing episodes, refining the shared values; the spread across
   episodes becomes the range of episode-scope values. ✅
5. **Write** it to the priors (`r4r fit --write`) and the description (`r4r export`). ✅
6. **Validate** on held-out episodes (residuals, agreement). ⬜ report command

Static cameras in a setup: anchored to a calibrated moving camera — RoMa v2 matches between the
two views (on frame pairs where it reports overlap) give pixels of static scene points the
moving camera triangulates — then PnP and the bundle (`r4r.rago.stages`, `r4r rig`). ✅ The focal of a static camera is weakly
constrained by one episode (focal ↔ distance): it comes from preparation and is shared across
episodes. ⬜

## Stage 3 — per-episode processing (every episode, batched)

Goal: camera poses for every frame of every episode.

- Everything dataset-scope is known: camera pose = forward kinematics × mount. No images are
  needed at all; this is bounded by reading the joint states. ✅
- Episode-scope unknowns (re-placed static camera, moved arm base, randomised mount) are
  fitted within their ranges on a few sampled frames (KLT tracks for moving cameras, a few
  dozen RoMa pairs for static ones), all other values frozen. Many episodes can share one
  batched solve (each its own copy, as the search already does). ⬜
- Frame-scope unknowns (mobile base): the base pose per frame, initialised from odometry or
  integrated base commands when the dataset has them, constrained by every on-robot camera
  seeing the static world (the cameras form a rig whose geometry comes from the joints, which
  also fixes the scale), with a smoothness prior. ⬜
- Quality per episode: reprojection residuals, inlier share, agreement with the setup. ⬜

## Stage 4 — dense reconstruction (optional)

Goal: depth for every pixel of every frame, 3D tracks, a fused point cloud, all metric and in
the robot frame.

- ARROW encodes a set of frames (all cameras, long episodes in several encodings) and answers
  per-pixel depth and track queries. It cannot take our camera poses as input (pose
  conditioning was tested and copies the draft instead of correcting it), so each encoding's
  output is aligned to our solved cameras afterwards: one scale and pose per encoding, fitted
  against our cameras and sparse triangulated points. The cameras stay ours; ARROW supplies
  the dense shape. ⬜
- Compute: measured on one episode earlier, ~15–20 GPU-minutes at depth stride 2 (1–1.5 GPU-h at
  stride 1). Per hour of video that is in the 10–20 GPU-h range at stride 2: too much for every
  episode of every dataset. Options: depth at keyframes (1–2 fps) with tracks in between, lower
  resolution, or a lighter per-frame depth model scaled with our sparse points. Encoding is
  batched over frames; episodes are independent, so it parallelises across GPUs. To be measured
  properly before choosing. ⬜

## Which tool is used where

| tool | used for | on what |
|---|---|---|
| KLT (OpenCV) | moving-camera tracks | every frame of the sampled episodes (cheap, CPU) |
| batched search + bundle | mounts, focal, any free parameter | tracks of sampled windows; batched copies |
| RoMa v2 (`r4r.matching`) | static↔moving and static↔static matches | a few dozen frame pairs per setup or episode, kept where it reports overlap |
| ARROW | focal prior for static cameras; robot-attached tracks; dense depth and tracks | sampled frames (preparation), all frames (dense pass) |

Bad frames (zoomed-in wrist, no texture, dark, motion blur) are not detected by hand-written
rules on the image: they show up as few surviving KLT tracks, low RoMa certainty, high
residuals or disagreement with the other episodes, and are left out by window selection and
consensus.
