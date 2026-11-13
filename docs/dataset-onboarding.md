# Onboarding a dataset

What a person or an agent does, once per dataset, before r4r processes all of its episodes.
The goal is a priors file (`priors/<family>/<dataset>.json`) and a configuration that work
unattended at scale. Everything here is tried on a handful of episodes; mass processing then
reuses the result with no further choices.

Keep this guide current: when a tool, a default or a lesson changes, update it here.
Legend: ✅ tool exists · ⬜ planned (do it by hand or with `research/` scripts until then).

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
- **Focal length**: from the dataset's intrinsics, the camera model's datasheet, or estimated
  (⬜ a focal tool: ARROW pooled over many frames for static cameras; the moving-camera fit
  gives it for wrist/head cameras). Fixed by default (`focal_range: 0`); a range only for
  datasets that really vary it (randomising simulators, per-unit calibrations).
- **Scope** of each pose: `dataset` (bolted, never moves), `episode` (re-placed between
  episodes, e.g. DROID's exterior cameras), `frame` (rare).
- **Lens**: pinhole only for now; fisheye datasets wait (⬜ distortion + correction tables).

## 3. Other unknowns

- **Bimanual**: where the second arm is relative to the first is usually a guess in the
  description. Declare it free: `"bodies": {"right_base": {"position_range": 0.2,
  "orientation_range_deg": 20, "scope": "dataset"}}` (episode scope if it moves between
  episodes). Keep the first arm as the anchor.
- **Mobile bases**: ⬜ (per-frame base pose). Check per episode whether the base moves at all
  (odometry, base commands, or the head camera's still share in `r4r links`); if it stays put
  within episodes, the fixed-base pipeline applies episode by episode.
- Always keep **one anchor** (a fixed arm, a known camera): with everything free, the whole
  scene can shift with no change in the images.

## 4. Calibrate the setup on a sample, and tune

Solve in stages (the solver refuses a free camera mount together with a free body above it):

1. Moving-camera mounts, each from its own video, over a sample of episodes with consensus:
   `r4r fit <dataset> <camera> --episodes … --write` ✅. Check that most episodes agree; the
   disagreeing ones are usually poor views (zoomed, dark, little texture).
2. Bodies and static cameras, jointly over many episodes, with the mounts frozen
   (⬜ in the CLI; `research/rig_setup.py` for bimanual rigs today).
3. Repeat on separate groups of episodes: the values must agree within a few millimetres /
   tenths of a degree for dataset-scope unknowns. Residuals alone do not reveal a poorly
   determined unknown.

**What to tune** (on the sample, then frozen for mass processing):

| setting | effect | start with |
|---|---|---|
| frames (stride) | more frames: longer, cleaner tracks; more time and memory | every frame for KLT, subsample for the search |
| episodes per setup fit | the main lever for dataset-scope unknowns | 5–20 |
| RoMa pairs (`every`) | links between cameras: what pins arm bases and static cameras | every 0.3–1 s |
| agreement tolerance | which episodes count as agreeing | 2 cm / 2° |
| focal range | 0 when known | 0 |

Record what you chose and why in the priors' `notes`.

## 5. Freeze and export

- `r4r fit … --write` ✅ stores mounts, focal and ranges in the priors; `r4r export
  <dataset> robot.xml` ✅ writes the description with every camera placed and each range.
- Add any metadata bug you found to `priors/ISSUES.md` (for the dataset maintainers).

## 6. Check before mass processing

- ⬜ `r4r report`: per-camera residuals, inlier share, agreement across episodes, held-out
  episodes. Today: `research/validation/` scripts and the rerun viewer
  (`research/viz_rerun.py`).
- Look at a recording: the robot skeleton must sit on the robot in every camera.

## Lessons so far

- Joint states can be commanded rather than measured (xrobotwin): FK is then off while the
  arm moves. Check the state against the video (sheet, shift scores).
- Simulators may randomise mounts and focals per episode (molmobot, interndata): declare
  episode-scope ranges instead of fighting for one value.
- A blind focal guess makes the moving-camera search fail on hard datasets (wrong orientation
  with an inflated focal): give the focal.
- One episode rarely pins down an arm base: pool episodes.
