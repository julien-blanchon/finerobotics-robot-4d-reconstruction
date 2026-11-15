# r4r: robot 4D reconstruction

r4r calibrates robot datasets in the LeRobot format (robot description + joint states +
videos). For every frame it outputs **camera poses, intrinsics and distortion in the robot
base frame, in metres**, end-effector poses, gripper openings and fingertips, the base pose and
the joint state in the description's convention, with corrected kinematics and timing
([output spec](docs/output-spec.md)), merged into a LeRobot dataset. Dense depth and 3D tracks
come later.

<table>
<tr>
<td width="50%"><img src="docs/assets/gallery/droid.gif" alt="DROID overlay"><br><sub><b>DROID</b> (real Panda), two exterior cameras · 0.55 px</sub></td>
<td width="50%"><img src="docs/assets/gallery/interndata-m1.gif" alt="InternData-M1 overlay"><br><sub><b>InternData-M1</b> (sim Panda + Robotiq) · 0.85 px</sub></td>
</tr>
<tr>
<td><img src="docs/assets/gallery/groot-omron.gif" alt="GR00T Panda on Omron base overlay"><br><sub><b>GR00T sim</b>, Panda on an Omron base · 0.38 px</sub></td>
<td><img src="docs/assets/gallery/molmobot.gif" alt="MolmoBot overlay"><br><sub><b>MolmoBot</b> (sim, per-episode mounts and focals) · 0.28 px</sub></td>
</tr>
<tr>
<td><img src="docs/assets/gallery/xvla-softfold.gif" alt="X-VLA soft-fold overlay"><br><sub><b>X-VLA soft-fold</b> (real bimanual AgileX) · 1.30 px</sub></td>
<td><img src="docs/assets/gallery/hsr.gif" alt="HSR overlay"><br><sub><b>HSR</b> (airoa-moma-5k, mobile base from its record), head camera · 0.62 px</sub></td>
</tr>
<tr>
<td colspan="2"><img src="docs/assets/stills/syngrasp-preview.jpg" alt="SynGrasp-1B previews"><br><sub><b>SynGrasp-1B</b> (11,292 h, 4.9M episodes): no solve needed, every episode's cameras come from its recorded scene config</sub></td>
</tr>
</table>

<sub>The robot description, drawn with the solved kinematics and cameras over the video
(`r4r overlay`). Numbers are the median reprojection error of each `r4r rig` solve.</sub>

## How it works

<img src="docs/assets/diagrams/pipeline.svg" alt="Data flow: dataset, description and priors into r4r rig; solution to overlay; written priors to reconstruct" width="100%">

- **Inputs:** a LeRobot dataset, its robot description, and a **priors JSON** that holds what
  is known about the dataset (joint mapping, cameras, focal lengths, ranges).
- **`r4r rig`** fits whatever the priors leave free, on a sample of episodes, with one solver:
  a kinematic bundle adjustment (camera pose = forward kinematics × mount).
- **Check** with `r4r overlay`; **fix** the priors; **write** them back; **`r4r reconstruct`**
  every episode.

## Philosophy: an agent in the loop

<img src="docs/assets/diagrams/onboarding-loop.svg" alt="Onboarding loop: draft priors, preview, fit/rig, overlay; fix and repeat; then write and reconstruct" width="100%">

- r4r is meant to be used on a new dataset by **an agent (or a person) working iteratively**:
  optimise, check by eye (previews, overlays, ground truth when a simulator gives it), tune
  the priors and the description, repeat until the drawn robot sits on the real one.
- **The library** provides the solver and the checks.
- **The agent** brings physical and visual judgement. It keeps its ad-hoc scripts in its own
  scratchpad and records what it found in a standard format:
  `priors/<family>/<dataset>.json` (values) and `priors/onboarding/<embodiment>.md` (evidence).

## Install

```bash
uv sync                    # core library and the `r4r` command
uv sync --extra lerobot    # + the LeRobot dataset loader (needed to read datasets)
uv run r4r --help
uv run pytest              # tests
```

Python 3.13. A dataset is named by its root: an absolute path, or a path relative to
`DELIVERY` in `src/r4r/cli.py` (e.g. `droid/datasets/droid`).

## Commands

| command | what it does | output |
|---|---|---|
| `inspect` | joint mapping, unmapped varying columns, joints past their limits, body tree | text |
| `sheet` | camera frames next to the robot skeleton | PNG |
| `links` | motion evidence for the body each camera rides on (static or moving) | text |
| `preview` | the description drawn at the recorded joints, cameras from the priors, **no solve** | PNG / MP4 |
| `fit` | one moving camera: its body, mount and focal; `--shifts` measures the video/state lag | text, `--write` |
| `rig` | the whole rig in 5 stages (below), checked by groups of episodes; `--picks` clicked robot points for cameras the scene cannot pin | text, `--write`, `--save` |
| `picks` | robot keypoints of a static camera carried from calibrated episodes by RoMa v2 (no clicks), for `rig --picks`; GPU | picks JSON |
| `overlay` | the robot drawn with a saved solution over every camera | PNG / MP4 |
| `export` | the description (MJCF) with every known camera placed and each range | XML |
| `focal` | a focal-length draft from ARROW (gen2 adapter) when nothing is recorded; GPU | text |
| `reconstruct` | every episode with frozen priors: the [output spec](docs/output-spec.md)'s columns per frame (no solve, no video when the priors hold every camera) | sharded parquet |
| `merge` | reconstruct shards into a LeRobot v3 dataset: a `--r4r` twin next to the main one, or `--full` a merged copy | LeRobot root |

```bash
r4r inspect droid/datasets/droid --episode 0
r4r preview droid/datasets/droid preview.png --episode 150
r4r fit     droid/datasets/droid wrist_left --episodes 0 1 2 --shifts -2 -1 0 1 2
r4r rig     droid/datasets/droid --episodes 150 401 1000 300 --stride 2 --groups 2 --save sol.pt
r4r overlay sol.pt check.png                 # a grid: a few moments of every episode
r4r overlay sol.pt episode.mp4 --episode 0   # a video of one solved episode
r4r rig     droid/datasets/droid --episodes 150 401 1000 300 --stride 2 --write  # once it looks right
r4r reconstruct droid/datasets/droid out/ --episodes 0 1 2
r4r reconstruct syngrasp-1b/datasets/syngrasp-1b out/ --shard 3/64  # all episodes, worker 3 of 64
r4r reconstruct droid/datasets/droid --plan  # routes and CPU / GPU hours, nothing run
r4r stations openneodata/datasets/arx5 third_view --write  # setups from thumbnails (CPU)
r4r merge   syngrasp-1b/datasets/syngrasp-1b out/ dest/  # dest/syngrasp-1b--r4r (LeRobot v3)
```

`reconstruct` writes `out/<family>/<dataset>/shard-<k>.parquet`, one file per `--shard-size`
episodes (default 1000) of the selection (`--episodes`, default all): one row per frame with
the columns of [`docs/output-spec.md`](docs/output-spec.md) (camera poses, intrinsics,
distortion; end-effector pose, gripper, aperture, fingertips; base pose; aligned joint state;
all in the episode frame). The schema metadata `r4r.episodes` (JSON) says per episode its
route, fit check, each camera's source and errors, or the error that stopped it; `r4r.layout`
holds the feature info. `r4r merge` writes them as a LeRobot v3 dataset with `meta/r4r.json`
(quality and provenance). Measured on a login node: fast route 13 ms per SynGrasp episode
(1,000 episodes in 13 s), 0.32 s per sim-panda episode sampled across 460k (one data file
each); light route 41-57 s per episode under load; merge ~12 s per 1,000 SynGrasp episodes.

Each episode takes the cheapest route its unknowns allow (`r4r.plan`): values of dataset or
station scope are held once `r4r rig --write` wrote them (or recorded); only episode-scope ones
stay free.

| route | when | cost (measured) |
|---|---|---|
| fast | everything held or recorded (a moving base on its odometry with a fitted `base_odometry.correction`, or a recorded pose) | forward kinematics × mount, no video: 13-40 ms CPU per episode |
| light | episode-scope 6-DoF unknowns with a start (a re-placed robot base, a re-placed static camera), or a real episode the fit check flags | 6 bursts of 16 frames decoded alone, KLT + SIFT links, LM on those unknowns only: 5-9 s CPU |
| full | its setup was never calibrated, or an unknown has no start | the whole rig: ~0.36 GPU s per second of video |

- **Fit check:** real episodes holding a recorded episode-scope camera (DROID's per-episode
  calibration) are checked on the bursts: the epipolar distance of verified matches between
  the held cameras; above 25 px the record is freed and solved light (`--check on/off/auto`).
- **`--plan`:** counts the routes over a dataset from its priors and episodes table alone, and
  estimates CPU and GPU hours (calibrating each uncalibrated setup once included).
- **`r4r stations`:** runs of episodes with one placement of a camera that is not on the robot,
  from two thumbnails per episode (~30 ms per episode); `--write` stores them as the `station`
  column of the episodes table.
- `--solve` solves every episode in full.
- **Resumable:** a shard is written once, atomically; shards already written are skipped.
- **Spread over processes:** `--shard i/n` takes the shards `k` with `k mod n = i`, e.g. a
  Slurm array: `--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT`.

## The stages of `r4r rig`

<img src="docs/assets/diagrams/rig-stages.svg" alt="The five stages of r4r rig" width="100%">

The same stages on one real DROID episode:

<img src="docs/assets/stills/rig-on-droid.jpg" alt="r4r rig on DROID episode 150: KLT tracks, RoMa links, solved overlay" width="100%">

## Frames

<img src="docs/assets/diagrams/frames.svg" alt="Base frame, link, moving wrist camera and static cameras, from a real DROID solution" width="100%">

- **Moving camera:** rigid on a link. Pose = FK(joint states) · mount; the mount is fitted once.
- **Static camera:** a pose in the base frame, per dataset or per episode.
- **Free bodies:** a second arm's base, a mobile base (per frame, started from its odometry).

## A new dataset, step by step

| step | do | tool |
|---|---|---|
| 1 | Is it in scope? joint states + a description; the loader must run | `r4r inspect` |
| 2 | Fix kinematics: mapping, signs, units, video/state lag | `inspect`, `sheet`, `fit --shifts` |
| 3 | Decide each camera: static or moving, link, focal (recorded first), lens, scope | `links`, `sheet`, dataset metadata |
| 4 | Write the draft priors: `priors/<family>/<dataset>.json` | editor |
| 5 | Place cameras by eye until the drawn robot is close | `r4r preview` (iterate) |
| 6 | Fit moving mounts, then the whole rig on 5–20 episodes | `r4r fit`, `r4r rig --groups 2 --save` |
| 7 | **Check the overlay**: every camera, several episodes | `r4r overlay` |
| 8 | Write the fitted values back; record the evidence | `r4r rig --write`, `priors/onboarding/` |
| 9 | Process every episode | `r4r reconstruct` |

A minimal priors file (trimmed from `priors/droid/droid.json`):

```json
{
  "description": "descriptions/franka__panda/panda_robotiq.urdf",
  "joints": [{"index": 7, "urdf_joint": "gripper_finger_joint", "from": [0, 1], "to": [0, 0.725]}],
  "cameras": {
    "wrist_left": {"mount": "moving", "link": "panda_link8",
                   "position": [-0.0769, 0.0326, 0.0181], "orientation": [0.6963, -0.1177, 0.1221, -0.6965],
                   "position_range": 0.03, "orientation_range_deg": 5.0, "focal": 183.0, "focal_range": 0.01},
    "exterior_1_left": {"mount": "static", "scope": "episode", "focal": 131.0, "focal_range": 0.02,
                        "position_range": 0.05, "orientation_range_deg": 3.0}
  }
}
```

**Low residuals are not enough. Look at the overlay.**

<img src="docs/assets/stills/before-after-droid.jpg" alt="Wrong vs right calibration of DROID's exterior cameras" width="100%">

Full guide: [`docs/dataset-onboarding.md`](docs/dataset-onboarding.md) (checklist, failure
patterns with images). Field notes: [`docs/field-notes.md`](docs/field-notes.md).

## Repository

| path | content |
|---|---|
| `src/r4r/` | `geometry/` (SE(3), lens models), `data/` (LeRobot loader, priors), `rago/` (tracks, search, bundle, stages), `arrow/` (ARROW reimplementation), `matching/` (RoMa v2), `viz/` (overlay, sheet), `cli.py` |
| `priors/` | one JSON per dataset, corrected descriptions, onboarding records per embodiment, `ISSUES.md` for dataset maintainers |
| `docs/` | [output spec](docs/output-spec.md) · [architecture](docs/architecture.md) · [onboarding](docs/dataset-onboarding.md) · [field notes](docs/field-notes.md) · [related work](docs/related-work.md) |
| `tests/` | mirrors `src/r4r/`: SE(3), lenses, kinematics, bundle and stages on synthetic data, parity with reference releases |
| `experiments/` | measured studies (focal sources, ARROW pose, Kosmo-GFM) with their results |

Built on ARROW ([arXiv:2610.01314](https://arxiv.org/abs/2610.01314), re-implemented) and
Calib3R-style robot-aware calibration ([arXiv:2509.08813](https://arxiv.org/abs/2509.08813)).
Code conventions: [`STYLE.md`](STYLE.md).
