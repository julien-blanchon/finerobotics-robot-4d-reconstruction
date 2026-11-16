# r4r output specification

What r4r writes for every episode of every dataset (`r4r reconstruct`), and how it lands in a
LeRobot v3 dataset (`r4r merge`). Code: `src/r4r/output.py` (columns), `src/r4r/effectors.py`
(end effectors), `src/r4r/merge.py` (packaging).

## Conventions

- **Per frame:** every value is given at each video frame's timestamp, after r4r's lag
  correction (`state_shift`, fitted time offset). Values constant within an episode are still
  written per frame, so every dataset has one uniform schema.
- **Units:** metres, radians.
- **Episode frame:** the robot base frame at the episode's first frame. For a static robot it
  is the base frame. Axes follow REP-103: x forward, y left, z up.
- **Camera axes:** OpenCV, x right, y down, z forward.
- **Pose:** the transform from an entity's frame to the episode frame. Its translation is the
  entity's origin in the episode frame.
- **Pose layout** `[7]`: `x, y, z, qx, qy, qz, qw`, a unit quaternion with `qw ≥ 0`.

```text
                          episode frame E = base at frame 0 (REP-103)
                                 z up
                                 |  x forward
                                 | /
                    y left ------+                     camera (OpenCV)
                                  \                         z forward
        base pose_t = E⁻¹·W_base(t)  \                     /
        (identity on a static robot)   \  ee pose_t        +---- x right
                                         \  = E⁻¹·W_b(q_t)·body_from_ee
                                          \                |
         gripper (TCP)         hand (palm centre)          y down
           z  approach           z  wrist → middle knuckle
           y  closing            y  towards the thumb
           x  = y × z            x  = y × z
```

## Columns

| feature | shape | content |
|---|---|---|
| `observation.camera.<cam>.pose` | 7 | camera pose (OpenCV axes) in the episode frame |
| `observation.camera.<cam>.intrinsics` | 4 | `fx, fy, cx, cy` in pixels of the delivered video, OpenCV pixel centres (integer coordinates) |
| `observation.camera.<cam>.distortion` | 4 | `pinhole`: OpenCV radial `k1, k2, k3, 0` (no tangential); `fisheye`: OpenCV fisheye (Kannala–Brandt) `k1..k4`; zeros when ideal |
| `observation.ee.<arm>.pose` | 7 | the `ee` frame in the episode frame, by forward kinematics with r4r's corrected kinematics |
| `observation.ee.<arm>.gripper` | 1 | two-jaw grippers with pads (parallel, linkage or a single moving jaw): jaw opening in metres |
| `observation.ee.<arm>.aperture` | 1 | jaw opening (parallel gripper, equal to `.gripper`), thumb–index fingertip distance (hand); absent for suction cups and tools |
| `observation.ee.<arm>.fingertips` | F × 3 | hands, multi-finger grippers and grippers whose fingers swing on arcs (HSR): fingertip positions in the episode frame |
| `observation.ee.<arm>.actuation` | 1 | optional: a suction cup's or a tool's recorded actuation |
| `observation.base.pose` | 7 | robot base in the episode frame; identity for static robots; NaN in every frame for a mobile base whose motion is not recorded (feature info `available: false`) |
| `observation.state_aligned` | J | joint state in the description's convention (below) |
| `action_aligned` | J | only for absolute joint-target actions (below) |

`<cam>` is the video key's suffix (`observation.images.<cam>`, or `observation.<cam>` for roots
that keep their videos there, e.g. FetchMan). `<arm>` comes from the priors'
`ee` block, or from the presets when the priors have none (`arm` for one end effector,
`left`/`right` from the body names).

**Cameras not calibrated for an episode** (for example a third view whose session was never
placed) are still written: their pose, intrinsics and distortion are NaN for every frame of that
episode, while the other cameras, end effectors and state are written as usual (`r4r reconstruct
--uncalibrated nan`; the default `solve` runs the full solve instead). With `nan`, an episode
whose robot itself is not calibrated (an arm base, the odometry) is listed as failed with the
reason rather than solved. The side file says, per episode and camera, how the camera was
obtained.

**Cameras the priors ignore** (worn by a person, or a mount no method pins) are written too, NaN
in every frame of every episode, and their video feature says `camera.calibrated: false`: every
camera of the dataset has its columns, so users see it exists and is not calibrated.

### Feature info

- **Video feature** `observation.images.<cam>`: `camera.lens` (`pinhole` | `fisheye`),
  `camera.extrinsics_frame: episode`, `camera.mount` (`static` | `moving` | `per station`: the
  stream is a static view in some setups and a wrist camera in others, robomind2 ark),
  `camera.role`, `camera.depth_key`.
- **`observation.ee.<arm>.pose`:** `ee.type` (`parallel` | `multi_finger` | `dexterous` |
  `suction` | `tool` | `none`), `ee.description_body` (the body the frame hangs on), `ee.preset`,
  `ee.state_indices` and `ee.state_names` (the end effector's own joints within
  `observation.state_aligned`).
- **`observation.ee.<arm>.fingertips`:** `names: {fingers: [thumb, index, middle, ring,
  little], axes: [x, y, z]}` where applicable.
- **`observation.state_aligned`:** `names` are the description's joint names.
- **`info.json`:** `robot.mobile: true | false`; `robot.base_pose: false` when a mobile base's motion
  is not recorded (camera and end-effector poses are then in the base frame at each frame).

### Reserved, not computed yet

`observation.images.<cam>_depth` (metres, along the optical axis) and
`observation.images.<cam>_depth_confidence`, linked by `camera.depth_key`. A depth video at a
different resolution states its own size; its intrinsics follow by rescaling
(`f' = f · s`, `c' = (c + ½) · s − ½`).

## End effectors

The `ee` frame's axes mean the same for grippers and hands:

- **z:** the approach direction (a hand: wrist → middle-finger knuckle);
- **y:** the closing direction (a hand: towards the thumb);
- **x:** `y × z`.

Left and right end effectors both get proper rotations. A left frame is never a mirror image of
a right one. Only the origins differ:

- **gripper or tool:** the tool centre point, between the fingertips;
- **hand:** the palm centre.

| field | how |
|---|---|
| pose | `E⁻¹ · W_b(q_t) · body_from_ee`, with `b` a description body and `body_from_ee` from the priors or a preset |
| gripper | `max((p_a − p_b) · y_t − closed, 0)`: two jaw pad points along the frame's y (`p_a` on the +y side when open), minus their signed distance when the jaws touch; jaws that pass through each other in the description (HSR's motor closing on its finger springs) read 0 |
| aperture | the gripper opening, or `‖p_thumb − p_index‖` for a hand |
| fingertips | `E⁻¹ · W_f(q_t) · o_f`: a named point (body + offset at the fingertip pad) per finger |

**Why `gripper` and the state indices, not the joints again.** The end effector's joint
positions are already in `observation.state_aligned`; `ee.state_indices` points at them, so they
are not repeated. `gripper` repeats the state's gripper information on purpose: raw states give
it as 0–1, percent, inverted ranges or encoder counts, `gripper` always as the opening in metres.
Fingertips and the pose are not in the state at all (forward kinematics).

**Priors** (`ee` block, `src/r4r/data/priors.py`):

```json
"ee": {"right": {"preset": "franka_hand", "body": "panda_hand",
                 "position": [0, 0, 0.1034], "orientation": [1, 0, 0, 0],
                 "pads": [{"body": "leftfinger$"}, {"body": "rightfinger$"}], "closed": 0.0,
                 "fingertips": {"thumb": {"body": "R_thumb_distal", "offset": [0.02, 0.01, -0.01]}},
                 "actuation": {"feature": "observation.suction", "index": 0}}}
```

A `preset` supplies defaults, and any other field overrides them. Pad and fingertip bodies are
regular expressions over the bodies below `body`. Without an `ee` block, presets are matched
against the description, and of nested matches the deepest is kept. A matched pad or fingertip
must end its branch (at most one more link below it). Presets that share body names apply
under a given parent body (`PARENTS` in `src/r4r/effectors.py`). Arms are `left`/`right` from
the body's name, else from its nearest ancestor naming a side.

| preset | body | `ee` origin in the body | jaws / fingers |
|---|---|---|---|
| `franka_hand` | `panda_hand`, `fr3_hand` | 0.1034 m along z (`panda_hand_tcp`) | pads on the finger origins, touching at q = 0 |
| `robotiq_2f85` | `robotiq_85_base_link` | 0.1493 m along z (`robotiq_tcp`), closing along x | pads on the inner fingers (0.079 m open) |
| `piper` | `*gripper_base` | 0.1358 m along z | `link8` / `link7` |
| `arx_x5` | `*link6` | 0.1576 m along x | `link7` / `link8`, 0.0498 m apart when touching |
| `rby1` | `ee_left`, `ee_right` | 0.1328 m along −z | `finger_*1` / `finger_*2` inner faces |
| `omnipicker` | `gripper_[lr]_base_link` (under `arm_*_link7`, AgiBot G1) | 0.14308 m along z | jaw plates `*_link4` near the tips, on the base's y; 0 to 0.129 m |
| `omnipicker_g2` | `gripper_[lr]_base_link` (under `arm_*_end_link`, AgiBot G2, corrected description) | 0.14308 m along z | the same, jaws on the base's x |
| `hsr` | `hand_palm_link` (Toyota HSR) | 0.093 m along z (closed fingertips) | `hand_[rl]_finger_tip_frame`, 4.4 mm apart at q = 0; 0 to 0.132 m; fingertips `left`/`right` at each distal finger's pad tip (fully open 0.046 m along z: the fingers swing) |
| `yam` | `left_gripper`, `right_gripper`, `gripper` (I2RT YAM) | 0.1468 m along −z | `tip_right` / `tip_left` inner faces; 0 to 0.094 m |
| `z1` | `gripperStator` (Unitree Z1) | 0.1489 m along x, on the fixed jaw's face | `gripperMover` / `gripperStator`, closing along the stator's z |
| `rs02_tool` | `[lr]tool_base_link` (RealSource RS-02) | 0.229 m along z, closing along x | `tool_Right_Pad_Link` / `tool_Left_Pad_Link`; 0 to 0.103 m |
| `galaxea_r1_pro` | `(left\|right)_gripper_link` under `arm_link7` | 0.108 m along −z | `gripper_finger_link1` / `2` inner faces; 0 to 0.1 m |
| `galaxea_r1_lite` | `(left\|right)_gripper_link` under `arm_link6` | 0.0784 m along x | `gripper_finger_link1` / `2`, 0.027 m apart touching; 0 to 0.1 m |
| `fourier_hand_right`, `fourier_hand_left` | `[RL]_hand_base_link` (Fourier GR1) | palm centre, 0.05 m from the wrist along −z, y towards the index | five `*_tip_link` fingertips |
| `inspire_right`, `inspire_left` | `[RL]_hand_base_link` | palm centre, 0.09 m from the wrist | five fingertips |

The geometry comes from the descriptions' own frames and collision or visual meshes (jaw tips,
finger pads).

## Joint state and action

**`observation.state_aligned`** is the joint state as r4r uses it, in the robot description's
convention. This is exactly what `qpos_from_state` builds from the joint mapping:

- one entry per description joint the state drives, in description order, named by the
  URDF/MJCF joint name;
- SI units (rad, m), with signs, offsets and unit conversions applied;
- r4r's refined joint offsets (`kinematics.offsets`) and lag correction (`state_shift`,
  `kinematics.time_offset`) applied, at the video frame times. A full solve adds its own
  bundle offsets.

Gripper joints appear as their description joints (for example finger positions in metres). The
raw `observation.state` is not changed. `meta/r4r.json` records the mapping
(`state_mapping`: raw column, sign, `from` → `to` range and fitted `offset` per joint;
`lag_frames`: `state_shift` and `time_offset`), so the conversion can be reversed.

**`action_aligned`** is written only when the dataset's action is an absolute joint target in
the state's columns: its feature info says `vector.space: joint` and `vector.reference:
absolute` with the state's names and shape, or the priors say `"action": "joint_absolute"`. It
uses the same mapping, without lag correction. Any other action (end-effector, delta) is left
untouched, and `meta/r4r.json` says so.

**Publisher columns:** the publisher's own end-effector or camera columns stay as they are under
`publisher.*`. r4r uses them only as checks.

## Side file `meta/r4r.json`

Quality and provenance, not features. The file can be dropped later.

- **Per dataset:** `r4r_commit` (`+dirty` when the checkout had changes), `priors_sha256` (the
  priors JSON and episodes table), `date`, `source` root, `failed_episodes`, `state_mapping`,
  `lag_frames`, `action`.
- **Per episode** (`episodes`; past 20,000 episodes in `meta/r4r.episodes.parquet`):
  `source_episode_index`, `route` (`fast` | `light` | `full`), `median_px`, `track_share`,
  `flagged` (fit check), and `cameras`: per camera `recorded`, `held calibration`,
  `light solve`, `full solve` or `not calibrated`.

## Packaging (`r4r merge`)

```bash
r4r reconstruct <dataset> out/ [--shard i/n] [--uncalibrated nan]   # sharded parquet, resumable
r4r merge <dataset> out/ dest/            # twin: dest/<dataset>--r4r
r4r merge <dataset> out/ dest/ --full     # merged copy: dest/<dataset>
```

- **Twin** (default), geometry-twin style. It holds the index columns (`episode_index,
  frame_index, index, timestamp, task_index`) and r4r's columns. `meta/episodes` links each
  episode back with `source/episode_index`. Videos are not copied: `videos` links to the main
  root's, and each episode keeps its video segments, so the twin opens with LeRobot's loader,
  images included.
- **Full:** the main dataset's rows with r4r's columns added. The videos are linked the same
  way.
- Only reconstructed episodes are written, renumbered from 0. The merge needs every frame
  (`reconstruct --stride 1`).
- File count stays small: one data file per 2M frames (a row group per episode), one
  `meta/episodes` file, `stats.json`, `tasks.parquet`, `info.json`, `r4r.json`.

## Checked on

| dataset | what | result |
|---|---|---|
| SynGrasp-1B (1,000 episodes of the geometry twin) | side camera in the front camera's frame vs the delivered `--geometry` twin (its extrinsics are camera-to-world with world = front camera, estimated) | rotation 2.7° median (9.0° p90), translation direction 1.7° (4.3°), twin/ours scale 1.13 |
| | EE vs `publisher.observation.state.eef_pose.eef` (= FK(`panda_hand`) + 0.132 m z) | 28.6 mm, the TCP convention (0.1034 vs 0.132 m); after that offset 0.001 mm median (0.009 max), 4·10⁻⁵ ° |
| axis sim-panda (116 episodes, held calibration) | wrist in the third-person camera's frame vs the publisher's `camera_to_world` (OpenGL axes) | 2.8 mm / 0.38° median |
| | EE vs `publisher.observation.ee_state` (`panda_hand` in a world 0.478 m above the base, w-first quaternion) | ours − 0.1034 m z: a constant 2.4 mm (spread < 0.01 mm), 3·10⁻⁶ ° |
| molmobot rby1-pick-10 (22 episodes, mobile) | base vs `robot_base_pose` | 6·10⁻⁶ mm, 1·10⁻⁵ ° |
| | cameras vs `extrinsic_cv` (camera-from-world) through the recorded base | 0.05–0.16 mm, ≤ 0.014° median |
| | left TCP vs `tcp_pose` (base frame, w-first) | a constant 29.8 mm along the approach axis (the record's TCP sits at the pad centre, ours at the fingertips), axes turned 180° |
| InternData-M1 franka-robotiq (5 episodes, light route) | EE vs `publisher.state.eef_position` (= FK(`panda_link8`)) | ours − 0.1493 m z: 1·10⁻⁵ mm; orientation a constant 45° turn about the flange axis (spread 5·10⁻⁶ °) |
| nvidia-kitchen-demos panda-omron-mobile (12 episodes, fast route, mobile) | cameras vs MuJoCo FK of the upstream model and `cam_configs` at the recorded qpos (arm joints from the side table) | ≤ 0.0002 mm, 0.00003 °; base ≤ 0.0002 mm over drives up to 3.4 m |
