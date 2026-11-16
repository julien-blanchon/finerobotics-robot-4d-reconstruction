# End-to-end verification of the ready datasets (2026-11-16)

Every dataset whose onboarding status row starts with "ready" (plus the ready stations of
abc-130k, AgiBot G1 and openneodata arx5/flexiv) was run end to end on a random sample, from
the delivered root to a merged twin. Nothing here is a full-dataset run.

## Held-out click validation of the real datasets (2026-11-16)

The overlays below were read by eye. Clicked points on held-out episodes measure them, and
**no real dataset holds at the bar of ~15 px median** (about 2.3 % of the image width, the
cobot lanes' 640 px bar scaled). Simulated datasets are covered by the ground truth below.

**Method** (`python -m r4r.viz.clicks`, `src/r4r/viz/clicks.py`):
1. `sample`: ~10 random episodes spread over data files and stations, without any episode
   that a rig, picks or check in the onboarding record used.
2. `r4r reconstruct … --episodes … --uncalibrated nan` (CPU, commit `5234868`/`698201f` priors).
3. `make`: per episode, one scene-view moment (the gripper farthest from where it started,
   inside the image) and one wrist-view moment showing the other arm (bimanual). The raw frame is
   saved with a pixel grid and a zoom, with nothing drawn.
4. One real fingertip is clicked per target (hidden or ambiguous ones skipped), and `score`
   measures the error:
   - to the nearer drawn jaw point `TCP ± y·w/2` (the opening recorded);
   - to the open jaw line when the opening is constant;
   - hiw-500 has no EE columns, so its clicks are scored on the flange point.
   Projection is OpenCV on the twin's columns, independent of r4r's camera code.
5. `score` also moves the drawn points along the approach axis and reports the error at the
   median best shift. A large drop at one consistent shift means the TCP depth (the
   description's gripper length) is wrong, not the camera.
6. Click noise is about 5-10 px: grippers are often small, blurred or in fisheye views.

Clicks, targets and drawn points, one JSON per dataset:
`/e/scratch/reformo/blanchon1/r4r-cache/verify/clicks/<family>__<dataset>.json`.

| dataset | h | n (skipped) | median / p90 / max px | image | % width | TCP shift → median | verdict | pattern |
|---|---|---|---|---|---|---|---|---|
| airoa-moma-5k (HSR) | 5025 | 8 (1) | 36.9 / 70.8 / 108 | 512×384 | 7.2 | −2 cm → 27.4 | fail | head off per episode, ahead of or beside the fingers |
| agibot-world g1-cohort-04 | 2620 | 23 (7) | 71.1 / 122 / 146 | 512×384 (fisheyes 512×410) | 13.9 | +10 cm (scan edge) → 27.6 | fail | top_head 71 px, drawn 1.3-1.6× farther from the image centre than the real grippers (focal); hands seeing the other arm 98 px; fisheyes 20-46 px; jaw points at the jaw roots (TCP short); lag ±45 frames ruled out |
| abc-130k abc-640x480 | 1752 | 21 (19) | 17.4 / 42.0 / 54.0 | 512×384 | 3.4 | +1 cm → 17.5 | fail | top 16.4 px (n 14), wrists 26.9 (left 38). Per station (top / wrist): 2: 17; 8: 16; 9: 54; 18: 17; 19: 21, 16 / 17; 22: 7 / 14; 24: 10; 26: wrist 42; 27: 15 / 34; 29: 8 / 26; 30: 10 / 27; 31: 20; 35: 24 / 47; 37: 41 |
| abc-130k abc-848x480 | 723 | 19 (11) | 17.0 / 49.4 / 54.8 | 512×290 | 3.3 | +3 cm → 15.7 | fail | top 14.1 px (n 12, holds); wrists 46.5 (left 47.8): the wrist mounts |
| molmoact2-yam bimanual-yam | 704 | 10 (10) | 35.6 / 113 / 115 | 512×288 | 7.0 | −5 cm → 31.7 | fail | top 25.2 px (per-episode light solve), wrist views of the other arm 114 px |
| openneodata arx5 (8 stations) | 660 | 7 (13) | 24.1 / 69.5 / 137 | 512×384 | 4.7 | +1 cm → 24.0 | fail (low confidence) | third view 16-25 px in no common direction (light solve per episode); one wrist 137 px |
| g2-cohort-01 | 516 | 18 (12) | 29.6 / 85.2 / 148 | 512×320 / 410 / 422 | 5.8 | +4 cm → 26.9 | fail | top_head 41 px, hands 27, head fisheye 14; per episode (cart episodes 1696, 10102, 26517 and 5166 worst: problem 8) |
| droid | 512 | 18 (2) | 13.4 / 86.6 / 109 | 320×180 | 4.2 | +3 cm → 12.9 | fail (% width) | 6 of 9 episodes 3-17 px; 29119 and 74644 (recorded calibration, not flagged) 80-110 px, 27915 (flagged, light solve) 57-86 px: exterior pose wrong per episode |
| galaxea-open-world r1-lite | 484 | 11 (6) | 24.4 / 32.1 / 49.9 | 512×288 | 4.8 | −2 cm → 22.4 | fail | head views 23.5 px in mixed directions (parked episodes; no driving one sampled), wrists 32 |
| flexiv-rizon4 (1 station) | 412 | 10 (0) | 50.4 / 70.1 / 72.1 | 512×384 | 9.8 | +10 cm → 11.6 | fail | all drawn 45-70 px above the fingertips, flange clicks 2-11 px: the description's gripper ~10 cm shorter than the real one |
| hiw-500 | 378 | 10 (5) | 11.0 / 31.8 / 41.5 | head 256×192, wrists 512×384 | 4.3 | — | fail (% width) | flange point: head drawn 7-22 px above in 5 of 7 (≈ 26 px at 640), wrist views of the other arm 29-38 px low |
| realsource-world | 341 | 15 (4) | 14.8 / 120 / 218 | 512×288 | 2.9 | +1 cm → 13.0 | fail (% width, borderline) | head 14.8 px (max 25); hand cameras 4 of 6 within 28 px, 2 at 181-218 px (18189, 20401: the per-episode hand record wrong) |
| robomind2 agilex-5b28 Piper | 333 | 12 (3) | 59.8 / 96.9 / 205 | 512×384 | 11.7 | +6 cm → 38.9 | fail | drawn above the real jaws in front and wrist views at every station; 17882 ~200 px (held front camera off) |
| g2-cohort-02 | 198 | 10 (5) | 20.0 / 38.3 / 43.1 | 512×320 / 410 / 422 | 3.9 | +6 cm → 11.6 | fail | mostly TCP depth (omnipicker_g2) |
| rw-rl r1lite (5 datasets, pooled) | 165 | 10 (10) | 15.1 / 57 / 59 | 512×288 | 2.9 | per dataset, signs differ | fail (borderline) | top views 4-22 px, three outliers (tissue 403, plug 239 wrist, scan 1845) |
| openneodata aloha | 157 | 7 (10) | 54.9 / 228 / 234 | 512×384 | 10.7 | +10 cm → 62.9 | fail | third view (light solve per episode) 45-68 px off on most; 2 clicks ~230 px |
| robomind agilex-real | 59 | 16 (4) | 55.9 / 75.5 / 80.3 | 512×384 | 10.9 | +6 cm → 42.0 | fail | as agilex-5b28: drawn jaw points at the jaw roots and above the real jaws in front and wrist views; state lag ruled out (best at 0-2 frames) |
| robomind ur5e-real | 36 | 11 (0) | 19.4 / 27.6 / 61.7 | 512×384 | 3.8 | +4 cm → 15.4 | fail | drawn 10-30 px above; part gripper length, rest per-session camera |
| rdt-ft-data | 30 | 11 (9) | 44.3 / 71.1 / 79.8 | 512×384 | 8.7 | 0 cm → 44.3 | fail | cam_high 34 px in a different direction per room (one held pose for every room); wrists 67 |
| xvla-soft-fold | 26 | 13 (7) | 22.1 / 68.1 / 98.0 | 512×384 | 4.3 | +2 cm → 20.1 | fail | cam_high 2-9 px on 5 of 10 episodes, 19-25 px on most others; wrists 22-73 px |
| g2-cohort-03 | 6 | 5 (4) | 12.8 / 23.3 / 26.0 | 512×320 / 410 | 2.5 | +8 cm → 7.8 | holds in px, 5 clicks only | TCP depth |
| openneodata ur5e (station stack-cups-reversely-A, partial) | ~1 | 10 (0) | 34.5 / 46.7 / 51.2 | 512×384 | 6.7 | +5 cm → 10.3 | fail | all drawn up-right of the fingertips: gripper length |
| robocoin lane103 (earlier) | 62 | 7 | 12.5 / — / 27 | | | | holds | `priors/onboarding/agilex__cobot_magic.md` |

**Hours.** Holding: lane103 (62 h) and g2-cohort-03 (6 h, 5 clicks), about 68 h. Failing:
about 15,100 h of real data by dataset hours (flexiv's ready part is one station of ~9 h, arx5
660 h is its 8 calibrated stations). Of these, abc-848 top (723 h) and realsource-world head
(341 h) are at the line and fail on their wrist or hand cameras.

**Patterns:**
- **The gripper drawn short of the real fingertips: TCP depth or gripper length.** At one
  shift for the whole dataset the error mostly goes: flexiv (+10 cm), openneodata ur5e
  (+5 cm), g2-cohort-02/03 (+6/+8 cm). It also accounts for part of the error in
  agilex-real, agilex-5b28 and g1 (+6 to +10 cm, ~40 / 28 px left). The cameras of these
  datasets are better than the plain numbers say; the twin's EE points are not.
- **Per-episode or per-session camera error.** droid (a third of episodes, recorded or
  light-solved), airoa-moma head, g2-cohort-01 (cart), molmoact2, openneodata arx5 and aloha
  (light-solved third views 16-68 px), rdt (one cam_high for many rooms), ur5e-real sessions.
  The keyframe check (`check_px`) did not flag droid 29119 and 74644 (80-110 px off).
- **Focal / scale:** g1-cohort-04 top_head (fx 256.6, aspect 1.33): the drawn points spread
  1.3-1.6× wider than the real ones.
- **Wrists:** abc-848 and abc-640 left wrists (38-48 px), molmoact2 (114 px), hiw-500,
  agilex wrists; realsource-world 2 per-episode hand records far off.
- Not far-reach only. Lag was scanned on agilex-real (±20 frames) and g1 (±45): no effect.


## Method

Per dataset, CPU only (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=8`), commit `54b8e49`:

1. `r4r reconstruct <dataset> --plan`: route mix and cost.
2. 10 episodes, seeded (`sha256(name)`), one per data file where possible; restricted to the
   ready stations (abc, openneodata) or the ready episodes (robomind ur5e-real: moving state;
   unitree-z1: episodes 0-2), and to non-full routes for mixed datasets.
   `r4r reconstruct … --episodes … --uncalibrated nan`, then `r4r merge` (twin).
3. Checks from the twin only: columns and feature info against `output-spec.md`, NaN only in
   cameras the side file calls `not calibrated` or the priors ignore, `|q| = 1` and `qw ≥ 0`,
   base pose identity at frame 0 (everywhere for static robots), `meta/r4r.json` fields.
4. Overlay from the twin's fields on frames loaded through `LeRobotDataset` (2 episodes ×
   3 frames, every camera): TCP axes, jaw points at ±aperture/2, fingertips, with the twin's
   pose, intrinsics and distortion. Twins without EE columns: the description's meshes (or
   body skeleton when it has no geometry) drawn with the twin's camera and base poses.
5. Ground truth where the dataset records it: per-episode rigid alignment of the record's
   frame, per-camera position/rotation error, EE convention offset (constant) and residual
   (its spread).

Outputs: `/e/scratch/reformo/blanchon1/r4r-cache/verify/out/` (plan, sample, check and GT
JSON per dataset) and `verify/contact-sheet.jpg` (one overlay row per dataset).

## Results (by hours)

Plan: routes after calibration (`fast/light/full`), CPU h / GPU h for the whole dataset.
Sample: routes taken by the 10 episodes. NaN: cameras NaN in the sample, all documented
unless stated.

| dataset | h | plan routes | CPU h / GPU h | sample | NaN | ground truth | overlay |
|---|---|---|---|---|---|---|---|
| syngrasp-1b | 11292 | 4.9M fast | 43 / 0 | 10 fast | none | EE 0.001 mm after the 28.6 mm TCP convention | ok |
| airoa-moma-5k (HSR) | 5025 | fast | 11 / 0 | 10 fast | none | EE vs `end_effector_pose.absolute` (x y z rpy): rotation between frames agrees (0.08° median, 0.9° p95); the record's position frame not resolved (no fixed base or odometry frame fits, 62 mm) | head: robot on the gripper; hand fisheye: fingers at the image edge (status: partial). EE (`hsr`, 3 episodes): TCP between the fingers in head and hand views, jaw points on the finger pads holding a can; aperture 0 when the motor closes past contact (springs), so it reads less than a held object |
| molmobot franka-15.15 | 4406 | fast | 14 / 0 | 10 fast | none | cameras 0.00-0.02 cm, < 0.01°; EE 0.13 mm (5.5 mm TCP convention) | ok |
| axis sim-panda-cohort-01 | 3155 | fast | 5 / 0 | 10 fast | none | (checked before, `output-spec.md`) | ok |
| agibot-world g1-cohort-04 | 2620 | fast (all checked) | 203 / 0 | 10 fast | none | — | ok on heads and top; gripper/aperture (`omnipicker`, 2 episodes): jaw points on the jaw tips open and closed in the hand views |
| abc-130k abc-640x480 (16 stations) | 1752 | light 74k (9 rig setups, 1 neighbour, 6 odd) | 106 / **2.5** | 10 light | none | — | skeleton on both arms in top and wrist views. EE (`yam`, 2 episodes): jaw points on the fingertips open and closed in both wrist views, TCPs on the grippers in top |
| interndata-a1-sim arx-lift-2 | 1516 | fast | 3 / 0 | 10 fast | none | head 0.001 cm; wrists 0.01-0.16 cm (p95 1.4 cm, 2.3°); EE resid 0.7-1.3 mm (p95 8 mm), convention 90° | ok |
| interndata-a1-sim agilex-split-aloha | 1508 | fast | 3 / 0 | 10 fast | none | head 0.002 cm; wrists 0.3-0.6 cm (p95 1.8 cm, 2.5°); EE resid 2.3-4.5 mm (p95 11 mm) | ok |
| groot gr1-arms-and-waist-fourier-hands | 841 | fast | 2 / 0 | 10 fast | none | — | arms on the arms; drawn fingers do not follow the rendered hands. EE (`fourier_hand_*`, 2 episodes): palm frames on the palms; fingertips a few cm off the rendered ones, the thumb tip far off (the hand's joint map, not the preset) |
| interndata-m1 franka-robotiq | 811 | light | 324 / 0 | 10 light | none | EE exact (0.00 mm) vs `eef_position/quaternion` = link8 (−149.3 mm, 45°) | ok |
| abc-130k abc-848x480 (15 stations) | 723 | light (227 full now: top focal unknown) | 31 / 0.2 | 10 light | none | — | skeleton ok. EE (`yam`): same description as abc-640x480 |
| molmoact2-yam bimanual-yam | 704 | light (top camera per episode) | 46 / 0 | 10 light (85 s) | none | — | skeleton on both arms in the top view (fixed 2026-11-16). EE (`yam`) |
| openneodata arx5 (8 stations) | 660 | 31.9k light, 414 neighbour, 1096 odd; 27.6k full now | 52 / **95** | 9 light, 1 fast | none | — | ok |
| interndata-a1-sim franka-5c46641bda | 527 | fast | 1 / 0 | 10 fast | none | head 0.00 cm; hand 0.34 cm (p95 1.1); EE resid 3.5 mm (45.7 mm, 45° convention) | ok |
| agibot-world g2-cohort-01 | 516 | fast | 0.6 / 0 | 10 fast | none | — | ok. EE: the delivered description draws the OmniPicker jaws across the wrist camera's side (camera mount at −8 cm y, jaws on y); corrected description `agibot__g2` (jaws on x) and preset `omnipicker_g2`: jaw points on both jaws in the hand views. The −90° turn and +90° put the plates at the same places (they differ only by which plate is on which side), and no view separates the two plates; the preset is tied to the −90° file. Head cameras: the arms drawn several cm off the real ones in 4 of 17 episodes checked (problem 8) |
| droid | 512 | 95.6k fast (checked), **8 full** | 66 / 0.02 | 3 fast, 7 light (flagged) | none | — | ok |
| robointer-droid | 512 | as droid, 8 full | 66 / 0.02 | 7 fast, 3 light | none | — | ok |
| galaxea-open-world r1-lite | 484 | fast | 0.3 / 0 | 10 fast | none | — | fingers on the wrist fingers; base motion not checkable on video (fixed 2026-11-16). EE (`galaxea_r1_lite`) |
| molmobot rby1-door-opening | 426 | fast | 0.7 / 0 | 10 fast | none | cameras 0.02-0.03 cm, 0.01° | ok |
| openneodata flexiv-rizon4 (1 station) | 412 | fast 444 / full 20.4k now | 0.4 / 0 | 10 fast | left_wrist (ignored) | — | ok |
| realsource-world | 341 | fast | 0.4 / 0 | 10 fast | none | — | meshes on the wrist grippers. EE (`rs02_tool`, 2 episodes): jaw points on the pad tips open and closed in the hand views |
| groot panda-omron-base | 273 | light | 100 / 0 | 10 light | none | — | ok |
| hiw-500 (added 2026-11-16) | 378 | fast | 0.4 / 0 | 10 fast | none (base pose NaN by design, `robot.base_pose: false`) | — | head points on the arms (episodes 604, 11969, from the twin's fields); wrists checked on the feet separately. Known exception: the head is 20-50 px off in clothes washing (14.7 h) and one bedroom home. No EE columns (Dex1 not modelled) |
| robocoin lane00-realman (added 2026-11-16, partial) | 154 | fast, all 3 cameras not calibrated | 0.2 / 0.25 | 10 fast | all 3 cameras (not calibrated) | — | head and wrists NaN; the former held head pose missed the grippers (`hiw-lane00/o/twin_l00.png`) and was removed |
| groot r1-pro (3 of 18 roots) | 254 / 20 / 10 | fast | ~0 | 10 fast each | none | — | arm meshes on the arms. EE (`galaxea_r1_pro`, root 88b8…, 2 episodes): jaw points on the other arm's fingertips seen from each wrist; own fingertips at the wrist image edges |
| agibot-world g2-cohort-02 / -03 | 198 / 6 | fast | ~0 | 10 fast | none | — | ok; EE as g2-cohort-01 (corrected description) |
| molmobot rby1-open / pick-10 / pick-15.15 | 189 / 143 / 55 | fast | ~0 | 10 fast | none | cameras ≤ 0.04 cm, ≤ 0.015°; EE 0.1-0.2 mm (30.3 mm, 180° convention) | ok |
| interndata-a1-sim genie-1 | 174 | fast | 0.5 / 0 | 10 fast | none | head 0.25 cm; wrists 0.2-1.1 cm (p95 3-4 cm, 4°); EE resid 1 mm, **p95 29 mm**; EE vs `states.*_tcp_to_robot_pose`: record TCP 76 mm further along z than ours (0.219 m from the gripper base, beyond the description's jaw tips), axes turned 90° about z, residual 1.1-1.6 mm median (p95 24-29 mm); aperture = 0.127 · `states.*_gripper.position` (1 = open) | ok; jaw points on the jaws |
| openneodata aloha | 157 | light (third view per episode) | 17 / 0 | 10 light | none | — | arms on the real ones within a few cm in the third view, right gripper up to ~5 cm off in one episode; wrist fingers approximate (fixed 2026-11-16) |
| interndata-a1-sim franka-b090d20bcf | 142 | fast | 0.5 / 0 | 10 fast | none | head 0.00 cm; hand 0.43 cm / 0.74°; EE resid 4.1 mm | ok |
| axis sim-sawyer-cohort-02 | 108 | fast | 0.1 / 0 | 10 fast | none | — | meshes on the arm. **No EE columns**: the description has no gripper, and the recorded finger values (state 7: 0 / 0.040, state 8: −0.030 / 0, two-valued) do not fit robosuite's Rethink gripper (finger ranges −0.0115..0.0208 / −0.0208..0.0115 m), so the simulated gripper is not the published one and its pads and closing axis cannot be placed |
| interndata-m1 franka-panda | 64 | light | 25 / 0 | 10 light | none | EE position exact (0.00 mm) at link8 | ok |
| open-x-embodiment maniskill-sim | 63 | fast (main camera per episode: 6 setups) | 0.3 / 0 | 10 fast | none | EE exact; cameras 0.00 cm in all 10 (fixed 2026-11-16) | ok |
| robomind agilex-real | 59 | fast | 0.1 / 0 | 10 fast | none | — | right gripper on the real one in camera_front, jaw points on the wrist jaws (fixed 2026-11-16) |
| deco50 body35 | 45 | fast | 0.1 / 0 | 10 fast | none | — | hands approximate (frames off by ~5% of the width in one episode) |
| rw-rl r1lite-spray-bottle-head-install | 44 | fast | ~0 | 10 fast | none | — | meshes on the jaws and arms. EE (`galaxea_r1_lite`, 2 episodes): jaw points on the fingertips open and closed in both wrist views; vs `left/right_ee_pose`: relative rotations agree (0.02° / 0.08° median), positions not in a fixed base frame (residual 22-49 mm, the record's frame moves with the torso) |
| robomind ur5e-real (moving episodes) | 36 | fast in 125 sessions; 35 sessions (4,392 episodes) not calibrated | 0.2 / 0 | 10 fast | camera_top in 2 (uncalibrated sessions) | — | arm on the real one in the 8 held episodes; one moving episode per session read (fixed 2026-11-16) |
| rw-rl r1lite-shelf-restock / tissue-box / scan-item / plug-charger | 33 / 33 / 30 / 25 | fast | ~0 / 0 | 10 fast each | none | — | fingers on the wrist fingers; base motion not checkable on video (fixed 2026-11-16). EE (`galaxea_r1_lite`) as spray-bottle |
| rw-rl g2-box-unpack-pack (+ carton, auto-parts, zip-tie, key) | 32 (+64) | fast | ~0 | 10 fast each | stereo pair (ignored) | — | ok; EE (`omnipicker_g2`, box-unpack 2 episodes): as g2-cohort-01 |
| rdt-ft-data | 30 | fast | 0.1 / 0 | 10 fast | none | — | cam_high sees the grippers only at its top edge, TCPs drawn near them; right base weakly pinned (fixed 2026-11-16) |
| xvla-soft-fold | 26 | fast | ~0 / 0 | 10 fast | none | — | TCPs on both grippers in cam_high; one episode-end frame off (its streams out of step) (fixed 2026-11-16) |
| abc-130k abc-1280x720 | 23 | light | 1.4 / 0 | 10 light | none | — | skeleton ok. EE (`yam`) |
| humanoid-everyday h1 | 15 | fast | ~0 | 10 fast | none | — | ok (hands, fingertips) |
| groot panda-dex-rh / panda | 14 / 12 | fast | ~0 | 10 fast | none | — | ego/front ok; dex-rh EE is typed parallel (dex hands not in the description) |
| community-v3 unitree-z1 (eps 0-2) | 9 | light | 1.7 / 0 | 3 light | none | — | arm mesh on the arm. EE (`z1`, 2 episodes): TCP on the jaws in top; jaw opening 0.04-0.08 m |
| molmobot franka-pick-and-place-fps5 | 9 | fast | ~0 | 10 fast | none | cameras 0.00 cm; EE 0.05 mm | ok |
| dexmimicgen transport | 6 | fast | ~0 | 10 fast | none | — | ok |

Every merged twin passed the format checks: columns and shapes as specified, `|q| − 1 <
1e-4` and `qw ≥ 0` everywhere, base identity at frame 0 (and in every frame for static
robots), `meta/r4r.json` with every dataset and per-episode field, NaN only where the side file
says `not calibrated` or the camera is ignored (`camera.calibrated: false`).

## Problems found

1. **Ready in the onboarding table, but the calibration is not in the priors.** Every
   sampled episode fails with `--uncalibrated nan` (and would take the full GPU solve without
   it); `--plan` counts them as "rig setups 1":
   - right arm base placement not written: molmoact2-yam bimanual-yam (704 h; the 2026-11-15
     rig result, right base (−0.029, +0.197, +0.002) m / 6.8°, is only in the notes),
     openneodata aloha, robomind agilex-real, rdt-ft-data;
   - no `base_odometry.correction`: galaxea-open-world r1-lite (484 h) and rw-rl
     r1lite-shelf-restock, -tissue-box, -scan-item, -plug-charger;
   - focal unknown for every camera: xvla-soft-fold.

   **Fixed 2026-11-16:** every one of these rigs had a `--save` file; the values were written
   from it as `rig --write` would (body corrections `E = nominal⁻¹ · fitted`, ranges = group
   disagreement; odometry gains with `odometry.fit_gain` against the saved base; xvla's focals
   and cam_high pose). openneodata aloha's third view is now episode scope (it is re-placed
   between sessions). Each was re-run as above (rows marked "fixed 2026-11-16"). Weak spots:
   agilex-real and rdt right bases (groups 6-8 cm / 10-14° apart), the r1lite gains (two fits
   differ by ~20 %; tissue-box, scan-item and plug-charger reuse shelf-restock's).
2. **No end-effector columns** (no `ee` block and no preset matches): fixed 2026-11-16 by
   presets keyed on the descriptions (`hsr`, `yam`, `z1`, `rs02_tool`, `galaxea_r1_pro`,
   `galaxea_r1_lite`, `fourier_hand_*`) for airoa-moma-5k, abc-130k, molmoact2-yam, gr1,
   groot r1-pro, realsource-world, r1lite (rw-rl and galaxea-open-world) and unitree-z1 (single
   and dual). Still none: axis sim-sawyer (the description has no gripper). HSR reads 0 when
   its motor closes past contact; GR1's fingertips follow the description's finger map, which
   does not match the rendered hands.
3. **maniskill-sim cameras are not one setup.** The delivered `main_camera_extrinsic_cv`
   takes 5 distinct values over the 30,213 episodes (22,046 / 4,597 / 1,691 / 971 / 908);
   the priors hold the most common one for all, so ~27% of episodes get a wrong static
   camera (24 cm / 90° off in sampled episode 2065). EE and wrist are exact.
   **Fixed 2026-11-16:** in the base frame there are 6 setups (the robot base also moves:
   17,151 / 4,895 / 4,597 / 1,691 / 971 / 908); every episode's main camera is now in
   `maniskill-sim.episodes.parquet`. Ground truth: 0.00 cm on all 10 sampled episodes.
4. **robomind ur5e-real** (ready for the moving episodes): its only camera, `camera_top`,
   is `not calibrated` in all sampled episodes (plan: 1 placement pending), so the twin
   has no calibrated camera.
   **Partly fixed 2026-11-16:** the camera is re-placed between recording sessions and is the
   only camera, so no automatic per-episode or per-setup route exists for it (light needs a
   second camera, `r4r station` checks against wrist links). Stations = source archives (167);
   the rig pose is held for the 125 sessions whose sampled episode draws the arm on the real one
   (1,473 of 2,733 moving episodes); 35 sessions stay NaN (30 visibly 5-10 cm off or another
   room, 5 not judgeable). They need one rig each or transferred picks.
5. **omnipicker grippers** (AgiBot G1/G2, genie-1, rw-rl g2): fixed 2026-11-16, pads on the
   jaw plates (`gripper` and `aperture`, 0 to 0.129 m). The G2 description draws its jaws
   across the wrist camera's side; G2 priors now use the corrected `agibot__g2` description
   (jaws on the base's x) and the `omnipicker_g2` preset. g2a-simulation: its gripper columns
   (state 0-1) are now mapped as on the real G2; the rendered jaws spread across the hand
   view and the drawn ones land on them.
6. **Light route on a loaded login node:** with default threading (~300 threads per process)
   10 light episodes took over 16 minutes; with `OMP_NUM_THREADS=8`, 150-170 s.
   **Fixed 2026-11-16:** `r4r reconstruct` sets 8 threads (torch, OpenCV, exported to the
   environment) unless `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` is set; 3 light
   yam episodes with no thread setting: 21 s.
7. Ground-truth outliers worth a look: genie-1 EE residual p95 29 mm and wrist cameras p95
   3-4 cm (median ~1 mm / 0.2-1 cm, so frames during fast motion); a1-sim wrists p95
   1-2 cm generally.
8. **G2 head cameras vs the arms** (checked on held calibration, 2026-11-16): the arms are drawn
   several cm off the real ones in the head views in 4 of 17 episodes checked: 1696, 10102,
   1600, 2000 (g2-cohort-01, supermarket aisles with a shopping cart; 22426 slightly). They
   sit on the real arms in 7906, 8166, 12243, 17602, 17708, 22336, 22777, g2-cohort-02 1217,
   4251, 7714, g2-cohort-03 113 and rw-rl g2-box 56. In 1696 the offset changes over time
   (up and to the left early on, gone at frame 2255 after a light solve). Ruled out:
   - waist mapping: head and arms both hang on `body_link5`, so the waist does not move one
     against the other;
   - recorded poses: the arm end poses equal FK on `arm_base_link` (0.0 mm), and the recorded
     head camera is 1.4-2.3 cm / 0.5-2.3° from FK in every episode, 1696 included;
   - head joints: flipping the pan sign makes 22777 worse, flipping the roll sign does not fix 1696, and a tilt
     offset fixes one moment but not the others;
   - state lag: ±20 and ±60 frames make no difference.
   Not fixed: no single priors or description value fits these episodes. The likely cause is
   something the record does not capture in these sessions, such as the arms deflecting while
   pushing the cart. A per-episode solve of the head cameras (light route, scope `episode`)
   moves top_head by 7 cm / 3° in 1696 but cannot follow a change within the episode.
