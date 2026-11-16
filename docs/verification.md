# End-to-end verification of the ready datasets (2026-11-16)

Every dataset whose onboarding status row starts with "ready" (plus the ready stations of
abc-130k, AgiBot G1 and openneodata arx5/flexiv) was run end to end on a random sample, from
the delivered root to a merged twin. Nothing here is a full-dataset run.

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
| agibot-world g2-cohort-01 | 516 | fast | 0.6 / 0 | 10 fast | none | — | ok. EE: the delivered description draws the OmniPicker jaws across the wrist camera's side (camera mount at −8 cm y, jaws on y); corrected description `agibot__g2` (jaws on x) and preset `omnipicker_g2`: jaw points on both jaws in the hand views. top_head draws the TCPs ~10 cm below the grippers in episode 1696 (camera, not EE) |
| droid | 512 | 95.6k fast (checked), **8 full** | 66 / 0.02 | 3 fast, 7 light (flagged) | none | — | ok |
| robointer-droid | 512 | as droid, 8 full | 66 / 0.02 | 7 fast, 3 light | none | — | ok |
| galaxea-open-world r1-lite | 484 | fast | 0.3 / 0 | 10 fast | none | — | fingers on the wrist fingers; base motion not checkable on video (fixed 2026-11-16). EE (`galaxea_r1_lite`) |
| molmobot rby1-door-opening | 426 | fast | 0.7 / 0 | 10 fast | none | cameras 0.02-0.03 cm, 0.01° | ok |
| openneodata flexiv-rizon4 (1 station) | 412 | fast 444 / full 20.4k now | 0.4 / 0 | 10 fast | left_wrist (ignored) | — | ok |
| realsource-world | 341 | fast | 0.4 / 0 | 10 fast | none | — | meshes on the wrist grippers. EE (`rs02_tool`, 2 episodes): jaw points on the pad tips open and closed in the hand views |
| groot panda-omron-base | 273 | light | 100 / 0 | 10 light | none | — | ok |
| groot r1-pro (3 of 18 roots) | 254 / 20 / 10 | fast | ~0 | 10 fast each | none | — | arm meshes on the arms. EE (`galaxea_r1_pro`, root 88b8…, 2 episodes): jaw points on the other arm's fingertips seen from each wrist; own fingertips at the wrist image edges |
| agibot-world g2-cohort-02 / -03 | 198 / 6 | fast | ~0 | 10 fast | none | — | ok; EE as g2-cohort-01 (corrected description) |
| molmobot rby1-open / pick-10 / pick-15.15 | 189 / 143 / 55 | fast | ~0 | 10 fast | none | cameras ≤ 0.04 cm, ≤ 0.015°; EE 0.1-0.2 mm (30.3 mm, 180° convention) | ok |
| interndata-a1-sim genie-1 | 174 | fast | 0.5 / 0 | 10 fast | none | head 0.25 cm; wrists 0.2-1.1 cm (p95 3-4 cm, 4°); EE resid 1 mm, **p95 29 mm**; EE vs `states.*_tcp_to_robot_pose`: record TCP 76 mm further along z than ours (0.219 m from the gripper base, beyond the description's jaw tips), axes turned 90° about z, residual 1.1-1.6 mm median (p95 24-29 mm); aperture = 0.127 · `states.*_gripper.position` (1 = open) | ok; jaw points on the jaws |
| openneodata aloha | 157 | light (third view per episode) | 17 / 0 | 10 light | none | — | arms on the real ones within a few cm in the third view, right gripper up to ~5 cm off in one episode; wrist fingers approximate (fixed 2026-11-16) |
| interndata-a1-sim franka-b090d20bcf | 142 | fast | 0.5 / 0 | 10 fast | none | head 0.00 cm; hand 0.43 cm / 0.74°; EE resid 4.1 mm | ok |
| axis sim-sawyer-cohort-02 | 108 | fast | 0.1 / 0 | 10 fast | none | — | meshes on the arm. **No EE columns**: the description has no gripper (robosuite's Rethink gripper; state 7-8 unmapped) |
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
   (jaws on the base's x) and the `omnipicker_g2` preset.
6. **Light route on a loaded login node:** with default threading (~300 threads per process)
   10 light episodes took over 16 minutes; with `OMP_NUM_THREADS=8`, 150-170 s.
   **Fixed 2026-11-16:** `r4r reconstruct` sets 8 threads (torch, OpenCV, exported to the
   environment) unless `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` is set; 3 light
   yam episodes with no thread setting: 21 s.
7. Ground-truth outliers worth a look: genie-1 EE residual p95 29 mm and wrist cameras p95
   3-4 cm (median ~1 mm / 0.2-1 cm, so frames during fast motion); a1-sim wrists p95
   1-2 cm generally.
