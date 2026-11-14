# Related work: what to adopt for r4r

Survey of 2026-11-14 (papers, code, datasets; nothing installed, run or downloaded). It is read
against `docs/architecture.md`, `docs/dataset-onboarding.md`, `research/pipeline.md` and the
earlier review `research/external_review.md`, which already covers CalibAll, DepthWorld /
DROID-3D, RoMa v2 and COLMAP in depth. Those are only summarised here.

Conventions:

- **[claim]** means the paper or README states it. None of these numbers were reproduced.
- **[ours]** is a measurement already in `research/pipeline.md`.
- **[infer]** is my judgement, not measured.
- **NC** means a non-commercial licence (code or weights, as stated).
- **Links:** arXiv and code links were opened or seen in search results on 2026-11-14.
  - "no code found" means none was found, not that none exists.
  - **UNVERIFIED** marks what could not be confirmed.
  - The classic papers (1989–2012) are cited by DOI or venue and were not re-opened.
- **Verdicts:** **adopt** (put into the pipeline), **test** (measure on our validation set
  first), **cite** (prior art, nothing to take), **no** (not relevant for us).

Our pain points, numbered as in the verdicts:

| # | pain point |
|---|---|
| P1 | focal unknown |
| P2 | wrist/head mount search fragile from one camera |
| P3 | video/joint-state time offsets |
| P4 | mobile bases without odometry |
| P5 | fisheye |
| P6 | calibration compute |
| P7 | validation without ground truth |

## What we should try first

Ranked by expected gain for the pipeline per unit of effort [infer].

1. **A second-order solver for the kinematic bundle** (P6, P7).
   - Replace Adam (800 iterations, cosine schedule, `rago/bundle.py`) with Gauss-Newton or
     Levenberg-Marquardt.
     - Points can be eliminated by Schur complement or re-triangulated (variable projection).
     - The free parameters are few: mounts, focals, bodies.
     - Every BA system in this survey works this way: COLMAP/Ceres, DROID-SLAM's dense BA,
       DROID-3D's Schur LM (~12.5 min per robot group over thousands of episodes [claim]).
   - Expected outcomes:
     - Iterations drop by one to two orders of magnitude [infer].
     - The Gauss-Newton Hessian gives a covariance per unknown for free. That measures "poorly
       determined" directly; today only the agreement between groups shows it.
   - Options: a small hand-written batched LM in torch (`torch.func.jacrev` on a few dozen
     parameters), PyPose (Apache-2.0, sparse LM since v0.9.5 [claim]), or Theseus (MIT).
   - **Complements:** the batched search, which can keep Adam; the full bundle is what changes.
2. **Continuous time offset and joint-encoder offsets as bundle unknowns** (P3, P2).
   - Today `state_shift` is an integer number of frames, chosen by a grid in `r4r fit --shifts`.
   - Prior art treats the offset as a continuous parameter, with the trajectory interpolated
     in time: Kalibr, VINS online temporal calibration, Furrer et al. 2017, and SyncHeC
     (Nov 2026, the only markerless robot-arm one).
   - DROID-3D adds 7 per-robot joint offsets. Removing them raised its wrist↔exterior error
     from 1.03 to 1.21 px; removing the hand-eye correction raised it to 2.07 px [claim].
   - For us:
     - `dt` at dataset scope: FK of states interpolated at `t + dt`. Cheap with our own FK.
     - `δq` as dataset-scope priors with a range.
   - **Replaces:** the integer shift grid.
3. **Lens models in the bundle, KLT and RoMa lookups, initialised by AnyCalib** (P5, P1).
   - Lens models:
     - Kannala-Brandt (OpenCV fisheye), UCM/EUCM or Double Sphere for the 14 fisheye roots
       (7,882 h).
     - A `k1`/`k2` radial term for real "pinhole" cameras. DROID-3D ships 12 distortion
       coefficients per ZED camera.
   - AnyCalib (ICCV 2025, Apache-2.0 code and weights) is the only permissive learned
     calibrator that outputs these parametric models and a principal point. Use it as the
     initial value only; the bundle refines.
   - **Unblocks** the fisheye roots. The lens models are in (`r4r.geometry.camera`:
     radial pinhole and Kannala-Brandt, refined by the bundle); AnyCalib is not wired yet.
4. **More references on DROID to bound reference error** (P7).
   - Besides DROID's shipped calibration and DROID-3D (already in hand), there are two more:
     - **PointWorld-DROID:** 42,935 episodes with `optimized_extrinsics`, NVIDIA research
       licence.
     - **DROID's own post-hoc calibration** (DROID paper, Appendix G; files at
       `KarlP/droid`): ~36k scenes.
   - These use different methods. Where they agree with each other we have a reference; where
     they disagree, that spread is the floor of what we can claim.
   - **Complements:** sim ground truth and the agreement between groups.
5. **Geometric focal solvers and two learned calibrators in the existing focal benchmark**
   (P1).
   - PoseLib (BSD-3) solvers: P4Pf for static cameras once 2D–3D pairs exist, 6-point
     shared-focal and monodepth-aided shared-focal relative pose for static↔static pairs.
   - Add AnyCalib and GeoCalib (`shared_intrinsics=True`) to `research/intrinsics/bench.py`.
   - Expected gain:
     - Modest for pinhole cameras: pooled ARROW is already 3.4 % [ours].
     - The real target is the static-camera dolly-zoom problem. Two static cameras with
       overlap give a shared-focal constraint that the wrist's near-planar points do not.
6. **MapAnything (Apache weights variant) for the dense pass, conditioned on our K and poses**
   (Stage 4, P6).
   - It takes intrinsics, poses and an `is_metric_scale` flag as inputs and outputs metric
     points [claim]. That is what ARROW cannot do: pose conditioning copied the draft [ours].
   - Caveats:
     - "Does not capture dynamic motion" [claim]: the arm needs masking or a separate
       treatment.
     - Pinhole only.
   - Alternative per-frame cheap depth, with scale fitted to our sparse points: DA3-Metric-Large
     (Apache) or Video-Depth-Anything Small (Apache).
   - **Complements or replaces:** ARROW in Stage 4. Measure against ARROW on cost and on
     agreement with our triangulated points.
7. **A permissive learned tracker for robot surfaces and long-gap static tracks** (P2,
   static-camera robot tracks).
   - Candidates: TAPNext / BootsTAPIR (Apache-2.0 code and weights) and AllTracker (MIT code).
   - KLT drifts on robot surfaces (45 % dropped at ground truth [ours]).
   - No published benchmark resolves sub-pixel error. TAP-Vid scores at 256², where 1 px is
     about 2–3 native pixels at 480p. Our ground-truth Sampson diagnostic is the only valid
     test.
   - **Complements:** KLT stays for scene tracks.

Low priority: kinematic-inertial base odometry (contact-aided InEKF), and only where a dataset
records an IMU (galaxea r1-lite has `observation.state.chassis.imu`). See §7.

### Is our approach missing something important?

- **Solver and uncertainty.** The bundle is first-order. That is the most likely single cause
  of "hours per setup", and the reason uncertainty only comes from splitting episodes into
  groups. A covariance per unknown would also feed the `r4r report` validation directly.
- **Lens and timing models are simpler than the field's standard.**
  - Lens: radial pinhole and Kannala-Brandt only (no tangential, rational or Double Sphere
    terms).
  - Time offset: integer frames only.
  - Joint-encoder offsets are not modelled.
  - Rolling shutter is not modelled either. I found no robot-calibration paper that models it,
    so its effect on fast wrist motion at our resolutions is unknown and would need a check
    [infer].
- **The closest competitors all use the robot mesh for their final accuracy.** CalibAll
  (render-and-compare), DROID-3D (URDF renders matched to images), PointWorld (robot depth vs
  mesh), DROID's own post-hoc calibration (rendered silhouettes), EasyHeC and SyncHeC all do.
  Our mesh-free stance is a real difference, and it costs us on static-only rigs: 11,470 h, of
  which syngrasp is 11,292 h with ground truth. Nothing mesh-free in the literature solves
  static-only rigs better than our robot-track bundle [infer from this survey].
- **Wrist-camera hand-eye from monocular video at scale.** Nothing found does this across
  heterogeneous datasets. Calib3R and Bi-JCR do it for one rig from MASt3R-style geometry,
  with no released code. Kalib and CalibAll do static cameras only.
- **Base motion.** Most legged/humanoid estimators need an IMU. Base motion from joint states
  alone, plus foot contacts, is a weaker and less-studied problem (§7).

---

## 1. Camera-to-robot pose and hand-eye calibration

### Markerless, robot as the target (2020–2026)

Almost all of these need either per-robot training or mesh rendering. Both are excluded for us:
meshes are often wrong, and 97 descriptions would each need a model.

- **DREAM**: "Camera-to-Robot Pose Estimation from a Single Image". ICRA 2020.
  - [arXiv 1911.09231](https://arxiv.org/abs/1911.09231) ·
    [code](https://github.com/NVlabs/DREAM) (NVIDIA Source Code License, NC).
  - Learned 2D robot keypoints plus PnP with known joint angles. Trained per robot on
    synthetic data. Its dataset is the standard benchmark.
  - **no**: per-robot training. DREAM data could only serve as a Franka test set.
- **CtRNet**: "Markerless Camera-to-Robot Pose Estimation via Self-supervised Sim-to-Real
  Transfer". CVPR 2023.
  - [arXiv 2302.14332](https://arxiv.org/abs/2302.14332) ·
    [code](https://github.com/ucsdarclab/CtRNet-robot-pose-estimation) (MIT).
  - Keypoints plus PnP, self-supervised against rendered silhouettes. Per robot.
  - **no**: mesh rendering and per-robot training.
- **CtRNet-X**: "Camera-to-Robot Pose Estimation in Real-world Conditions Using a Single
  Camera". arXiv 2024 (venue UNVERIFIED).
  - [arXiv 2409.10441](https://arxiv.org/abs/2409.10441) · no code found.
  - Handles a partially visible robot by choosing keypoints on links a VLM detects. A tuned
    version contributed to DROID's post-hoc calibration (~12k scenes [claim]).
  - **no** as a method. Its DROID outputs are part of reference 4 in the list above.
- **EasyHeC**: "Accurate and Automatic Hand-eye Calibration via Differentiable Rendering and
  Space Exploration". RA-L 2023.
  - [arXiv 2305.01191](https://arxiv.org/abs/2305.01191) ·
    [code](https://github.com/ootts/EasyHeC) (MIT).
  - Mask-vs-render optimisation, with active exploration of joint space.
- **EasyHeC++**: IROS 2024.
  - [arXiv 2410.09293](https://arxiv.org/abs/2410.09293) · code "coming soon".
  - Pretrained-model initialisation, then rendering.
  - **no** for both: rendering, and active robot motion (offline data has none).
- **RoboPose**: "Single-View Robot Pose and Joint Angle Estimation via Render & Compare".
  CVPR 2021.
  - [arXiv 2104.09359](https://arxiv.org/abs/2104.09359) ·
    [code](https://github.com/ylabbe/robopose) (MIT).
  - **no**: render-and-compare, per robot.
- **Real-time Holistic Robot Pose Estimation with Unknown States**: ECCV 2024.
  - [arXiv 2402.05655](https://arxiv.org/abs/2402.05655) ·
    [code](https://github.com/Oliverbansk/Holistic-Robot-Pose-Estimation) (no licence stated).
- **RoboKeyGen**: ICRA 2024.
  - [arXiv 2403.18259](https://arxiv.org/abs/2403.18259) ·
    [repo](https://github.com/Nimolty/RoboKeyGen) (code not released per its checklist).
- **RoboPEPP**: CVPR 2025.
  - [arXiv 2411.17662](https://arxiv.org/abs/2411.17662) ·
    [code](https://github.com/raktimgg/RoboPEPP) (no licence file).
  - **no** for these three: per-robot networks that estimate joint angles we already record.
- **Kalib**: "Markerless Hand-Eye Calibration with Keypoint Tracking". v2 title "Easy Hand-Eye
  Calibration with Reference Point Tracking"; IROS 2025 per DOI (UNVERIFIED).
  - [arXiv 2408.10562](https://arxiv.org/abs/2408.10562) ·
    [code](https://github.com/robotflow-initiative/Kalib) (no licence file).
  - Tracks one predefined reference point on the robot (SpatialTracker), gets its 3D position
    from FK, then PnP. Needs no mesh and no training.
  - DROID reprojected-mask IoU 0.80, vs EasyHeC 0.77 and marker-based 0.87 [claim].
  - **cite.** It is the mesh-free baseline closest to our static-camera path. Our track model
    is a superset: many link-attached points with free positions instead of one known point.
    Its tracker, SpatialTracker, is CC BY-NC.
- **CalibAll / "Unify Robot Actions in Camera Frame"**: arXiv 2025, v2 May 2026.
  - [arXiv 2511.17001](https://arxiv.org/abs/2511.17001) ·
    [code](https://github.com/SII-dannyXSC/CalibAll) (MIT).
  - CoTracker3 point plus FK, temporal PnP, then mask render refinement.
  - Ran on 16 datasets (~97k episodes). Tracking and mask failures were fixed by hand.
  - DROID: 14/20 scenes succeed, 3.9° / 4.4 cm [claim].
  - **cite.** Closest project in spirit. Details in `research/external_review.md` §1.
- **FEEPE**: "Foundation Feature-Driven Online End-Effector Pose Estimation". arXiv 2025.
  - [arXiv 2503.14051](https://arxiv.org/abs/2503.14051) · no code found.
  - DINOv2 matches against the end-effector CAD model.
  - **no**: needs a mesh.
- **ARC-Calib**: arXiv 2025.
  - [arXiv 2503.14701](https://arxiv.org/abs/2503.14701) · no code found.
  - Exploratory robot motions with coplanarity/collinearity constraints.
  - **no**: needs active motion.
- **Hydra**: "Marker-Free RGB-D Hand-Eye Calibration". arXiv 2025.
  - [arXiv 2504.20584](https://arxiv.org/abs/2504.20584) (seen in search only).
  - **no**: mesh plus depth.
- "RoboCalib" and "robot pose from SAM2": no matching papers found.

### Hand-eye from scene structure (no robot model)

These are our family: the moving camera observes the static scene, and FK gives the hand
motion.

- **Classic AX = XB**:
  - Tsai & Lenz, IEEE T-RA 1989 (doi:10.1109/70.34770).
  - Park & Martin, IEEE T-RA 1994.
  - Daniilidis, dual quaternions, IJRR 1999 (doi:10.1177/02783649922066213).
  - **cite.** Our search is a robust replacement for these closed forms, which break with
    noisy relative motions.
- **Hand-eye from SfM up to scale**:
  - Andreff, Horaud & Espiau, IJRR 2001 (re-posted as
    [arXiv 2311.11808](https://arxiv.org/abs/2311.11808)): linear, recovers the SfM scale,
    and analyses degenerate motions.
  - Heller et al., CVPR 2011 (L∞; doi:10.1109/CVPR.2011.5995629).
  - **cite.** Andreff's degeneracy analysis (which motions leave which mount components
    undetermined) is useful for window selection, where today we use whole episodes.
- **Koide & Menegatti**, "General Hand-Eye Calibration Based on Reprojection Error
  Minimization". RA-L/ICRA 2019.
  - **cite**: reprojection-based like ours.
- **Calib3R**: arXiv 2025, v2 Aug 2026.
  - [arXiv 2509.08813](https://arxiv.org/abs/2509.08813) · code "upon acceptance".
  - MASt3R pointmaps plus robot poses: joint hand-eye, scale and scene.
  - 0.4–1.7 cm and 0.011–0.023 rad on 5 datasets [claim].
  - **cite.** We built on it and moved to KLT plus a bundle because ARROW's wrist matches were
    8–30 px off [ours]. Notes in `research/calib3r.md`.
- **Bi-JCR**: "Bi-Manual Joint Camera Calibration and Scene Representation". ICRA 2026.
  - [arXiv 2505.24819](https://arxiv.org/abs/2505.24819) · no code found.
  - 3D foundation-model correspondences give each wrist camera's hand-eye and the arm-to-arm
    pose jointly.
  - **cite.** It is our bimanual stage 2 with dense foundation geometry in place of KLT plus
    RoMa. Without code or numbers checked there is nothing to take; it confirms the design.

### Recalibration of large robot datasets

- **DROID post-hoc calibration**: Appendix G of the DROID paper
  ([arXiv 2403.12945](https://arxiv.org/abs/2403.12945)); files at
  [huggingface.co/KarlP/droid](https://huggingface.co/KarlP/droid).
  - Camera-to-base from SAM masks vs PyTorch3D silhouettes (~30k scenes) plus tuned CtRNet-X
    (~12k). Camera-to-camera from a modified DUSt3R [claim].
  - **adopt as a reference** (try-first #4).
- **DepthWorld / DROID-3D**: CoRL 2026.
  - [arXiv 2610.08780](https://arxiv.org/abs/2610.08780) ·
    [data](https://huggingface.co/datasets/jaibrdhn/droid_3d_extrinsics) (calibration CC BY 4.0,
    pseudo-labels NC, gated) · repo `Jai2500/depthworld` (MIT, calibration optimiser not
    released).
  - Factor graph per robot serial: shared hand-eye plus joint offsets, per-episode
    extrinsics, stereo depth, URDF-render matches.
  - **adopt as a reference** (already downloaded). **Adopt its ideas**: joint offsets, a
    shared per-setup bundle with Schur LM, and residuals whitened by RoMa precision.
- **PointWorld**: CVPR 2026.
  - [arXiv 2601.03782](https://arxiv.org/abs/2601.03782) ·
    [code](https://github.com/NVlabs/PointWorld) (Apache-2.0) ·
    [data](https://huggingface.co/datasets/nvidia/PointWorld-DROID) (NVIDIA licence, research
    use; 42,935 episodes with `optimized_extrinsics` for successful optimisations).
  - FoundationStereo depth, VGGT initialisation, then refinement by aligning robot depth to
    the mesh.
  - Median 1.8 cm / 1.9° against a proxy reference [claim].
  - **adopt as a reference** (try-first #4). Its method is mesh-based.
- **3DROID**: arXiv Oct 2026.
  - [arXiv 2610.01744](https://arxiv.org/abs/2610.01744) ·
    [data](https://huggingface.co/datasets/wonguen/3DROID).
  - Reuses PointWorld extrinsics, filtered by a stereo-depth check.
  - **test** as a filter: its kept/rejected flag is a cheap quality label for references.
- **yangyi02/droid**: a [community repo](https://github.com/yangyi02/droid) refining DROID
  extrinsics against PyBullet renders and depth. No licence, no numbers.
  - **no.**
- **Bridge / Open X-Embodiment.** No general recalibration was found.
  - ECoT ([arXiv 2407.08693](https://arxiv.org/abs/2407.08693),
    [code](https://github.com/MichalZawalski/embodied-CoT), MIT) fits a RANSAC projection
    matrix per Bridge trajectory, from OWLv2/SAM gripper detections to proprioceptive 3D.
  - SIMPLER ([arXiv 2405.05941](https://arxiv.org/abs/2405.05941),
    [code](https://github.com/simpler-env/SimplerEnv)) hand-tuned the camera poses for Bridge
    WidowX and Google robot.
  - H-VLA ([arXiv 2609.22895](https://arxiv.org/abs/2609.22895), no code found) uses PnP from
    hand-annotated gripper pixels plus FK.
  - **cite.** Gripper-detection PnP is a cheap first guess for a static camera when nothing
    else exists. Our robot-track bundle already does better [infer].

## 2. Kinematic and temporal calibration

- **Kalibr** (Furgale, Rehder & Siegwart): "Unified temporal and spatial calibration for
  multi-sensor systems". IROS 2013 (doi:10.1109/IROS.2013.6696514) ·
  [toolbox](https://github.com/ethz-asl/kalibr) (BSD-style).
  - Continuous-time trajectory (B-splines) with a time offset estimated jointly. Needs
    calibration targets.
  - **adopt the idea** (try-first #2). Not the tool.
- **Online temporal calibration for monocular VIO** (Qin & Shen): IROS 2018.
  - [arXiv 1808.00692](https://arxiv.org/abs/1808.00692) · code in VINS-Mono.
  - The time offset is a state. Feature observations are shifted along their image velocity,
    so it is cheap.
  - **adopt the idea.** The equivalent for us is shifting the FK pose by `dt` × its
    finite-difference velocity, or interpolating joint states.
- **Furrer et al.**, "Evaluation of Combined Time-Offset Estimation and Hand-Eye Calibration on
  Robotic Datasets". FSR 2017 · [code](https://github.com/ethz-asl/hand_eye_calibration)
  (BSD-3).
  - Aligns pose streams in time, then hand-eye.
  - **cite.**
- **Koide & Menegatti**, "General Robot-Camera Synchronization Based on Reprojection Error
  Minimization". 2019 (checkerboard).
  - Found ~56 ms latency in one setup [claim, from search].
  - **cite**: the order of magnitude matters. At 15–30 fps that is 1–2 frames, as in
    interndata and abc-130k [ours].
- **SyncHeC**: "Markerless Spatio-Temporal Hand-Eye Calibration via Differentiable Rendering
  and Continuous-Time Joint Optimization". arXiv 2611.02999, Nov 2026 ·
  [code](https://github.com/hongzhengdong/SyncHeC) (academic/NC).
  - Continuous-time FK makes latency differentiable. Joint extrinsics plus latency, 2.7 ms
    latency error [claim].
  - Needs meshes, metric depth, SAM2 masks and a planned excitation trajectory.
  - **cite; adopt the continuous-time FK idea**, not the rendering.
- **Pradeep, Konolige & Berger**, "Calibrating a Multi-arm Multi-sensor Robot: A Bundle
  Adjustment Approach". ISER 2010 (doi:10.1007/978-3-642-28572-1_15).
  - Joint offsets and sensor poses in one BA (PR2).
  - **cite**: the classic precursor of our kinematic bundle with `δq`.
- **Vicente et al.**, "Markerless Eye-Hand Kinematic Calibration on the iCub Humanoid Robot".
  Frontiers 2018.
  - Joint offsets from stereo.
  - **cite.**
- **Dr. Robot**: "Differentiable Robot Rendering". arXiv 2024.
  - [arXiv 2410.13851](https://arxiv.org/abs/2410.13851) ·
    [code](https://github.com/cvlab-columbia/drrobot) (no licence shown).
  - Per-robot Gaussian splats trained on URDF renders.
  - **no**: rendering, per robot.
- No 2024–2026 work was found that estimates joint offsets or link lengths from monocular
  video without markers or meshes. DROID-3D estimates joint offsets with stereo plus renders.

## 3. SLAM, SfM and 4D reconstruction

Common pattern [infer]: none uses robot kinematics. Most give scale-free or approximately
metric poses, and most are pinhole only. Their role for us is at most initialisation, dense
geometry, or a cross-check. FK already gives metric poses of every moving camera once the
mount is known.

| method | venue | links | licence | metric? | intrinsics | known K / poses in | dynamic | fisheye | verdict |
|---|---|---|---|---|---|---|---|---|---|
| DROID-SLAM | NeurIPS 2021 | [2108.10869](https://arxiv.org/abs/2108.10869) · [code](https://github.com/princeton-vl/DROID-SLAM) | BSD-3 | stereo/RGB-D only | required | K required | no | no (OpenCV radial only) | **no** for arm cameras |
| DPVO / DPV-SLAM | NeurIPS 2023 / ECCV 2024 | [2208.04726](https://arxiv.org/abs/2208.04726), [2408.01654](https://arxiv.org/abs/2408.01654) · [code](https://github.com/princeton-vl/DPVO) | MIT | no | required | K | no | not stated | **no** (VO dropped from scope) |
| MegaSaM | CVPR 2025 | [2412.04463](https://arxiv.org/abs/2412.04463) · [code](https://github.com/mega-sam/mega-sam) | Apache-2.0 (deps: UniDepth NC) | no (Sim(3)-evaluated) | one shared focal, optimised | K optional | yes | no ("strong radial distortion" unsupported [claim]) | **no**: FK gives poses; focal from our bundle is better posed |
| MASt3R-SLAM | CVPR 2025 | [2412.12392](https://arxiv.org/abs/2412.12392) · [code](https://github.com/rmurai0610/MASt3R-SLAM) | CC BY-NC-SA | no | generic central camera | K optional | no | plausible, not shown | **no**: NC, static scenes |
| DUSt3R / MASt3R | CVPR 2024 / ECCV 2024 | [2312.14132](https://arxiv.org/abs/2312.14132), [2406.09756](https://arxiv.org/abs/2406.09756) · [dust3r](https://github.com/naver/dust3r), [mast3r](https://github.com/naver/mast3r) | CC BY-NC-SA, restrictive checkpoints | MASt3R "metric" | from alignment | UNVERIFIED | no | no | **no** (licence; superseded by RoMa v2 for matching [ours]) |
| MonST3R | ICLR 2025 | [2410.03825](https://arxiv.org/abs/2410.03825) · [code](https://github.com/Junyi42/monst3r) | CC BY-NC-SA | no | per frame | no | yes | no | **no** |
| CUT3R | CVPR 2025 | [2501.12387](https://arxiv.org/abs/2501.12387) · [code](https://github.com/CUT3R/CUT3R) | CC BY-NC-SA | claimed metric | UNVERIFIED | UNVERIFIED | yes | no | **no** (licence) |
| Spann3R | 3DV 2025 | [2408.16061](https://arxiv.org/abs/2408.16061) · [code](https://github.com/HengyiWang/spann3r) | CC BY-NC-SA | UNVERIFIED | — | — | no | no | **no** |
| VGGT | CVPR 2025 | [2503.11651](https://arxiv.org/abs/2503.11651) · [code](https://github.com/facebookresearch/vggt) | VGGT licence; 1B weights NC, gated commercial variant | no | per view | no | no | no | **no**: focal 15 % / 36 % [ours]; ARROW is better |
| VGGT-Long / StreamVGGT / VGGT-SLAM (2.0) | ICRA 2026 / ICLR 2026 / NeurIPS 2025, RSS 2026 | [2507.16443](https://arxiv.org/abs/2507.16443), [2507.11539](https://arxiv.org/abs/2507.11539), [2505.12549](https://arxiv.org/abs/2505.12549), [2601.19887](https://arxiv.org/abs/2601.19887) | VGGT's / CC BY-NC-SA / BSD-2 | backbone-dependent | per view | no | no | no | **no** |
| VGGT-X | arXiv 2025 | [2509.25191](https://arxiv.org/abs/2509.25191) | UNVERIFIED | no | — | no | no | no | **no** |
| π³ / Pi3X | ICLR 2026 | [2507.13347](https://arxiv.org/abs/2507.13347) · [code](https://github.com/yyfz/Pi3) | code BSD-3, **weights NC** | Pi3X approx. metric | — | Pi3X: poses, K, depth | no | no | **no** (weights NC) |
| **MapAnything** | 3DV 2026 | [2509.13414](https://arxiv.org/abs/2509.13414) · [code](https://github.com/facebookresearch/map-anything) | code Apache-2.0; `map-anything-apache` weights Apache, default weights NC | yes [claim] | pinhole out | **K or rays, poses, depth, metric flag** | **no** [claim] | via rays, not trained [claim] | **test** for Stage 4 (try-first #6) |
| Depth Anything 3 (ARROW's base) | arXiv 2025 | [2511.10647](https://arxiv.org/abs/2511.10647) · [code](https://github.com/ByteDance-Seed/Depth-Anything-3) | code Apache; Giant/Large/Nested **NC**; Base/Small/Metric-Large/Mono-Large Apache | Nested and Metric-Large (needs focal) | per view | poses + FOV via a camera token [claim] | not a design goal | no | **in use** via ARROW. Check which DA3 weights ARROW derives from (NC risk) |
| Pow3R | CVPR 2025 | [2503.17316](https://arxiv.org/abs/2503.17316) · [code](https://github.com/naver/pow3r) | NAVER NC | — | — | K, relative pose, depth | no | no | **no** (NC) |
| HunyuanWorld-Mirror (WorldMirror) | arXiv 2025 | [2510.10726](https://arxiv.org/abs/2510.10726) · [code](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) | Tencent licence **excludes EU/UK/KR** | — | — | poses, K, depth | — | — | **no**: not licensed in Germany |
| Aether | ICCV 2025 | [2503.18945](https://arxiv.org/abs/2503.18945) · [code](https://github.com/OpenRobotLab/Aether) | code MIT, weights UNVERIFIED | — | — | raymaps (generation) | yes | no | **no** (world model) |
| Any4D | arXiv Dec 2025 | [2512.10935](https://arxiv.org/abs/2512.10935) · no code found | — | yes [claim] | yes | RGB-D, IMU, not K or poses | yes | — | **cite**; revisit if code appears |
| Align3R / Uni4D / Shape of Motion | CVPR 2025 / CVPR 2025 / ICCV 2025 | [2412.03079](https://arxiv.org/abs/2412.03079), [2503.21761](https://arxiv.org/abs/2503.21761), [2407.13764](https://arxiv.org/abs/2407.13764) | NC / MIT / MIT | no | from priors | — | yes | no | **no**: per-video optimisation, slower than our pass and not metric |
| Stereo4D | CVPR 2025 | [2412.09621](https://arxiv.org/abs/2412.09621) · [code](https://github.com/Stereo4D/stereo4d-code) | code UNVERIFIED; data CC BY 4.0 | from stereo | rectified 60° pinhole | — | yes | VR180 input rectified | **no** (data pipeline for VR180) |
| St4RTrack | ICCV 2025 | [2504.13152](https://arxiv.org/abs/2504.13152) · [code](https://github.com/HavenFeng/St4RTrack) | NC | no | — | — | yes | no | **no** |
| COLMAP rigs / GLOMAP | — / ECCV 2024 | [colmap rigs](https://colmap.github.io/rigs.html), [2407.20219](https://arxiv.org/abs/2407.20219) | BSD-3 | no | refined, many models incl. fisheye | rigs, fixed `sensor_from_rig` | no | yes | **test** as an independent BA cross-check and as a source of camera-model formulas (`external_review.md` §3) |

Robot-specific SLAM with kinematics (cite only):

- ARM-SLAM (Klingensmith et al. 2016): RGB-D SLAM in joint-angle space.
- KO-Fusion (ICRA 2019): RGB-D fusion tightly coupled with arm kinematics and wheel odometry.
- Climbing-robot gripper camera ([arXiv 2511.05816](https://arxiv.org/abs/2511.05816), 2025):
  limb FK plus monocular SLAM in a factor graph for metric scale; simulation only.

None handles heterogeneous datasets, and none estimates the mount from video.

## 4. Monocular and video depth

Our need is a cheap dense pass with scale fitted to our sparse metric points. Metric output
and predicted intrinsics are secondary because we have both [infer].

| model | venue | links | licence | metric | intrinsics | known K in | video | verdict |
|---|---|---|---|---|---|---|---|---|
| Depth Pro | arXiv 2024 (ICLR 2025 UNVERIFIED) | [2410.02073](https://arxiv.org/abs/2410.02073) · [code](https://github.com/apple/ml-depth-pro) | weights `apple-amlr` (research only, per secondary sources) | yes | focal | `f_px` (UNVERIFIED) | no | **no** (licence; not in our focal benchmark) |
| UniDepth / UniDepthV2 | CVPR 2024 / TPAMI 2025 | [2403.18913](https://arxiv.org/abs/2403.18913), [2502.20110](https://arxiv.org/abs/2502.20110) · [code](https://github.com/lpiccinelli-eth/UniDepth) | CC BY-NC | yes | K (pinhole, Fisheye624) | yes | no | **no**: focal 34 % / 27 % [ours], NC |
| MoGe / MoGe-2 / MoGe-3 | CVPR 2025 / 2025 / 2026 | [2410.19115](https://arxiv.org/abs/2410.19115), [2507.02546](https://arxiv.org/abs/2507.02546), [2607.17967](https://arxiv.org/abs/2607.17967) · [code](https://github.com/microsoft/MoGe) | code MIT, weights not stated | MoGe-2/3 yes | FOV | FOV | no | **no** for focal (20 % / 53 % [ours]); possible cheap per-frame depth if weight licence is clarified |
| Video Depth Anything (+ metric) | CVPR 2025 | [2501.12375](https://arxiv.org/abs/2501.12375) · [code](https://github.com/DepthAnything/Video-Depth-Anything) | code Apache; Small Apache, Base/Large NC; metric weights not stated | metric variants | no | no | yes, long videos | **test** Small as the cheap dense pass (7.5 ms per 518² frame on A100 [claim]) |
| DepthCrafter | CVPR 2025 | [2409.02095](https://arxiv.org/abs/2409.02095) · [code](https://github.com/Tencent/DepthCrafter) | Tencent research only | relative | no | no | yes | **no** (slow diffusion, licence) |
| Metric3D v2 | TPAMI 2024 | [2404.15506](https://arxiv.org/abs/2404.15506) · [code](https://github.com/YvanYin/Metric3D) | code BSD-2, weights not stated | yes (with K) | no | yes | no | **test** only if DA3-Metric fails; depends on focal we supply |
| Depth Anything V2 (metric) | NeurIPS 2024 | [code](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) | Small Apache, Base/Large NC | yes (indoor/outdoor heads) | no | no | no | **no** (superseded by DA3) |
| DA3-Metric-Large | 2025 | see DA3 above | Apache | yes, `focal × out / 300` | — | needs focal | no | **test** with VDA-Small (try-first #6) |
| Prompt Depth Anything | CVPR 2025 | [code](https://github.com/DepthAnything/PromptDA) (arXiv id UNVERIFIED) | code Apache | yes, needs a LiDAR depth prompt | no | — | no | **no** (no depth sensor) |
| Depth Any Camera | CVPR 2025 | [2501.02464](https://arxiv.org/abs/2501.02464) · [code](https://github.com/yuliangguo/depth_any_camera) | code MIT, weights not stated | yes | no, needs calibration | yes, fisheye/360 | no | **test** for fisheye dense depth once lens models exist |
| UniK3D | CVPR 2025 | [2503.16591](https://arxiv.org/abs/2503.16591) · [code](https://github.com/lpiccinelli-eth/UniK3D) | CC BY-NC | yes | dense ray field (any camera) | yes | no | **no** (NC); AnyCalib covers the calibration use |
| FlashDepth | ICCV 2025 | [2504.07093](https://arxiv.org/abs/2504.07093) · [code](https://github.com/Eyeline-Research/FlashDepth) | code Apache | not stated | no | no | yes, streaming | **no** (2K target; we are 180–480p) |
| RollingDepth | CVPR 2025 | [2411.19189](https://arxiv.org/abs/2411.19189) · [code](https://github.com/prs-eth/RollingDepth) | code Apache, weights RAIL++-M | likely affine (UNVERIFIED) | no | no | yes | **no** (diffusion cost) |
| Calibration Tokens / DEX | 2025 / 2026 | [2508.04928](https://arxiv.org/abs/2508.04928), [2608.27860](https://arxiv.org/abs/2608.27860) | not checked | — | — | — | — | **cite** (fisheye adapters for depth models) |

## 5. Point tracking

Key fact for us: TAP-Vid's δ^x_avg averages thresholds of 1–16 px at 256×256 resolution
([TAP-Vid, arXiv 2211.03726](https://arxiv.org/abs/2211.03726)). No tracker paper reports a
sub-pixel score. No paper was found comparing learned trackers with KLT for SfM or BA accuracy.
Our KLT numbers (0.19–0.64 px Sampson at ground truth [ours]) cannot be compared with any
published number. A learned tracker has to be measured with our own ground-truth diagnostic.

| tracker | venue | links | licence | notes [claim] | verdict |
|---|---|---|---|---|---|
| CoTracker / CoTracker3 | ECCV 2024 / ICCV 2025 | [2307.07635](https://arxiv.org/abs/2307.07635), [2410.11831](https://arxiv.org/abs/2410.11831) · [code](https://github.com/facebookresearch/co-tracker) | **CC BY-NC** | CoTracker3 δ_avg DAVIS 76.9, RoboTAP 78.0 (offline) | **no** (licence) |
| TAPIR / BootsTAPIR | ICCV 2023 / ACCV 2024 | [2306.08637](https://arxiv.org/abs/2306.08637), [2402.00847](https://arxiv.org/abs/2402.00847) · [tapnet](https://github.com/google-deepmind/tapnet) | Apache-2.0, code and weights | BootsTAPIR AJ RoboTAP 69.2; best at 512² | **test** (try-first #7) |
| TAPNext | ICCV 2025 | [2504.05579](https://arxiv.org/abs/2504.05579) · tapnet | Apache-2.0, code and weights | AJ DAVIS 65.3, RoboTAP 64.1; causal | **test** |
| AllTracker | ICCV 2025 | [2506.07310](https://arxiv.org/abs/2506.07310) · [code](https://github.com/aharley/alltracker) | MIT code, weights not stated | dense all-pixel, 16M params, high resolution | **test**: dense long-gap static correspondences |
| Track-On / Track-On2 / Track-On-R | ICLR 2025 / TPAMI 2026 / CVPR 2026 | [2501.18487](https://arxiv.org/abs/2501.18487), [2509.19115](https://arxiv.org/abs/2509.19115), [2603.12217](https://arxiv.org/abs/2603.12217) · [code](https://github.com/gorkaydemir/track_on) | MIT; DINOv3 backbone separate | Track-On-R RoboTAP 82.6 (best seen) | **test** second choice (DINOv3 gating) |
| LocoTrack | ECCV 2024 | [2407.15420](https://arxiv.org/abs/2407.15420) · [code](https://github.com/cvlab-kaist/locotrack) | Apache-2.0 | ~6× faster than prior SOTA | **no** (superseded) |
| TAPTR v1–v3 | ECCV 2024 – ICLR 2026 | [2403.13042](https://arxiv.org/abs/2403.13042), [2411.18671](https://arxiv.org/abs/2411.18671) · [code](https://github.com/IDEA-Research/TAPTR) | IDEA licence (terms unchecked) | — | **no** |
| PIPs / PIPs++ | ECCV 2022 / ICCV 2023 | [2204.04153](https://arxiv.org/abs/2204.04153), [2307.15055](https://arxiv.org/abs/2307.15055) · [pips2](https://github.com/aharley/pips2) | MIT | median L2 6.9 px on DAVIS | **no** |
| SpatialTracker / V2 | CVPR 2024 / ICCV 2025 | [2404.04319](https://arxiv.org/abs/2404.04319), [2507.12462](https://arxiv.org/abs/2507.12462) · [V2 code](https://github.com/henry123-boy/SpaTrackerV2) | **CC BY-NC** | 3D tracks plus poses | **no** (licence); used by Kalib and Kinema4D |
| TAPIP3D | NeurIPS 2025 | [2504.14717](https://arxiv.org/abs/2504.14717) · [code](https://github.com/zbw001/TAPIP3D) | code Apache; monocular mode runs MegaSaM + MoGe | world-frame 3D tracks given RGB-D, K and poses | **later**: Stage 4 3D tracks once our cameras and depth exist |
| DELTA | ICLR 2025 | [2410.24211](https://arxiv.org/abs/2410.24211) · [code](https://github.com/snap-research/DELTA_densetrack3d) | Snap NC | dense 3D | **no** |
| VGGSfM / Detector-Free SfM / MASt3R-SfM | CVPR 2024 / CVPR 2024 / 3DV 2025 | [2312.04563](https://arxiv.org/abs/2312.04563) · [vggsfm](https://github.com/facebookresearch/vggsfm), [dfsfm](https://github.com/zju3dv/DetectorFreeSfM) | CC BY-NC / custom / CC BY-NC-SA | tracking-based SfM | **cite** |
| RAFT / SEA-RAFT / MemFlow | ECCV 2020 / ECCV 2024 / CVPR 2024 | [2003.12039](https://arxiv.org/abs/2003.12039), [2405.14793](https://arxiv.org/abs/2405.14793), [2404.04808](https://arxiv.org/abs/2404.04808) | BSD-3 / BSD-3 / Apache | dense flow | **test** SEA-RAFT only if KLT survival on blurred wrist frames turns out to limit us |

## 6. Focal, intrinsics and distortion

Already measured [ours], median focal error, non-wrist / wrist:

| method | non-wrist | wrist |
|---|---|---|
| ARROW pooled | 3.4 % | 16 % |
| GeoCalib | 5.6 % | 29 % |
| VGGT | 15 % | 36 % |
| MoGe-2 | 20 % | 53 % |
| UniDepthV2 | 34 % | 27 % |

`research/intrinsics/estimators.py` has a GeoCalib shared-intrinsics mode. `pipeline.md` does
not record which variant produced the 5.6 %. Wrist focal comes from the motion fit; this
section is about static cameras and lenses.

- **AnyCalib**: "On-Manifold Learning for Model-Agnostic Single-View Camera Calibration".
  ICCV 2025.
  - [arXiv 2503.12701](https://arxiv.org/abs/2503.12701) ·
    [code](https://github.com/javrtg/AnyCalib) (Apache-2.0, code and weights).
  - Predicts per-pixel rays, then fits pinhole, radial 1–4, Kannala-Brandt 1–4, UCM, EUCM or
    division models, including the principal point.
  - Single image; no shared-intrinsics mode is documented, so we would take a median over
    frames.
  - **adopt as a fisheye initialiser** (try-first #3). **test** for pinhole in the benchmark.
  - How it would plug in (not installed; needs its own environment, e.g. a clone with
    `pip install -e .`, weights downloaded on first use):
    `AnyCalib(model_id="anycalib_dist").predict(image, cam_id="kb:4")` returns
    `intrinsics = [fx, fy, cx, cy, k1, k2, k3, k4]`, exactly our fisheye priors (`lens:
    fisheye`, `focal`, `distortion`); `cam_id="radial:3"` gives our pinhole `[k1, k2, k3]`.
    Predict on frames resized to the video resolution, take the median over frames, then
    let the bundle refine within a `distortion_range`.
- **GeoCalib**: ECCV 2024.
  - [arXiv 2409.06704](https://arxiv.org/abs/2409.06704) ·
    [code](https://github.com/cvg/GeoCalib) (code Apache-2.0, weights CC BY 4.0).
  - Pinhole, simple_radial, radial and simple_divisional models. Shared intrinsics across
    images. Principal point fixed at the centre.
  - **test** `shared_intrinsics=True` and the divisional model on fisheye roots, against
    AnyCalib.
- **Perspective Fields**: CVPR 2023.
  - [arXiv 2212.03239](https://arxiv.org/abs/2212.03239) ·
    [code](https://github.com/jinlinyi/PerspectiveFields) (licence not named).
  - Roll, pitch, vertical FOV; no distortion output.
  - **no.**
- **Tame a Wild Camera (WildCamera)**: NeurIPS 2023.
  - [arXiv 2306.10988](https://arxiv.org/abs/2306.10988) ·
    [code](https://github.com/ShngJZ/WildCamera) (Apache-2.0).
  - 4-DoF intrinsics; detects crops and resizes.
  - **test, low priority.** Crop/resize detection is relevant (abc-130k K at 848×480 vs video
    640×480 [ours]), but we already check that by hand from recorded K.
- **DeepCalib**: CVMP 2018 · [code](https://github.com/alexvbogdan/DeepCalib) (licence not
  stated).
  - Focal plus one distortion parameter.
  - **no** (superseded by AnyCalib).
- **DiffCalib**: AAAI 2025.
  - [arXiv 2405.15619](https://arxiv.org/abs/2405.15619) ·
    [code](https://github.com/zjutcvg/DiffCalib) (README says NC).
  - **no.**
- **Lopez et al.**, "Deep Single Image Camera Calibration with Radial Distortion". CVPR 2019.
  No code found.
  - **no.**
- **SimFIR**: ICCV 2023.
  - [arXiv 2308.09040](https://arxiv.org/abs/2308.09040) ·
    [code](https://github.com/fh2019ustc/SimFIR) (MIT).
  - Rectifies fisheye images, without parameters.
  - **no**: we need a model, not warped pixels.
- **AnyCam**: "Learning to Recover Camera Poses and Intrinsics from Casual Videos". CVPR 2025.
  - [arXiv 2503.23282](https://arxiv.org/abs/2503.23282) ·
    [code](https://github.com/Brummi/anycam) (MIT; depends on a UniDepth fork, NC).
  - **no** (effective NC; FK already gives poses).
- **CalibAnyView**: "Beyond Single-View Camera Calibration in the Wild". arXiv 2026.
  - [arXiv 2605.14615](https://arxiv.org/abs/2605.14615) · no code found.
  - Multi-view shared intrinsics with strong distortion: exactly the video-level fisheye case.
  - **watch**; test if code is released.
- **InFlux**: NeurIPS 2025 D&B, [arXiv 2510.23589](https://arxiv.org/abs/2510.23589).
  - Benchmark for video self-calibration with dynamic intrinsics; most baselines struggle
    [claim].
  - **cite.**
- **PoseLib**: [code](https://github.com/PoseLib/PoseLib) (BSD-3).
  - Minimal solvers: P4Pf, 6-point shared-focal relative pose, monodepth-aided shared- and
    varying-focal relative pose, Bougnoux and iterative focal from F, radial-distortion
    absolute and relative pose.
  - COLMAP-compatible camera models including OPENCV_FISHEYE.
  - **adopt** for static-camera focal (try-first #5), in place of the focal grid.
- **Kalibr** camera models ([code](https://github.com/ethz-asl/kalibr), BSD-style):
  pinhole/omni/Double Sphere/EUCM projections with radtan/equidistant distortion.
  - Needs targets.
  - **adopt the model formulas** (well documented, closed form).
- **Lens models**:
  - Kannala & Brandt, TPAMI 2006 (= OpenCV fisheye).
  - Mei & Rives, ICRA 2007 (UCM).
  - Khomutenko et al., RA-L 2016 (EUCM).
  - Usenko et al., "The Double Sphere Camera Model", 3DV 2018
    ([arXiv 1807.08957](https://arxiv.org/abs/1807.08957)).
  - **adopt one or two**, with closed-form unprojection for the bundle: Double Sphere and EUCM
    have it.
- **Classic self-calibration** (cite):
  - Hartley, "Self-calibration of stationary cameras", IJCV 1997 (rotation-only focal, our
    `research/intrinsics/rotation_focal.py`).
  - Kruppa equations (Faugeras, Luong & Maybank, ECCV 1992).
  - Bujnak, Kukelova & Pajdla, CVPR 2008 (P4Pf).
  - Larsson et al., "Revisiting Radial Distortion Absolute Pose", ICCV 2019.

## 7. Base and root trajectory from proprioception

Scope changed on 2026-11-14: visual odometry is dropped. Base motion comes from recorded
state and actions. This section lists estimators of a legged or humanoid base trajectory from
joint states, contacts and IMU.

Finding: every verified method needs an IMU. None estimates the base trajectory from joint
states and contacts alone.

- **Bloesch et al.**, "State Estimation for Legged Robots: Consistent Fusion of Leg Kinematics
  and IMU". RSS 2012 ([paper](https://www.roboticsproceedings.org/rss08/p03.html)).
  - EKF with footholds in the state.
  - No verified code.
  - **cite.**
- **Hartley et al.**, contact-aided invariant EKF. RSS 2018 and IJRR 2020.
  - [arXiv 1805.10410](https://arxiv.org/abs/1805.10410),
    [arXiv 1904.09251](https://arxiv.org/abs/1904.09251) ·
    [code](https://github.com/RossHartley/invariant-ekf) (BSD-3, C++).
  - Needs an IMU plus contact indicators.
  - **test, only for datasets with an IMU column** (galaxea r1-lite chassis IMU; check
    humanoid roots).
- **Lin et al.**, "Legged Robot State Estimation using Invariant Kalman Filtering and Learned
  Contact Events". CoRL 2021.
  - [arXiv 2106.15713](https://arxiv.org/abs/2106.15713) ·
    [code](https://github.com/UMich-CURLY/deep-contact-estimator) (no licence file).
  - Contacts are learned from proprioception.
  - **cite.** Contact inference from joint torques or positions matters when contacts are not
    recorded.
- **Pronto** (Camurri et al.): Frontiers in Robotics and AI 2020 ·
  [code](https://github.com/ori-drs/pronto) (LGPL-2.1).
  - IMU plus leg odometry, bipeds and quadrupeds.
  - **cite.**
- **Cerberus**: ICRA 2023.
  - [arXiv 2209.07654](https://arxiv.org/abs/2209.07654) ·
    [code](https://github.com/ShuoYangRobotics/Cerberus) (GPL-3.0).
  - Needs stereo, IMU and contacts.
  - **no.**
- **Learned leg-inertial odometry**:
  - Buchanan et al., CoRL 2021 ([arXiv 2111.00789](https://arxiv.org/abs/2111.00789)).
  - Legolas, CoRL 2024 (no code).
  - AutoOdom ([arXiv 2511.18857](https://arxiv.org/abs/2511.18857), humanoid, no code).
  - CoCo-InEKF (RSS 2026, code not checked).
  - **cite**: all need an IMU.

For our datasets [infer]:

- **Root pose recorded** (simulators, some humanoids): use it, as `base_odometry: pose`.
- **Legs and IMU recorded:** an InEKF could provide the starting trajectory, and the bundle
  refines it.
- **Legs only:** the trajectory is unobservable without stance assumptions (a stance foot is
  fixed in the world).
  - That gives a kinematic chain from the stance foot to the root.
  - Stance detection from foot height or velocity in FK is a heuristic.
  - With priors it could serve as an extra `base_odometry` kind, not yet supported.

## 8. Projects lifting robot datasets to 3D/4D at scale

How each got camera poses:

| project | data | camera poses from | mesh / render? | links | verdict |
|---|---|---|---|---|---|
| DROID post-hoc calibration | DROID ~36k scenes | SAM masks vs rendered silhouettes; CtRNet-X; DUSt3R for camera↔camera | yes | [2403.12945](https://arxiv.org/abs/2403.12945), [files](https://huggingface.co/KarlP/droid) | reference |
| DROID-3D (DepthWorld) | DROID 71,006 episodes | factor graph: stereo depth, RoMa v2, URDF-render matches, shared hand-eye + joint offsets | partly | [2610.08780](https://arxiv.org/abs/2610.08780) | reference + ideas |
| PointWorld | DROID 42,935 episodes | FoundationStereo + VGGT init + robot-depth-to-mesh alignment | yes | [2601.03782](https://arxiv.org/abs/2601.03782), [data](https://huggingface.co/datasets/nvidia/PointWorld-DROID) | reference |
| 3DROID | DROID subset | PointWorld extrinsics, stereo-depth filtered | inherited | [2610.01744](https://arxiv.org/abs/2610.01744) | filter |
| CalibAll | 16 datasets, ~97k episodes (RoboMIND, RDT, OXE) | point track + FK PnP, then mask rendering; MoGe for unknown K | yes | [2511.17001](https://arxiv.org/abs/2511.17001) | cite; compare on overlapping datasets if its outputs are released (only demo data found) |
| Kinema4D / Robo4D-200k | DROID, Bridge, RT-1 | SpatialTrackerV2 4D; robot registered via mesh reconstruction or manual base calibration | yes | [2603.16669](https://arxiv.org/abs/2603.16669) (code "upon acceptance") | cite |
| RynnWorld-4D | RoboMIND, RDT | Depth Anything 3 poses; no kinematic grounding | no | [2607.06559](https://arxiv.org/abs/2607.06559), [code](https://github.com/alibaba-damo-academy/RynnWorld-4D) | cite: DA3 poses alone are not robot-frame |
| RoGSW4RLD | DROID | kinematics + multi-camera 4D Gaussians; calibration source UNVERIFIED | — | [2609.35311](https://arxiv.org/abs/2609.35311) | cite |
| ECoT | Bridge V2 | per-trajectory RANSAC projection from gripper detections | no | [2407.08693](https://arxiv.org/abs/2407.08693) | cite |
| SIMPLER | Bridge, Google robot | hand-tuned; fSpy for K | no | [2405.05941](https://arxiv.org/abs/2405.05941) | cite |
| H-VLA | RoboCOIN, Agilex, Bridge, Fractal, DROID | PnP on annotated gripper pixels + FK; SIMPLER; DROID | no | [2609.22895](https://arxiv.org/abs/2609.22895) | cite |
| 3D-VLA, SpatialVLA, MolmoAct | OXE etc. | none: camera-frame monocular depth (ZoeDepth, Depth Anything V2) | no | [2403.09631](https://arxiv.org/abs/2403.09631), [2501.15830](https://arxiv.org/abs/2501.15830), [2508.07917](https://arxiv.org/abs/2508.07917) | no |
| Lift3D, SPA, 3D-MVP, 3D Diffuser Actor, Track2Act, AnyPos, HOMIE, RoboVerse, EgoDex, TraceVLA, HAMSTER | — | no pose estimation (simulated, already calibrated, 2D only, or device tracking) | — | — | no |

Takeaways [infer]:

- Every project that reports robot-frame camera poses on real data used either depth sensors
  (stereo) or robot meshes, and most used both. None works across heterogeneous embodiments
  with wrist, head and static cameras from monocular video.
- None models episode or frame scope beyond "per scene" plus "per robot serial" (DROID-3D).
  Our scope/range priors are more general than anything published.
- DROID is the only dataset with several independent references. Real-data validation of
  other kinds (bimanual, mobile, humanoid) must rely on sim ground truth, shipped poses
  (agibot, syngrasp), and agreement between groups.
