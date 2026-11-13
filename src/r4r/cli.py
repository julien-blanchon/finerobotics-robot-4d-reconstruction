"""`r4r`: one subcommand per tool and pipeline stage.

Datasets are named by their delivery root, absolute or relative to `DELIVERY`
(e.g. `agibot-world/datasets/g2-cohort-01`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

DELIVERY = Path("/e/data1/datasets/playground/mmlaion/guybrush/delivery")


def dataset_root(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else DELIVERY / path


@dataclass(frozen=True, slots=True)
class Inspect:
    """Cameras, joint mapping and how the robot's bodies move in one episode."""

    dataset: tyro.conf.Positional[str]
    episode: int = 0
    stride: int = 1

    def run(self) -> None:
        from r4r.data.lerobot import load_episode
        from r4r.inspect import inspect_episode

        root = dataset_root(self.dataset)
        episode = load_episode(root, self.episode, self.stride, decode=())
        print(inspect_episode(episode, root).text())


@dataclass(frozen=True, slots=True)
class Links:
    """Evidence about which body each camera rides on (still share, rotation residuals)."""

    dataset: tyro.conf.Positional[str]
    episode: int = 0
    cameras: tuple[str, ...] = ()  # default: all
    # Focal length guess in pixels (searched ×0.2..1.5); default 0.7 × image width
    focal: float | None = None
    top: int = 4

    def run(self) -> None:
        import torch

        from r4r.data.lerobot import load_episode
        from r4r.rago.kinematics import forward_kinematics
        from r4r.rago.klt import KLTConfig, track_video
        from r4r.rago.links import body_evidence
        from r4r.rago.robot import load_robot, qpos_from_state

        root = dataset_root(self.dataset)
        episode = load_episode(root, self.episode, decode=self.cameras or None)
        robot = load_robot(episode.robot).to("cpu", torch.float64)
        world_from_body = forward_kinematics(
            robot, qpos_from_state(robot, episode.state, episode.joints)
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for camera in episode.cameras:
            if self.cameras and camera.name not in self.cameras:
                continue
            height, width = camera.frames.shape[-2:]
            focal = self.focal or 0.7 * width
            k = torch.tensor(
                [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]],
                dtype=torch.float64,
                device=device,
            )
            evidence = body_evidence(track_video(camera.frames, KLTConfig()), world_from_body, k)
            groups = "; ".join(
                f"{residual:.2f}° "
                + ", ".join(robot.body_names[b] for b in group[:3])
                + (f" (+{len(group) - 3})" if len(group) > 3 else "")
                for group, residual in zip(
                    evidence.groups[: self.top], evidence.residual, strict=False
                )
            )
            declared = (
                f"declared {'moving' if camera.moving else 'static'} on {camera.mount or '?'}"
            )
            print(f"{camera.name}: still share {evidence.still_share:.2f} ({declared})\n  {groups}")


@dataclass(frozen=True, slots=True)
class Fit:
    """Fit a moving camera's body (among candidates), mount and focal length.

    Several episodes are one setup: fitted one by one, their consensus kept, then pooled.
    """

    dataset: tyro.conf.Positional[str]
    camera: tyro.conf.Positional[str]
    episodes: tuple[int, ...] = (0,)
    # Candidate bodies (default: the camera's prior or declared link)
    bodies: tuple[str, ...] = ()
    # Focal length guess in pixels when the priors give none (searched ×0.2..1.5);
    # default 0.7 × image width
    focal: float | None = None
    # Video/joint-state offsets to test (frames), on top of the priors' state_shift
    shifts: tuple[int, ...] = (0,)
    # Store the result in the camera's priors (mount, focal, ranges from episode agreement)
    write: bool = False

    def run(self) -> None:
        import math
        from dataclasses import replace

        import torch

        from r4r.data.lerobot import load_episode
        from r4r.data.priors import write_camera
        from r4r.geometry import se3
        from r4r.rago.calibrate import (
            FitConfig,
            MountFit,
            fit_moving_camera,
            fit_setup,
            refine,
            video_tracks,
        )
        from r4r.rago.robot import CameraMount, body_mounts, load_robot, qpos_from_state

        root = dataset_root(self.dataset)
        loaded = [load_episode(root, e, decode=(self.camera,)) for e in self.episodes]
        stream = next(c for c in loaded[0].cameras if c.name == self.camera)
        bodies = self.bodies or ((stream.mount,) if stream.mount else ())
        assert bodies, f"{self.camera}: no candidate body (give --bodies)"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mount = CameraMount.from_prior(self.camera, stream.prior, bodies[0], moving=True)
        free = body_mounts(loaded[0].priors)
        robot = load_robot(loaded[0].robot, cameras=(mount,), bodies=free).to(device, torch.float64)
        height, width = stream.frames.shape[-2:]
        focal = stream.prior.focal or self.focal or 0.7 * width
        k = torch.tensor(
            [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]],
            dtype=torch.float64,
            device=device,
        )
        config = replace(FitConfig(), shifts=self.shifts)
        episodes = []
        for e in loaded:
            frames = next(c for c in e.cameras if c.name == self.camera).frames
            qpos = qpos_from_state(robot, e.state.to(device), e.joints)
            episodes.append((qpos, video_tracks(frames, k, config.klt)))
        ids = tuple(robot.body(b) for b in bodies)
        focal_range = stream.prior.focal_range if stream.prior.focal else None

        def describe(fit: MountFit) -> str:
            t = [round(v, 4) for v in fit.body_from_camera[:3, 3].tolist()]
            axis = [round(v, 3) for v in fit.body_from_camera[:3, 2].tolist()]
            quat = [
                round(v, 4) for v in se3.matrix_to_quaternion(fit.body_from_camera[:3, :3]).tolist()
            ]
            return (
                f"on {robot.body_names[fit.body]}: position {t} m, optical axis {axis}, "
                f"quaternion {quat}, focal {float(fit.intrinsics[0, 0]):.1f} px; "
                f"score {fit.score:.3f}, median {fit.median_error:.2f} px, "
                f"inliers {fit.inlier_fraction:.0%}"
            )

        spread = (0.005, 1.0)  # ranges written for a single episode (m, deg): indicative
        if mount.body_from_camera is not None:
            # Known mount (priors): refine within its ranges, no search
            start = robot.body_from_camera[robot.camera(self.camera)]
            body = robot.camera_body[robot.camera(self.camera)]
            fit = refine(robot, episodes, self.camera, body, start, k, config, k[0, 0], focal_range)
        elif len(episodes) == 1:
            fit = fit_moving_camera(robot, episodes, self.camera, ids, k, config, focal_range)
            for b, score in zip(bodies, fit.body_scores.tolist(), strict=True):
                print(f"  search score on {b}: {score:.3f}")
        else:
            setup = fit_setup(robot, episodes, self.camera, ids, k, config, focal_range)
            for e, single, agree in zip(
                self.episodes, setup.episodes, setup.agree.tolist(), strict=True
            ):
                print(f"  episode {e} {'agrees' if agree else 'DISAGREES'}: {describe(single)}")
            fit = setup.fit
            deviation = [
                se3.inverse(fit.body_from_camera) @ single.body_from_camera
                for single, agree in zip(setup.episodes, setup.agree.tolist(), strict=True)
                if agree and single.body == fit.body
            ]
            if len(deviation) > 1:
                d = torch.stack(deviation)
                spread = (
                    float(d[:, :3, 3].norm(dim=-1).max()),
                    math.degrees(float(se3.so3_log(d[:, :3, :3]).norm(dim=-1).max())),
                )
        print(f"{self.camera} {describe(fit)}")
        if len(self.shifts) > 1:
            scores = ", ".join(
                f"{s:+d}: {v:.3f}"
                for s, v in zip(self.shifts, fit.shift_scores.tolist(), strict=True)
            )
            print(
                f"  state shift scores ({scores}): best {fit.shift:+d} -> set state_shift in priors"
            )
        if self.write:
            pose = fit.body_from_camera.cpu()
            values = {
                "mount": "moving",
                "link": robot.body_names[fit.body],
                "position": [round(v, 5) for v in pose[:3, 3].tolist()],
                "orientation": [
                    round(v, 6) for v in se3.matrix_to_quaternion(pose[:3, :3]).tolist()
                ],
                "position_range": round(max(spread[0], 0.002), 4),
                "orientation_range_deg": round(max(spread[1], 0.2), 3),
                "focal": round(float(fit.intrinsics[0, 0]), 2),
                "focal_range": focal_range if focal_range is not None else 0.0,
                "fitted": {
                    "episodes": list(self.episodes),
                    "median_px": round(fit.median_error, 3),
                    "score": round(fit.score, 4),
                },
            }
            print(f"wrote {write_camera(root, self.camera, values)}")


@dataclass(frozen=True, slots=True)
class Rig:
    """Calibrate a whole rig in stages: moving mounts, then free bodies (arm bases) and static
    cameras over pooled episodes, checked by agreement between groups of episodes.

    Cameras, links, focals and free bodies come from the priors (see
    docs/dataset-onboarding.md, step 4).
    """

    dataset: tyro.conf.Positional[str]
    episodes: tuple[int, ...] = (0, 1, 2, 3)
    # Groups for the agreement check (episodes dealt round-robin)
    groups: int = 2
    # Frame stride when decoding (1: every frame)
    stride: int = 1
    # RoMa links between cameras every this many (strided) timesteps
    every: int = 10
    # Focal guess (× image width) for cameras the priors give none
    focal: float = 0.7
    # Store fitted mounts and focals in the camera priors
    write: bool = False

    def run(self) -> None:
        import math
        from dataclasses import replace

        import torch

        from r4r.data.lerobot import load_episode
        from r4r.data.priors import write_camera
        from r4r.geometry import se3
        from r4r.matching import RoMaV2
        from r4r.rago.robot import CameraMount, body_mounts, load_robot, qpos_from_state
        from r4r.rago.stages import RigConfig, RigEpisode, calibrate_rig, spread

        root = dataset_root(self.dataset)
        loaded = [load_episode(root, e, stride=self.stride) for e in self.episodes]
        streams = loaded[0].cameras
        names = tuple(c.name for c in streams)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mounts = tuple(CameraMount.from_prior(c.name, c.prior, c.mount, c.moving) for c in streams)
        bodies = body_mounts(loaded[0].priors)
        robot = load_robot(loaded[0].robot, cameras=mounts, bodies=bodies)
        robot = robot.to(device, torch.float64)
        k = []
        for c in streams:
            height, width = c.frames.shape[-2:]
            f = c.prior.focal or self.focal * width
            k.append([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]])
        intrinsics = torch.tensor(k, dtype=torch.float64, device=device)
        episodes = [
            RigEpisode(
                qpos_from_state(robot, e.state.to(device), e.joints),
                {c.name: c.frames for c in e.cameras},
            )
            for e in loaded
        ]
        known_focal = tuple(c.prior.focal is not None for c in streams)
        known_mount = tuple(m.body_from_camera is not None for m in mounts)
        model = RoMaV2.from_checkpoint(device=device)
        config = replace(RigConfig(), link_every=self.every, groups=self.groups)
        fit = calibrate_rig(
            robot, episodes, names, intrinsics, known_focal, known_mount, model, config
        )

        print(f"median {fit.median_error:.2f} px, inliers {fit.inlier_fraction:.0%}")
        for c, stream in enumerate(streams):
            pose = fit.mounts[c].cpu()
            where = robot.body_names[robot.camera_body[robot.camera(stream.name)]]
            print(
                f"{stream.name} on {where}: position "
                f"{[round(v, 4) for v in pose[:3, 3].tolist()]} m, "
                f"focal {float(fit.intrinsics[c, 0, 0]):.1f} px"
            )
        free = sorted(robot.free_bodies, key=robot.body)
        for b, (name, (cm, deg)) in enumerate(
            zip(free, spread(fit.group_corrections), strict=True)
        ):
            offset = fit.group_corrections[:, b].mean(0)
            print(
                f"{name}: offset {[round(v, 4) for v in offset[:3, 3].tolist()]} m, "
                f"{math.degrees(float(se3.so3_log(offset[:3, :3]).norm())):.2f} deg; "
                f"groups disagree by {cm:.1f} cm / {deg:.2f} deg"
            )
        if self.write:
            for c, stream in enumerate(streams):
                values: dict[str, object] = {
                    "focal": round(float(fit.intrinsics[c, 0, 0]), 2),
                    "focal_range": stream.prior.focal_range if known_focal[c] else 0.0,
                }
                if stream.moving and not known_mount[c]:
                    pose = fit.mounts[c].cpu()
                    values |= {
                        "mount": "moving",
                        "link": robot.body_names[robot.camera_body[robot.camera(stream.name)]],
                        "position": [round(v, 5) for v in pose[:3, 3].tolist()],
                        "orientation": [
                            round(v, 6) for v in se3.matrix_to_quaternion(pose[:3, :3]).tolist()
                        ],
                    }
                print(f"wrote {write_camera(root, stream.name, values)}")


@dataclass(frozen=True, slots=True)
class Sheet:
    """Contact sheet: camera frames next to the robot skeleton (top, side, front views)."""

    dataset: tyro.conf.Positional[str]
    output: tyro.conf.Positional[Path]
    episode: int = 0
    moments: int = 4

    def run(self) -> None:
        import cv2
        import torch

        from r4r.data.lerobot import load_episode
        from r4r.rago.robot import load_robot
        from r4r.viz.sheet import contact_sheet

        episode = load_episode(dataset_root(self.dataset), self.episode)
        robot = load_robot(episode.robot).to("cpu", torch.float64)
        image = contact_sheet(episode, robot, self.moments)
        cv2.imwrite(str(self.output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"wrote {self.output} ({image.shape[1]}x{image.shape[0]})")


@dataclass(frozen=True, slots=True)
class Export:
    """The robot description (MJCF) with every camera whose mount the priors know placed."""

    dataset: tyro.conf.Positional[str]
    output: tyro.conf.Positional[Path]

    def run(self) -> None:
        from r4r.data.lerobot import load_episode
        from r4r.rago.robot import CameraMount, body_mounts, describe

        episode = load_episode(dataset_root(self.dataset), 0, decode=())
        mounts = tuple(
            CameraMount.from_prior(c.name, c.prior, c.mount, c.moving)
            for c in episode.cameras
            if c.prior.position is not None and c.prior.orientation is not None
        )
        self.output.write_text(describe(episode.robot, mounts, body_mounts(episode.priors)))
        print(f"wrote {self.output} ({len(mounts)} cameras placed)")


type Command = (
    Annotated[Inspect, tyro.conf.subcommand("inspect")]
    | Annotated[Links, tyro.conf.subcommand("links")]
    | Annotated[Fit, tyro.conf.subcommand("fit")]
    | Annotated[Rig, tyro.conf.subcommand("rig")]
    | Annotated[Sheet, tyro.conf.subcommand("sheet")]
    | Annotated[Export, tyro.conf.subcommand("export")]
)


def main() -> None:
    tyro.cli(Command).run()
