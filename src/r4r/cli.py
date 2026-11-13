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
    # Focal length guess in pixels (searched ×0.5..1.5); default 0.7 × image width
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
    """Fit a moving camera's body (among candidates), mount and focal length on one episode."""

    dataset: tyro.conf.Positional[str]
    camera: tyro.conf.Positional[str]
    episode: int = 0
    # Candidate bodies (default: the camera's prior or declared link)
    bodies: tuple[str, ...] = ()
    # Focal length guess in pixels (searched ×0.5..1.5); default 0.7 × image width
    focal: float | None = None

    def run(self) -> None:
        import torch

        from r4r.data.lerobot import load_episode
        from r4r.geometry import se3
        from r4r.rago.calibrate import FitConfig, fit_moving_camera
        from r4r.rago.robot import CameraMount, load_robot, qpos_from_state

        root = dataset_root(self.dataset)
        episode = load_episode(root, self.episode, decode=(self.camera,))
        stream = next(c for c in episode.cameras if c.name == self.camera)
        bodies = self.bodies or ((stream.mount,) if stream.mount else ())
        assert bodies, f"{self.camera}: no candidate body (give --bodies)"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mount = CameraMount(self.camera, bodies[0], 0.05, 0.5)
        robot = load_robot(episode.robot, cameras=(mount,)).to(device, torch.float64)
        qpos = qpos_from_state(robot, episode.state.to(device), episode.joints)
        height, width = stream.frames.shape[-2:]
        focal = self.focal or 0.7 * width
        k = torch.tensor(
            [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]],
            dtype=torch.float64,
            device=device,
        )
        ids = tuple(robot.body(b) for b in bodies)
        fit = fit_moving_camera(robot, qpos, stream.frames, self.camera, ids, k, FitConfig())
        for b, score in zip(bodies, fit.body_scores.tolist(), strict=True):
            print(f"  search score on {b}: {score:.3f}")
        t = fit.body_from_camera[:3, 3].tolist()
        axis = fit.body_from_camera[:3, 2].tolist()
        quat = se3.matrix_to_quaternion(fit.body_from_camera[:3, :3]).tolist()
        body = robot.body_names[fit.body]
        print(
            f"{self.camera} on {body}: position {[round(v, 4) for v in t]} m, "
            f"optical axis {[round(v, 3) for v in axis]}, "
            f"quaternion {[round(v, 4) for v in quat]}\n"
            f"  focal {float(fit.intrinsics[0, 0]):.1f} px; score {fit.score:.3f}, "
            f"median {fit.median_error:.2f} px, inliers {fit.inlier_fraction:.0%}"
        )


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


type Command = (
    Annotated[Inspect, tyro.conf.subcommand("inspect")]
    | Annotated[Links, tyro.conf.subcommand("links")]
    | Annotated[Fit, tyro.conf.subcommand("fit")]
    | Annotated[Sheet, tyro.conf.subcommand("sheet")]
)


def main() -> None:
    tyro.cli(Command).run()
