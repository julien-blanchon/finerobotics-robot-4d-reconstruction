# Code style & readability policy

> Agreed 2026-11-12. Adapted from a previous project's policy.

Every Python file in `src/` and `tests/` follows these conventions. When in doubt, match the
surrounding code. Tooling (ruff, pyrefly) enforces formatting, imports, and types; this document
covers what tooling cannot.

---

## 1. Typing: static shapes with pyrefly, never a bare `Tensor`

We use pyrefly's native tensor-shape types (`pyrefly-torch-stubs`, `pyrefly-einops-stubs`). Shapes
are checked statically, at zero runtime cost.

- **Every public function/method signature** carries shaped tensors: `Tensor[[B, N, 3]]`.
  Bare `Tensor` only for a genuinely rank-unknown value, with a comment.
- **Dimensions are PEP 695 type parameters** bounded by `IntVar`, declared on the function or
  class that owns them. Constructor sizes become class parameters bound with `Int[...]`;
  per-call sizes (batch, queries, frames) are method parameters.

  ```python
  class CrossAttention[D: IntVar, NHead: IntVar](nn.Module):
      def __init__(self, dim: Int[D], num_heads: Int[NHead]) -> None: ...

      def forward[B: IntVar, Q: IntVar](self, query: Tensor[[B, Q, D]], ...) -> Tensor[[B, Q, D]]:
  ```
- **One parameter per independent dimension**; derived sizes are expressions (`D // NHead`,
  `3 * D`), never fresh names.
- **Variadic leading dims** for geometry helpers that broadcast: bound the parameter with
  `IntTuple` and unpack it, `def compose[Bs: IntTuple](a: Tensor[[*Bs, 4, 4]], ...)`.
- **`assert_type` at shape-changing steps** (reshape, attention head split/merge, gather)
  as free, statically-checked documentation. Not on every line.
- **Gaps are contained, not hidden.** When pyrefly cannot follow (data-dependent lengths from
  boolean masks, `N * (X // N)`, dynamic `ModuleList`s), use a precise
  `cast("Tensor[[B, N, 3]]", ...)` at the boundary, or
  `# type: ignore[pyrefly:<exact-code>]  # <why>`. Never a blanket ignore.
- **Imports for shape typing** live under `TYPE_CHECKING`; with `from __future__ import
  annotations`, `shape_extensions` stays a dev-only dependency:

  ```python
  from __future__ import annotations
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from shape_extensions import Int, IntTuple, IntVar
  ```

- **Modern generics everywhere**: `def first[T](xs: Sequence[T]) -> T`, `type Alias = ...`,
  `Self`, `@override`, `Literal[...]` for small closed sets. No `TypeVar(...)`, no `Optional`,
  no `typing.List`.

- **Generic classes always carry their parameters**, in signatures, returns, fields and
  locals. Sizes passed as arguments are `Int[X]` so they bind a parameter, and a result's
  dimensions are expressions of the inputs' ones:

  ```python
  def per_frame[N: IntVar, F: IntVar](xy: Tensor[[N, 2]], frames: Int[F], ...) -> Queries[1, F * N]
  def slice[Start: IntVar, Stop: IntVar](self, start: Int[Start], stop: Int[Stop]) -> Queries[B, Stop - Start]
  ```

  `int` in a shape means "data-dependent" (a mask, a file's length), never "not written yet".
  Where pyrefly has no form for a size (`ceil(N / stride)`), keep `int` with a one-line comment.

**Axis glossary.** Single-letter (or short PascalCase) dimension names mean the same thing in
every file. Defined once in `src/r4r/types.py`; add a name there before using it.

```text
Bs leading batch dims        B  batch                  S  images in an encoding
N  points / pixels           Q  queries                P  patch tokens per frame
D  model width               NHead, HD heads, head width
H, W image height/width      C  cameras                T  robot timesteps
J  joints                    L  kinematic links        K  correspondences
M  observations             V  views (camera × time)  E  episodes
G  groups of episodes        F  frames in a query      NB free bodies
NQ qpos width                NState robot-state width  NWheel wheels of a drive
NPair image/motion pairs     In, Out, Hidden layer widths
R, Cl patch-grid rows/columns
Ho, Wo output height/width   Side square resize side   Radius correlation radius
Depth decoder blocks         Crop query RGB crop side  Start, Stop slice bounds
NRef reference frames per target (keypoint transfer)
```

**Named fixed-size axes.** A fixed axis is named by its components, so the name shows the size:
`XYZ`, `XYZW`, `UV` (pixels), `XY`, `XYTheta`, `WXYZ` (quaternion), `VxVyVzWxWyWz` (se(3)
tangent), `Mat3x3`, `Mat4x4`, `FxFyCxCy`, `K1K2K3K4` (lens), `RGB`, all in `r4r/types.py`.
They are unpacked into the shape: `Tensor[[N, *XYZ]]`, `Tensor[[C, *Mat4x4]]`.

- Pyrefly 1.4 rejects `type XYZ = Literal[3]` (and `TypeAlias`, `Final`, `Int[3]`) as a
  dimension; it accepts an `IntTuple[3]` alias unpacked. A shape holds at most one unpacked
  part, so next to `*Bs` write the literal: `Tensor[[*Bs, 3, 3]]` (the geometry helpers).
- `Scalar` is `Tensor[[]]`: losses, priors, reduced statistics.
- No aliases for whole tensors (`type Points = Tensor[...]`): they hide the shape at the use
  site. Frozen dataclasses carry the semantics (`Predictions.points: Tensor[[B, Q, *XYZ]]`).

## 2. Geometry conventions (single source of truth)

- **Frames are named `a_from_b`.** A 4×4 `world_from_camera` maps camera coordinates to
  world: `x_world = world_from_camera @ x_camera`. Composition reads left to right:
  `world_from_camera = world_from_hand @ hand_from_camera`. Never `T_wc`, `pose`, `extrinsics`
  without direction.
- **Rigid transforms are 4×4 matrices** in tensors `[*Bs, 4, 4]`. Quaternions and axis-angle
  appear only at I/O boundaries (data loaders) and inside `exp`/`log`.
- **Cameras use OpenCV convention**: x right, y down, z forward. Pixel centres at integer
  coordinates (`x = u * W - 0.5`).
- **Units: metres and radians** everywhere behind the data boundary. Unscaled reconstruction
  units are explicit in names (`points_unscaled`, `scale`).

## 3. einops for every reshape

Use `rearrange` / `reduce` / `repeat` / `einops.einsum` with named axes for reshapes,
reductions, broadcasts, and contractions. Do not use `.view` / `.reshape` / `.permute` /
`.transpose` / `.flatten` / `torch.einsum`. The pyrefly einops stubs check the pattern
against the input shape. Indexing, masking, gather, and matmul (`@`) are fine.

Casing: shape types **UPPERCASE** (`Tensor[[B, Q, D]]`); einops patterns **lowercase**
(`"b q (h d) -> b h q d"`).

Exception: `@` on `[*Bs, 4, 4]` / `[*Bs, 3, 3]` transforms and `x[..., :3, :3]` / `x[..., :3, 3]`
slicing on poses are allowed. They are clearer than einsum here.

## 4. Comments and docstrings

- A short comment labels each logical step, plus a *why* on anything non-obvious (numerical
  tricks, stop-grad, a convention imposed by pretrained weights). Not line-by-line.
- Module and public function docstrings: a one-line imperative summary, an optional short
  why/how paragraph, and a paper reference on its own line (`Ref: Calib3R arXiv:2509.08813, Eq. 9`).
  No `Args:`/`Returns:` boilerplate unless a parameter is genuinely non-obvious.
- **Math in docstrings** where the code implements a model or an estimator: the formula in
  compact plain-text/Unicode, in backticks, then one line on meaning and units. Use the code's
  names or the paper's symbols, not both.

  ```python
  """Pixel of a camera-frame point: `p = f · d(s) · (x, y)/r + c`, `s = r/z` or `θ`.

  `d` is the radial lens polynomial; `p` in pixels, `x` in metres.
  """
  ```
- Ported code carries a source line: `# Adapted from <url> (<license>)`.
- Separate logical steps with a single blank line; no `# ---- banners ----`.

## 5. Minimalism and repo organisation

- **100 % live code.** Every module is reachable from an entrypoint (CLI, public API, test).
  No dead branches, no commented-out alternatives, no `if use_old:`. Git is the archive.
- **One path per subsystem.** A config field changes a *quantity* (sizes, weights, steps),
  never selects between code paths. Changing an approach means replacing it.
- **Experiments are guests.** Investigations live in `research/` (memo + throwaway script) and
  end *adopted* (replacing the old path, in the same change) or *abandoned* (deleted, memo
  says why).
- **~300 lines soft ceiling per module**; one concept per file; layout below. New files need
  a new *concept*, not a new *variant*.
- **Evaluation exception**: competing methods under active evaluation (currently the three
  calibration methods) may coexist, each in its own module, until the evaluation names a
  winner. The losers are then deleted.
- **`research/` is local and not committed**: memos and throwaway scripts. Durable
  explanations go to `docs/`.
- **Limit generalisation.** Hard-code until something must vary. A little duplication beats
  a premature abstraction. Composition over inheritance; a `Protocol` only with ≥ 2 real
  implementers.
- **Few dependencies.** Runtime: `torch`, `einops`, `safetensors`, `huggingface-hub`, `av`,
  `tyro`, optionally `opencv-python-headless`. Anything else needs a reason written in the
  PR. Optional groups for boundaries: `lerobot` (dataset loading), dev/typing tools, test-only
  oracles.
- **One canonical form per boundary.** External data (LeRobot v3 / finerobotics delivery) is
  converted once, in one loader module, into our dataclasses. Nothing behind it knows the
  source format. Dataset-specific mess lives in that loader only.

### Repository layout

Two pillars over shared foundations. Dependencies point one way:
`geometry, data → arrow → rago → reconstruction/io/metrics/viz → cli`. `arrow` never knows a robot
exists; `rago` consumes ARROW outputs through plain dataclasses.

```text
src/r4r/
  __init__.py, types.py, cli.py         public API, axis glossary, tyro entrypoint
  geometry/   se3.py camera.py alignment.py              pure math, no I/O
  data/       episode.py lerobot.py images.py            the one canonical input form
  arrow/      ARROW: visual reconstruction (weight-compatible reimplementation)
    models/   arrow.py encoder.py decoder.py query_embed.py
      layers/ attention.py block.py rope.py swiglu_ffn.py
    queries.py tasks.py
  rago/       RAGO: robot-aware geometric optimisation
    robot.py kinematics.py meshes.py observations.py initialize.py losses.py
    soft.py hard.py hybrid.py solve.py
  reconstruction.py io.py metrics.py
  viz/        rerun.py (optional `viz` group)
tests/        mirrors src/r4r/
docs/         usage, FAQ, results, technical overview with figures (SVG)
```

## 6. House rules

- **Configuration**: `@dataclass(frozen=True, slots=True, kw_only=True)` configs, nested per
  subsystem. CLI via `tyro`, reading the same dataclasses (one subcommand per pipeline stage).
- **Data carriers**: frozen slotted dataclasses (`Episode`, `Encoding`, `Queries`,
  `Predictions`, `Calibration`). No dicts of tensors crossing module boundaries.
- **Naming**: descriptive (`world_from_camera`, `point_confidence`). Carve-outs: shape
  unpacking (`b, q, d = x.shape`), glossary dims, standard math symbols next to a paper ref
  (`R`, `t`, `K`, `λ` → `scale`).
- **Constants**: tunables are config fields; fixed constants (ImageNet mean, patch size) are
  module-level `UPPER_CASE`. No bare magic numbers in logic.
- **Validation**: trust types and einops. `assert` with a message only at a genuinely
  ambiguous edge (user input, file boundary). Validate at the point of use.
- **Device/dtype agnostic**: new tensors inherit `device`/`dtype` from inputs; never `.cuda()`.
  Non-learned tensors are `register_buffer`s.
- **Autograd context**: `@torch.inference_mode()` for encoding/decoding;
  Calib3R optimisation runs with grad on its own small parameter set.
- **Logging**: no `print` in `src/` except the CLI entrypoint; `logging` with %-style lazy
  args.
- **`__init__.py`**: re-exports only (the public Python API) plus `__all__`.

## 7. PyTorch performance conventions

- bf16 `torch.autocast` for the ARROW encoder/decoder trunk; numerically sensitive parts
  (heads, Procrustes/SVD, Calib3R optimisation, FK) in fp32 (SVD in fp64).
- `F.scaled_dot_product_attention` only; never a hand-rolled softmax attention.
- Cache everything that depends only on the encoding (per-block decoder K/V) once per encoding.
- No host–device syncs in hot loops (`.item()`, `.cpu()`, `print(tensor)`); gather on device,
  transfer once.
- `torch.compile` regionally (the repeated block), `fullgraph=True` in development; static
  shapes where possible (pad query chunks to a fixed size).
- Weights load with `safetensors` only (no pickle). `torch.device("meta")` construction +
  `load_state_dict(assign=True)` to avoid double allocation.

## 8. Tooling (delegated, configured in `pyproject.toml`)

- `ruff format` (line length 100) + `ruff check` with
  `E, F, I, UP, B, SIM, C4, C90, RUF, PIE, PERF, FURB, PTH, RET, ANN`; `required-imports =
  ["from __future__ import annotations"]`.
- `pyrefly check` in `default` mode with the torch + einops shape stubs; zero errors is the
  bar.
- `pytest` for behaviour: numerical parity with the reference ARROW, FK vs MuJoCo, SE(3)
  identities, Calib3R on synthetic data with known ground truth.
- Python 3.13 (PEP 695 generics, PEP 696 type-parameter defaults; matches pyrefly's shape corpus).
