"""Frame-rate planning helpers for interpolation.

The source cadence, generated cadence and delivery cadence are deliberately
separate.  This keeps the clip duration unchanged when a 24 fps source is
interpolated at 4x internally and delivered at (for example) 60 fps.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FPSPlan:
    source_fps: float
    requested_multiplier: int
    interpolation_multiplier: int
    generated_fps: float
    output_fps: float


def _next_power_of_two(value: float) -> int:
    result = 1
    while result < value:
        result *= 2
    return result


def build_fps_plan(source_fps: float, multiplier: int,
                   target_fps: float | None) -> FPSPlan:
    source_fps = float(source_fps)
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("입력 FPS는 0보다 커야 합니다.")
    if int(multiplier) != multiplier or int(multiplier) not in (2, 4, 8):
        raise ValueError("보간 배율은 2x, 4x, 8x 중 하나여야 합니다.")
    multiplier = int(multiplier)
    if target_fps is not None:
        target_fps = float(target_fps)
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("최종 출력 FPS는 0보다 커야 합니다.")

    required = multiplier
    if target_fps is not None and target_fps > source_fps * required:
        required = _next_power_of_two(target_fps / source_fps)
    if required > 8:
        raise ValueError("현재 최대 내부 보간 배율은 8x입니다.")

    generated_fps = source_fps * required
    return FPSPlan(
        source_fps=source_fps,
        requested_multiplier=multiplier,
        interpolation_multiplier=required,
        generated_fps=generated_fps,
        output_fps=(target_fps if target_fps is not None else generated_fps),
    )


def output_frame_count(source_frame_count: int, source_fps: float,
                       output_fps: float) -> int:
    if source_frame_count <= 0:
        return 0
    duration = source_frame_count / float(source_fps)
    return max(1, int(math.floor(duration * float(output_fps) + 0.5)))


def selected_generated_indices(source_frame_count: int, source_fps: float,
                               interpolation_multiplier: int,
                               output_fps: float) -> list[int]:
    """Map CFR output timestamps to nearest internally generated frames."""
    count = output_frame_count(source_frame_count, source_fps, output_fps)
    if count <= 0:
        return []
    last = max(0, source_frame_count * interpolation_multiplier - 1)
    generated_fps = float(source_fps) * interpolation_multiplier
    indices = [
        min(last, int(round(i * generated_fps / float(output_fps))))
        for i in range(count)
    ]
    # Oversampling is guaranteed by build_fps_plan, therefore duplicates
    # would indicate an invalid caller plan rather than something to hide.
    if any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("출력 FPS가 내부 생성 FPS보다 높습니다.")
    return indices
