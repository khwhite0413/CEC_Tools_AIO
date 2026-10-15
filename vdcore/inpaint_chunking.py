"""Bounded-memory window planning for long-form video inpainting."""

AUTO_WINDOW_FRAMES = 80


def resolve_window_size(requested_frames):
    """Resolve 0/None to a safe automatic window size."""
    try:
        requested = int(requested_frames or 0)
    except (TypeError, ValueError):
        requested = 0
    return requested if requested > 0 else AUTO_WINDOW_FRAMES


def iter_context_windows(read_frame, window_size=0, overlap=10,
                         frame_limit=-1):
    """Yield bounded windows while emitting each source frame exactly once.

    Each item is ``(frames, keep_start, keep_end)``.  ``frames`` contains
    temporal context from both sides; only ``frames[keep_start:keep_end]``
    belongs in the output.  The reader is consumed sequentially exactly once.
    """
    window_size = resolve_window_size(window_size)
    overlap = max(0, int(overlap or 0))
    overlap = min(overlap, (window_size - 1) // 2)
    core_size = max(1, window_size - (2 * overlap))
    limit = int(frame_limit) if frame_limit and frame_limit > 0 else -1

    left_context = []
    pending = []
    read_count = 0
    eof = False

    while True:
        target_pending = core_size + overlap
        while not eof and len(pending) < target_pending:
            if limit > 0 and read_count >= limit:
                eof = True
                break
            frame = read_frame()
            if frame is None:
                eof = True
                break
            pending.append(frame)
            read_count += 1

        if not pending:
            break

        core_count = min(core_size, len(pending))
        right_count = min(overlap, len(pending) - core_count)
        frames = left_context + pending[:core_count + right_count]
        keep_start = len(left_context)
        keep_end = keep_start + core_count
        yield frames, keep_start, keep_end

        emitted = pending[:core_count]
        if overlap:
            left_context = (left_context + emitted)[-overlap:]
        else:
            left_context = []
        del pending[:core_count]


def split_window_on_boundaries(frames, keep_start, keep_end, boundaries):
    """Split a context window so no ProPainter call crosses a scene cut."""
    starts = [0] + sorted(
        int(boundary) for boundary in boundaries
        if 0 < int(boundary) < len(frames))
    ends = starts[1:] + [len(frames)]
    for scene_start, scene_end in zip(starts, ends):
        output_start = max(int(keep_start), scene_start)
        output_end = min(int(keep_end), scene_end)
        if output_start >= output_end:
            continue
        yield (frames[scene_start:scene_end],
               output_start - scene_start,
               output_end - scene_start)
