"""Temporal flow average (motion-compensated denoise reference).

Mirrors upstream ComfyUI-FlowDenoise master (May 2026):
  - bf16 autocast for MEMFOF / RAFT inference
  - low-resolution optical flow + magnitude-scaled upsample
  - GPU sliding-window buffer (eliminates redundant H2D transfers)

`precision` and `flow_scale` are now driven by the settings dict so the
UI's developer mode can override the fast-path defaults from
fdcore.pipeline.DEFAULT_SETTINGS.
"""

import torch
import torch.nn.functional as F

from .flow_warp import (warp_with_flow, color_similarity_mask,
                        forward_backward_occlusion_mask)
from .flow_models import get_memfof, get_raft
from .progress import ProgressReporter, CancelledError
from .scene_detection import detect_scene_boundaries


def _detect_scene_changes(frames: torch.Tensor, threshold: float,
                          mode: str = "mse"):
    return detect_scene_boundaries(
        frames, mode=mode, mse_threshold=threshold, mse_size=None)


def _same_scene(i: int, j: int, boundaries: set) -> bool:
    lo, hi = min(i, j), max(i, j)
    for b in range(lo + 1, hi + 1):
        if b in boundaries:
            return False
    return True


def _accumulate(warped, ref, w, color_threshold, weighted_sum, weight_sum):
    mask = color_similarity_mask(warped, ref, color_threshold)
    weighted_sum += w * mask * warped
    weight_sum += w * mask


def _flow_resolution(H: int, W: int, scale: float):
    """Round to multiple of 32 (MEMFOF requirement)."""
    if scale >= 1.0:
        return H, W
    fh = max(32, (int(H * scale) // 32) * 32)
    fw = max(32, (int(W * scale) // 32) * 32)
    return fh, fw


def _run_memfof(frames, B, H, W, settings, boundaries,
                output, weight_maps, device, reporter: ProgressReporter):
    window_size = settings["window_size"]
    weight_decay = settings["weight_decay"]
    flow_iterations = settings["flow_iterations"]
    color_threshold = settings["color_threshold"]
    batch_size = max(1, int(settings.get("batch_size", 1)))

    model = get_memfof(device)
    use_bf16 = settings.get("precision", "bf16") == "bf16"
    flow_scale = max(0.3, min(1.0, float(settings.get("flow_scale", 0.5))))
    amp_enabled = use_bf16 and device.type == "cuda"
    fh, fw = _flow_resolution(H, W, flow_scale)
    downsample = (fh, fw) != (H, W)

    for chunk_start in range(0, B, batch_size):
        if reporter.cancelled:
            raise CancelledError()
        chunk_end = min(chunk_start + batch_size, B)
        chunk_size = chunk_end - chunk_start

        # GPU sliding-window buffer — upload chunk + window padding once.
        buf_start = max(0, chunk_start - window_size)
        buf_end = min(B, chunk_end + window_size)
        gpu_buf = frames[buf_start:buf_end].to(device)

        chunk_frames = gpu_buf[chunk_start - buf_start:chunk_end - buf_start]
        weighted_sums = chunk_frames.clone()
        weight_sums = torch.ones(chunk_size, 1, H, W, device=device)

        for offset in range(1, window_size + 1):
            w = weight_decay ** offset
            triplet_list = []
            meta = []
            for k, i in enumerate(range(chunk_start, chunk_end)):
                has_prev = (i - offset) >= 0 and _same_scene(i, i - offset, boundaries)
                has_next = (i + offset) < B and _same_scene(i, i + offset, boundaries)
                if not has_prev and not has_next:
                    continue
                prev_idx = i - offset if has_prev else i
                next_idx = i + offset if has_next else i
                triplet_list.append(torch.stack([
                    gpu_buf[prev_idx - buf_start],
                    gpu_buf[i - buf_start],
                    gpu_buf[next_idx - buf_start],
                ], dim=0))
                meta.append((k, has_prev, has_next, prev_idx, next_idx))

            if not triplet_list:
                continue

            batch_triplets = torch.stack(triplet_list, dim=0)
            batch_triplets = (batch_triplets * 255.0).clamp(0, 255)

            if downsample:
                Nb, Tb = batch_triplets.shape[:2]
                triplet_in = F.interpolate(
                    batch_triplets.flatten(0, 1),
                    size=(fh, fw), mode='bilinear', align_corners=False
                ).unflatten(0, (Nb, Tb))
            else:
                triplet_in = batch_triplets

            with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=amp_enabled):
                out_dict = model(triplet_in, iters=flow_iterations)

            flow_lr = out_dict["flow"][-1].float()  # [N, 2, 2, fh, fw]
            del batch_triplets, triplet_in, out_dict

            if downsample:
                Nf = flow_lr.shape[0]
                flow_flat = flow_lr.view(Nf * 2, 2, fh, fw)
                flow_full = F.interpolate(flow_flat, size=(H, W),
                                          mode='bilinear', align_corners=False)
                flow_full[:, 0] = flow_full[:, 0] * (float(W) / float(fw))
                flow_full[:, 1] = flow_full[:, 1] * (float(H) / float(fh))
                flow_fields = flow_full.view(Nf, 2, 2, H, W)
                del flow_lr, flow_flat, flow_full
            else:
                flow_fields = flow_lr

            for j, (k, has_prev, has_next, prev_idx, next_idx) in enumerate(meta):
                bwd_flow = flow_fields[j:j + 1, 0]
                fwd_flow = flow_fields[j:j + 1, 1]
                ref_frame = chunk_frames[k:k + 1]
                if has_prev:
                    nb = gpu_buf[prev_idx - buf_start:prev_idx - buf_start + 1]
                    warped = warp_with_flow(nb, bwd_flow)
                    _accumulate(warped, ref_frame, w, color_threshold,
                                weighted_sums[k:k + 1], weight_sums[k:k + 1])
                    del warped
                if has_next:
                    nb = gpu_buf[next_idx - buf_start:next_idx - buf_start + 1]
                    warped = warp_with_flow(nb, fwd_flow)
                    _accumulate(warped, ref_frame, w, color_threshold,
                                weighted_sums[k:k + 1], weight_sums[k:k + 1])
                    del warped
            del flow_fields

        output[chunk_start:chunk_end] = (weighted_sums / (weight_sums + 1e-8)).cpu()
        weight_maps[chunk_start:chunk_end] = weight_sums.cpu()
        del gpu_buf, chunk_frames, weighted_sums, weight_sums
        # Chunk-scoped: B is the current buffer size (e.g. streaming chunk
        # of ~200 frames + overlap), not the full clip's total. The
        # pipeline's streaming loop emits overall progress separately.
        reporter.update_chunk(chunk_end, B)


def _run_raft(frames, B, H, W, settings, boundaries,
              output, weight_maps, device, reporter: ProgressReporter):
    window_size = settings["window_size"]
    weight_decay = settings["weight_decay"]
    flow_iterations = settings["flow_iterations"]
    color_threshold = settings["color_threshold"]
    flow_model = settings["flow_model"]

    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8

    model = get_raft(flow_model, device)
    use_bf16 = settings.get("precision", "bf16") == "bf16"
    amp_enabled = use_bf16 and device.type == "cuda"

    for i in range(B):
        if reporter.cancelled:
            raise CancelledError()
        ref_cpu = frames[i:i + 1]
        ref_padded = F.pad(ref_cpu, [0, pad_w, 0, pad_h], mode='reflect') \
            if (pad_h or pad_w) else ref_cpu

        ref_gpu = ref_padded.to(device)
        ref_orig = ref_cpu.to(device)
        weighted_sum = ref_orig.clone()
        weight_sum = torch.ones(1, 1, H, W, device=device)

        for offset in range(1, window_size + 1):
            w = weight_decay ** offset
            for j in [i - offset, i + offset]:
                if j < 0 or j >= B or not _same_scene(i, j, boundaries):
                    continue
                nb_cpu = frames[j:j + 1]
                nb_padded = F.pad(nb_cpu, [0, pad_w, 0, pad_h], mode='reflect') \
                    if (pad_h or pad_w) else nb_cpu
                nb_gpu = nb_padded.to(device)
                ref_255 = (ref_gpu * 255.0).clamp(0, 255)
                nb_255 = (nb_gpu * 255.0).clamp(0, 255)
                with torch.no_grad(), torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=amp_enabled):
                    flow_fwd = model(ref_255, nb_255,
                                     num_flow_updates=flow_iterations)[-1]
                    flow_bwd = model(nb_255, ref_255,
                                     num_flow_updates=flow_iterations)[-1]
                flow_fwd = flow_fwd[:, :, :H, :W].float()
                flow_bwd = flow_bwd[:, :, :H, :W].float()
                nb_orig = nb_cpu.to(device)
                warped = warp_with_flow(nb_orig, flow_fwd)
                cmask = color_similarity_mask(warped, ref_orig, color_threshold)
                omask = forward_backward_occlusion_mask(flow_fwd, flow_bwd)
                mask = cmask * omask
                weighted_sum += w * mask * warped
                weight_sum += w * mask
                del nb_gpu, nb_orig, warped, flow_fwd, flow_bwd, mask

        output[i:i + 1] = (weighted_sum / (weight_sum + 1e-8)).cpu()
        weight_maps[i:i + 1] = weight_sum.cpu()
        del ref_gpu, ref_orig, weighted_sum, weight_sum
        # Chunk-scoped: see _run_memfof's equivalent note.
        reporter.update_chunk(i + 1, B)


def temporal_flow_average(images: torch.Tensor, settings: dict,
                          reporter: ProgressReporter):
    """Motion-compensated temporal averaging.

    Returns:
        (clean_bhwc, None): clean is a float32 tensor in [0,1]. The second
        tuple slot used to carry a normalized 3-channel weight visualization
        for debug purposes, but no caller ever read it — building it on a
        long clip allocated B×H×W×3 float32 on CPU (~28 GB per 8000 720p
        frames just for the base weight map, and 3× more when expanded to
        RGB), which OOMed DefaultCPUAllocator on long runs. Callers unpack
        the tuple as `clean, _`.
    """
    B, H, W, C = images.shape
    src_device = images.device
    compute = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = images.permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW

    boundaries = _detect_scene_changes(
        frames, settings["scene_threshold"],
        settings.get("scene_detection_mode", "mse"))
    if boundaries:
        print(f"[FlowDenoise] Scene cuts at: {sorted(boundaries)}")

    output = torch.zeros_like(frames)
    weight_maps = torch.zeros(B, 1, H, W)

    flow_model = settings["flow_model"]
    if flow_model == "memfof":
        _run_memfof(frames, B, H, W, settings, boundaries,
                    output, weight_maps, compute, reporter)
    else:
        _run_raft(frames, B, H, W, settings, boundaries,
                  output, weight_maps, compute, reporter)

    # Don't clamp here — pipeline.run_on_tensor honors clamp_output instead.
    clean = output.permute(0, 2, 3, 1).to(src_device)

    # weight_maps is still populated in-place by the workers above (harmless
    # single-channel accumulator). It used to be normalized and expanded to
    # a full B×H×W×3 float32 tensor here for return, but nothing in the file
    # save path consumed it — see docstring above. Skipping the derivation
    # avoids a ~28 GB CPU allocation spike per 8000 720p frames.

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return clean, None
