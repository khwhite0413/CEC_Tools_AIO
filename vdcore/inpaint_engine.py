import sys
import os
import numpy as np
import scipy.ndimage
import torch
import cv2
from PIL import Image

from vdcore.config import PROPAINTER_DIR, WEIGHTS_DIR


def _patch_propainter_misc():
    """Rewrite ProPainter's misc.py version-detection regex if it's
    still the upstream-broken one.

    ProPainter ships an anchored regex (`^...$`) that requires the
    torch version string to end exactly with an optional pre-release
    + `+git` tag. Modern PyTorch reports things like `2.6.0+cu124`
    where the optional groups don't match, returning an empty list;
    indexing `[0]` then raises IndexError on import:

        File ".../ProPainter/model/misc.py", line 56, in <module>
            IS_HIGH_VERSION = [int(m) for m in list(re.findall(
            r"^([0-9]+)\\.([0-9]+)\\.([0-9]+)([^0-9][a-zA-Z0-9]*)?
            (\\+git.*)?$", torch.__version__))[0][:3]] >= [1, 12, 0]
        IndexError: list index out of range

    setup.bat downloads ProPainter's main branch as-is so every fresh
    install hits this. Patch idempotently here at first import — if
    the buggy marker isn't present, we leave the file alone.
    """
    misc_path = os.path.join(PROPAINTER_DIR, "model", "misc.py")
    if not os.path.isfile(misc_path):
        return
    try:
        with open(misc_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return

    # Locate the offending assignment. The buggy regex literal is
    # unique enough to use as a fingerprint.
    start_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("IS_HIGH_VERSION") and 'r"^([0-9]+)' in line:
            start_idx = i
            break
    if start_idx is None:
        return  # already patched or a ProPainter variant we don't recognize

    # Walk forward through any line-continuation `\` so we replace the
    # full logical statement, not just its first physical line.
    end_idx = start_idx
    while end_idx < len(lines) and lines[end_idx].rstrip().endswith("\\"):
        end_idx += 1

    replacement = [
        "try:\n",
        "    _ver_match = re.findall("
        'r"([0-9]+)\\.([0-9]+)\\.([0-9]+)", torch.__version__)\n',
        "    IS_HIGH_VERSION = (["
        "int(m) for m in _ver_match[0][:3]] >= [1, 12, 0]) "
        "if _ver_match else True\n",
        "except Exception:\n",
        "    IS_HIGH_VERSION = True\n",
    ]

    new_lines = lines[:start_idx] + replacement + lines[end_idx + 1:]
    try:
        with open(misc_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except Exception:
        # Best-effort. If the write fails (permissions, antivirus, etc.)
        # the original import will still error and surface the issue —
        # better than silently producing wrong results.
        pass


def _add_propainter_path():
    # Patch before sys.path manipulation so the broken misc.py never
    # gets imported in its original form.
    _patch_propainter_misc()
    if PROPAINTER_DIR not in sys.path:
        sys.path.insert(0, PROPAINTER_DIR)


def _get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    """Copied from ProPainter inference_propainter.py to avoid import conflicts."""
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


def _load_to_tensors():
    """Load ProPainter's to_tensors without conflicting with our package."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pp_core_utils",
        os.path.join(PROPAINTER_DIR, "core", "utils.py")
    )
    pp_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp_utils)
    return pp_utils.to_tensors


class InpaintEngine:
    """Wrapper around ProPainter inference pipeline."""

    def __init__(self, device, fp16=True):
        self.device = device
        self.fp16 = fp16 and device.type == "cuda"
        self.fix_raft = None
        self.fix_flow_complete = None
        self.model = None
        self._loaded = False

    def load_models(self):
        if self._loaded:
            return

        _add_propainter_path()

        from model.modules.flow_comp_raft import RAFT_bi
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet
        from model.propainter import InpaintGenerator

        # RAFT (optical flow)
        raft_path = os.path.join(WEIGHTS_DIR, "raft-things.pth")
        self.fix_raft = RAFT_bi(raft_path, self.device)

        # Flow completion
        flow_path = os.path.join(WEIGHTS_DIR, "recurrent_flow_completion.pth")
        self.fix_flow_complete = RecurrentFlowCompleteNet(flow_path)
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(self.device)
        self.fix_flow_complete.eval()

        # ProPainter
        pp_path = os.path.join(WEIGHTS_DIR, "ProPainter.pth")
        self.model = InpaintGenerator(model_path=pp_path).to(self.device)
        self.model.eval()

        self._loaded = True

    def inpaint_frames(self, frames_bgr, mask_np, settings, progress_callback=None):
        """
        Inpaint video frames.

        Args:
            frames_bgr: list of BGR numpy arrays (H, W, 3)
            mask_np: single binary mask (H, W), 255 = inpaint region
            settings: dict with subvideo_length, neighbor_length, ref_stride, etc.
            progress_callback: callable(current, total)

        Returns:
            list of inpainted BGR numpy arrays at original resolution
        """
        self.load_models()
        _add_propainter_path()
        to_tensors = _load_to_tensors()

        neighbor_length = settings.get("neighbor_length", 10)
        ref_stride = settings.get("ref_stride", 10)
        subvideo_length = settings.get("subvideo_length", 80)
        raft_iter = settings.get("raft_iter", 20)
        mask_dilation = settings.get("mask_dilation", 4)

        video_length = len(frames_bgr)
        if video_length == 0:
            return []

        # --- Prepare frames as PIL Images (RGB) ---
        original_h, original_w = frames_bgr[0].shape[:2]
        pil_frames = []
        for f in frames_bgr:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))

        # Resize to be divisible by 8
        out_w, out_h = original_w, original_h
        process_w = out_w - out_w % 8
        process_h = out_h - out_h % 8
        if process_w != out_w or process_h != out_h:
            pil_frames = [f.resize((process_w, process_h)) for f in pil_frames]
        w, h = process_w, process_h

        # --- Prepare masks ---
        mask_gray = mask_np.copy()
        if mask_gray.shape[0] != h or mask_gray.shape[1] != w:
            mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_gray > 127).astype(np.uint8)

        # Flow masks (dilated for optical flow)
        flow_masks = []
        masks_dilated = []
        for _ in range(video_length):
            if mask_dilation > 0:
                flow_mask = scipy.ndimage.binary_dilation(mask_binary, iterations=mask_dilation).astype(np.uint8)
            else:
                flow_mask = mask_binary.copy()
            flow_masks.append(Image.fromarray(flow_mask * 255))

            if mask_dilation > 0:
                dilated = scipy.ndimage.binary_dilation(mask_binary, iterations=mask_dilation).astype(np.uint8)
            else:
                dilated = mask_binary.copy()
            masks_dilated.append(Image.fromarray(dilated * 255))

        # --- Convert to tensors ---
        frames_inp = [np.array(f).astype(np.uint8) for f in pil_frames]
        frames = to_tensors()(pil_frames).unsqueeze(0) * 2 - 1
        flow_masks_t = to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated_t = to_tensors()(masks_dilated).unsqueeze(0)
        frames = frames.to(self.device)
        flow_masks_t = flow_masks_t.to(self.device)
        masks_dilated_t = masks_dilated_t.to(self.device)

        use_half = self.fp16

        with torch.no_grad():
            # --- 1. Compute optical flow (RAFT) ---
            if progress_callback:
                progress_callback(0, video_length)

            if frames.size(-1) <= 640:
                short_clip_len = 12
            elif frames.size(-1) <= 720:
                short_clip_len = 8
            elif frames.size(-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2

            if video_length > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f_idx in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f_idx + short_clip_len)
                    if f_idx == 0:
                        flows_f, flows_b = self.fix_raft(frames[:, f_idx:end_f], iters=raft_iter)
                    else:
                        flows_f, flows_b = self.fix_raft(frames[:, f_idx - 1:end_f], iters=raft_iter)
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()
                    if progress_callback:
                        progress_callback(0, video_length)
                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = self.fix_raft(frames, iters=raft_iter)
                torch.cuda.empty_cache()
                if progress_callback:
                    progress_callback(0, video_length)

            # --- Apply fp16 ---
            if use_half:
                frames = frames.half()
                flow_masks_t = flow_masks_t.half()
                masks_dilated_t = masks_dilated_t.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                self.fix_flow_complete = self.fix_flow_complete.half()
                self.model = self.model.half()

            # --- 2. Complete flow ---
            flow_length = gt_flows_bi[0].size(1)
            if flow_length > subvideo_length:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f_idx in range(0, flow_length, subvideo_length):
                    s_f = max(0, f_idx - pad_len)
                    e_f = min(flow_length, f_idx + subvideo_length + pad_len)
                    pad_len_s = max(0, f_idx) - s_f
                    pad_len_e = e_f - min(flow_length, f_idx + subvideo_length)
                    pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        flow_masks_t[:, s_f:e_f + 1])
                    pred_flows_bi_sub = self.fix_flow_complete.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        pred_flows_bi_sub,
                        flow_masks_t[:, s_f:e_f + 1])
                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()
                    if progress_callback:
                        progress_callback(0, video_length)
                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                pred_flows_bi = (pred_flows_f, pred_flows_b)
            else:
                pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks_t)
                pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks_t)
                torch.cuda.empty_cache()
                if progress_callback:
                    progress_callback(0, video_length)

            # --- 3. Image propagation ---
            masked_frames = frames * (1 - masks_dilated_t)
            subvideo_length_img_prop = min(100, subvideo_length)
            if video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f_idx in range(0, video_length, subvideo_length_img_prop):
                    s_f = max(0, f_idx - pad_len)
                    e_f = min(video_length, f_idx + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f_idx) - s_f
                    pad_len_e = e_f - min(video_length, f_idx + subvideo_length_img_prop)

                    b, t, _, _, _ = masks_dilated_t[:, s_f:e_f].size()
                    pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
                    prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(
                        masked_frames[:, s_f:e_f], pred_flows_bi_sub,
                        masks_dilated_t[:, s_f:e_f], 'nearest')
                    updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated_t[:, s_f:e_f]) + \
                        prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated_t[:, s_f:e_f]
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)

                    updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()
                    if progress_callback:
                        progress_callback(0, video_length)
                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated_t.size()
                prop_imgs, updated_local_masks = self.model.img_propagation(
                    masked_frames, pred_flows_bi, masks_dilated_t, 'nearest')
                updated_frames = frames * (1 - masks_dilated_t) + prop_imgs.view(b, t, 3, h, w) * masks_dilated_t
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                torch.cuda.empty_cache()
                if progress_callback:
                    progress_callback(0, video_length)

        # --- 4. Feature propagation + Transformer ---
        ori_frames = frames_inp
        comp_frames = [None] * video_length

        neighbor_stride = neighbor_length // 2
        if video_length > subvideo_length:
            ref_num = subvideo_length // ref_stride
        else:
            ref_num = -1

        for f_idx in range(0, video_length, neighbor_stride):
            neighbor_ids = [
                i for i in range(max(0, f_idx - neighbor_stride),
                                 min(video_length, f_idx + neighbor_stride + 1))
            ]
            ref_ids = _get_ref_index(f_idx, neighbor_ids, video_length, ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated_t[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (
                pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :]
            )

            with torch.no_grad():
                l_t = len(neighbor_ids)
                pred_img = self.model(selected_imgs, selected_pred_flows_bi,
                                      selected_masks, selected_update_masks, l_t)
                pred_img = pred_img.view(-1, 3, h, w)
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks_dilated_t[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                        + ori_frames[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)

            torch.cuda.empty_cache()

            if progress_callback:
                done = min(f_idx + neighbor_stride, video_length)
                progress_callback(done, video_length)

        # --- 5. Resize back and convert to BGR ---
        result_frames = []
        for f in comp_frames:
            if f is None:
                result_frames.append(frames_bgr[len(result_frames)])
                continue
            if f.shape[0] != original_h or f.shape[1] != original_w:
                f = cv2.resize(f, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            result_frames.append(bgr)

        return result_frames

    def release(self):
        """Free GPU memory."""
        self.fix_raft = None
        self.fix_flow_complete = None
        self.model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
