"""
Run WASB/HRNet ball-tracking inference on a pickleball video using the
fine-tuned wasb_pickleball_final checkpoint (falls back to the zero-shot
WASB tennis checkpoint if the fine-tuned weights aren't present).

Reproduces the model, weight loading, affine-warp/inverse-warp and
postprocessing logic from wasb_pickleball_correct_coords.ipynb (sections
2, 3, 4, 10b and 11), applied to every frame of a full video instead of a
pre-extracted, pre-labelled clip.

Detections are then run through OnlineTracker, a physics-based online
tracker (ported from the training notebook's evaluate_tracked()) instead of
a plain per-frame argmax, following the tracking-based postprocessing
described in the WASB paper (Tarashima et al., "A Widely Applicable Strong
Baseline for Sports Ball Detection and Tracking"). Per frame it:
  1. predicts the ball's position from a short trajectory fit (linear in x,
     parabolic in y, refit on every confirmed point -- a bounce or reversal
     resets the fit to just the new monotone run);
  2. re-searches the RAW heatmap in a disc around that prediction with a
     lower threshold than the global detector uses, so a locally-brightest
     but globally-subthreshold blob can still be recovered;
  3. blends the found candidate with the prediction (--interp-alpha), or,
     if nothing is found near the prediction at all, either coasts on pure
     trajectory extrapolation for a few frames (default) or reports the
     frame invisible (--no-coast).
Every output row's Source column says which of those three cases produced
it (global / local / predicted), so genuinely model-detected points stay
distinguishable from physics-only guesses.

Usage:
    python infer_video.py vidoes/3_3.mp4
    python infer_video.py vidoes/3_3.mp4 --weights wasb_pickleball_final.pth.zip
    python infer_video.py vidoes/3_3.mp4 --no-coast          # invisible instead of extrapolating
    python infer_video.py vidoes/3_3.mp4 --tracker median    # old per-frame behaviour

Outputs (under predictions/<video_stem>/):
    <stem>_pred.csv           Frame, Visibility, X, Y, Score, Source (pixel coords, original frame)
    <stem>_annotated_frames/  one PNG per frame with the predicted ball circled
    <stem>_annotated.mp4      the source video with the same overlay burned in
"""
import argparse
import glob
import math
import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

BN_MOMENTUM = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

WASB_TENNIS_ID = "14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z"
TENNIS_CKPT = osp.join(WEIGHTS_DIR, "wasb_tennis_best.pth.tar")

# fine-tuned pickleball checkpoint, preferred over the zero-shot tennis one
# whenever present. torch.load() reads this .zip directly (a .pth file
# already *is* a zip archive internally) so no extraction step is needed.
PICKLEBALL_CKPT = "wasb_pickleball_final.pth.zip"

CFG = {
    "model": {
        "name": "hrnet",
        "frames_in": 3,
        "frames_out": 3,
        "inp_height": 288, "inp_width": 512,
        "out_height": 288, "out_width": 512,
        "rgb_diff": False,
        "out_scales": [0],
        "MODEL": {
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "PRETRAINED_LAYERS": ["*"],
                "STEM":   {"INPLANES": 64, "STRIDES": [1, 1]},
                "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK",
                           "NUM_BLOCKS": [1], "NUM_CHANNELS": [32], "FUSE_METHOD": "SUM"},
                "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC",
                           "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32], "FUSE_METHOD": "SUM"},
                "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": "BASIC",
                           "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64],
                           "FUSE_METHOD": "SUM"},
                "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": "BASIC",
                           "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [16, 32, 64, 128],
                           "FUSE_METHOD": "SUM"},
                "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
            },
            "INIT_WEIGHTS": True,
        },
    },
    "detector": {"postprocessor": {"score_threshold": 0.5, "scales": [0],
                                    "blob_det_method": "concomp", "use_hm_weight": True}},
}

INP_W, INP_H = CFG["model"]["inp_width"], CFG["model"]["inp_height"]
OUT_W, OUT_H = CFG["model"]["out_width"], CFG["model"]["out_height"]
FRAMES_IN = CFG["model"]["frames_in"]
FRAMES_OUT = CFG["model"]["frames_out"]
SCALE0 = 0


# ===========================================================================
# HRNet — verbatim from nttcom/WASB-SBDT src/models/hrnet.py (Microsoft, MIT)
# ===========================================================================
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super().__init__()
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([self._make_one_branch(i, block, num_blocks, num_channels)
                               for i in range(num_branches)])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches, num_inchannels = self.num_branches, self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            outc = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], outc, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(outc)))
                        else:
                            outc = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], outc, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(outc), nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                y = y + x[j] if i == j else y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class HRNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self._frames_in = cfg['frames_in']
        self._frames_out = cfg['frames_out']
        self._out_scales = cfg['out_scales']
        self._stem_strides = cfg['MODEL']['EXTRA']['STEM']['STRIDES']
        self._stem_inplanes = cfg['MODEL']['EXTRA']['STEM']['INPLANES']

        self.conv1 = nn.Conv2d(3 * self._frames_in, self._stem_inplanes, kernel_size=3,
                                stride=self._stem_strides[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self._stem_inplanes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(self._stem_inplanes, self._stem_inplanes, kernel_size=3,
                                stride=self._stem_strides[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self._stem_inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = cfg['MODEL']['EXTRA']['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, self._stem_inplanes, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        self.num_deconvs = cfg['MODEL']['EXTRA']['DECONV']['NUM_DECONVS']
        self.deconv_config = cfg['MODEL']['EXTRA']['DECONV']
        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

        self.deconv_layers = self._make_deconv_layers(cfg, pre_stage_channels[0])
        self.final_layers = self._make_final_layers(cfg, pre_stage_channels)

    def _make_final_layers(self, cfg, channels):
        kernel_size = cfg['MODEL']['EXTRA']['FINAL_CONV_KERNEL']
        layers = []
        for scale in self._out_scales:
            layers.append(nn.Conv2d(in_channels=channels[scale], out_channels=self._frames_out,
                                     kernel_size=kernel_size))
        return nn.ModuleList(layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            return deconv_kernel, 1, 0
        if deconv_kernel == 3:
            return deconv_kernel, 1, 1
        if deconv_kernel == 2:
            return deconv_kernel, 0, 0

    def _make_deconv_layers(self, cfg, input_channels):
        extra = cfg.MODEL.EXTRA
        deconv_cfg = extra.DECONV
        deconv_layers = []
        for i in range(deconv_cfg.NUM_DECONVS):
            output_channels = input_channels
            deconv_kernel, padding, output_padding = self._get_deconv_cfg(deconv_cfg.KERNEL_SIZE[i])
            layers = [nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, deconv_kernel, stride=2,
                                    padding=padding, output_padding=output_padding, bias=False),
                nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True))]
            deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels
        return nn.ModuleList(deconv_layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = not (not multi_scale_output and i == num_modules - 1)
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels,
                                                 num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = [self.transition1[i](x) if self.transition1[i] is not None else x
                  for i in range(self.stage2_cfg['NUM_BRANCHES'])]
        y_list = self.stage2(x_list)

        x_list = [self.transition2[i](y_list[-1]) if self.transition2[i] is not None else y_list[i]
                  for i in range(self.stage3_cfg['NUM_BRANCHES'])]
        y_list = self.stage3(x_list)

        x_list = [self.transition3[i](y_list[-1]) if self.transition3[i] is not None else y_list[i]
                  for i in range(self.stage4_cfg['NUM_BRANCHES'])]
        y_list = self.stage4(x_list)

        y_out = {}
        for scale in self._out_scales:
            x = y_list[scale]
            for i in range(self.num_deconvs):
                x = self.deconv_layers[i][scale](x)
            y_out[scale] = self.final_layers[scale](x)
        return y_out


class AttrDict(dict):
    def __init__(self, d=None):
        super().__init__()
        for k, v in (d or {}).items():
            self[k] = AttrDict(v) if isinstance(v, dict) else v

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def build_model(cfg):
    assert cfg["model"]["name"] == "hrnet"
    return HRNet(AttrDict(cfg["model"]))


def load_state_dict_clean(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


def ensure_tennis_checkpoint():
    if osp.exists(TENNIS_CKPT):
        return TENNIS_CKPT
    hits = glob.glob("**/wasb_tennis*.pth.tar", recursive=True)
    if hits:
        return hits[0]
    import gdown
    gdown.download(id=WASB_TENNIS_ID, output=TENNIS_CKPT, quiet=False)
    assert osp.exists(TENNIS_CKPT), "download failed - check internet connectivity"
    return TENNIS_CKPT


def resolve_checkpoint(weights_arg):
    """--weights override > fine-tuned pickleball checkpoint > zero-shot tennis checkpoint."""
    if weights_arg:
        assert osp.exists(weights_arg), f"weights file not found: {weights_arg}"
        return weights_arg, "fine-tuned (user-specified)"
    if osp.exists(PICKLEBALL_CKPT):
        return PICKLEBALL_CKPT, "fine-tuned pickleball"
    hits = glob.glob("**/wasb_pickleball*.pth*", recursive=True)
    if hits:
        return hits[0], "fine-tuned pickleball"
    return ensure_tennis_checkpoint(), "zero-shot tennis (no fine-tuned checkpoint found)"


# ===========================================================================
# Affine warp (verbatim from utils/image.py, Microsoft / Xingyi Zhou, MIT)
# ===========================================================================
def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [src_point[0] * cs - src_point[1] * sn, src_point[0] * sn + src_point[1] * cs]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, (np.ndarray, list)):
        scale = np.array([scale, scale], dtype=np.float32)
    src_w = scale[0]
    dst_w, dst_h = output_size[0], output_size[1]
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    return np.dot(t, new_pt)[:2]


def get_transform(img, input_wh, inv=0):
    h, w = img.shape[:2]
    c = np.array([w / 2., h / 2.], dtype=np.float32)
    s = max(h, w) * 1.0
    return get_affine_transform(c, s, 0, [input_wh[0], input_wh[1]], inv=inv)


# ===========================================================================
# Postprocessor (verbatim from detectors/postprocessor.py, concomp branch)
# ===========================================================================
class TracknetV2Postprocessor:
    def __init__(self, cfg):
        pp = cfg["detector"]["postprocessor"]
        self._score_threshold = pp["score_threshold"]
        self._use_hm_weight = pp["use_hm_weight"]

    def _detect_blob_concomp(self, hm):
        xys, scores = [], []
        if np.max(hm) > self._score_threshold:
            _, hm_th = cv2.threshold(hm, self._score_threshold, 1, cv2.THRESH_BINARY)
            n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))
            for m in range(1, n_labels):
                ys, xs = np.where(labels == m)
                ws = hm[ys, xs]
                if self._use_hm_weight:
                    score = ws.sum()
                    x = np.sum(xs * ws) / np.sum(ws)
                    y = np.sum(ys * ws) / np.sum(ws)
                else:
                    score = ws.shape[0]
                    x, y = xs.mean(), ys.mean()
                xys.append(np.array([x, y]))
                scores.append(float(score))
        return xys, scores

    def run(self, logits, affine_inv):
        hms = torch.sigmoid(logits).cpu().numpy()
        inv = affine_inv.cpu().numpy()
        out = defaultdict(dict)
        for b in range(hms.shape[0]):
            for s in range(hms.shape[1]):
                xys, scores = self._detect_blob_concomp(hms[b, s])
                out[b][s] = {"xys": [affine_transform(xy.astype(np.float32), inv[b]) for xy in xys],
                             "scores": scores}
        return out


IMNORM = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


@torch.no_grad()
def predict_triplet(model, post, frames_bgr):
    """3 consecutive BGR frames (np arrays, full resolution) ->
    list (len frames_out) of candidate lists [(x, y, score), ...] (possibly empty).

    Unlike a plain top-1 decode, every blob the heatmap fires on is kept so
    that a downstream tracker can pick the trajectory-consistent one instead
    of always taking the highest-scoring (and not necessarily correct) blob.
    """
    trans_in = trans_inv = None
    chans = []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if trans_in is None:
            trans_in = get_transform(rgb, (INP_W, INP_H))
            trans_inv = get_transform(rgb, (OUT_W, OUT_H), inv=1)
        w = cv2.warpAffine(rgb, trans_in, (INP_W, INP_H), flags=cv2.INTER_LINEAR)
        chans.append(IMNORM(Image.fromarray(w)))
    x = torch.cat(chans, 0)[None].to(DEVICE)

    logits = model(x)[SCALE0]
    res = post.run(logits, torch.from_numpy(trans_inv).float()[None])
    out = []
    for s in range(logits.shape[1]):
        d = res[0][s]
        out.append([(float(xy[0]), float(xy[1]), float(score))
                    for xy, score in zip(d["xys"], d["scores"])])
    return out


def top1(cands):
    """Highest-score candidate from a list of (x, y, score), or (None, None, 0.0)."""
    if not cands:
        return None, None, 0.0
    return max(cands, key=lambda c: c[2])


def merge_median(cands):
    """cands: list of (x, y, score) from overlapping windows for one frame."""
    pts = [(x, y) for x, y, _ in cands if x is not None]
    if not pts:
        return None, None, 0.0
    arr = np.array(pts)
    x, y = np.median(arr, axis=0)
    score = max(s for _, _, s in cands)
    return float(x), float(y), float(score)


# ===========================================================================
# Online tracker (postprocessing) -- constant-acceleration trajectory
# prediction + local heatmap re-search, ported from the training notebook's
# evaluate_tracked()/OnlineTracker (itself following the WASB paper's
# tracking-based postprocessing). Unlike the simpler nearest-candidate
# version this replaces, it does its own weighted-centroid re-decode of the
# raw heatmap around the predicted point (not just a pick among the blobs
# that already cleared --score-threshold), so it can recover a ball that's
# locally the brightest thing in that patch even if it never reached the
# global score threshold -- and it can fall back to pure physics
# extrapolation for a few frames if the heatmap shows nothing at all there.
# ===========================================================================
from collections import deque


class OnlineTracker:
    """Predict next ball position from recent confirmed detections.

    Fits:
        x(t) = a1*t + a0          (linear    -- roughly constant horizontal velocity)
        y(t) = a2*t^2 + a1*t + a0 (quadratic -- gravity gives a parabolic arc)

    Resets on:
        - large speed jump between consecutive frames (bounce, serve)
        - gap > max_age frames with no confirmed update (ball left the frame)
        - a y-direction reversal with >= 4 points (bounce detected) -- keeps
          only the most recent monotone run so the fit doesn't straddle it
    """

    def __init__(self, window=8, max_age=3, max_speed_px=150):
        self.window = window
        self.max_age = max_age
        self.max_speed = max_speed_px
        self._history = deque(maxlen=window)  # (frame_idx, x, y)
        self._last_frame = -999
        self._px = self._py = None  # polynomial coefficients

    def reset(self):
        self._history.clear()
        self._px = self._py = None
        self._last_frame = -999

    def update(self, frame_idx, x, y):
        if (self._history
                and frame_idx == self._history[-1][0] + 1
                and math.hypot(x - self._history[-1][1], y - self._history[-1][2]) > self.max_speed):
            self._history.clear()  # big jump on a consecutive frame -> bounce/new rally

        self._history.append((frame_idx, x, y))
        self._last_frame = frame_idx
        self._refit()

    def _refit(self):
        n = len(self._history)
        if n < 2:
            self._px = self._py = None
            return
        fs = np.array([h[0] for h in self._history], float)
        xs = np.array([h[1] for h in self._history], float)
        ys = np.array([h[2] for h in self._history], float)
        self._px = np.polyfit(fs, xs, min(1, n - 1))
        self._py = np.polyfit(fs, ys, min(2, n - 1))

        if n >= 4:
            dy = np.diff(ys)
            if not np.all(dy >= 0) and not np.all(dy <= 0):
                last_sign = np.sign(dy[-1])
                keep = 1
                for d in reversed(dy[:-1]):
                    if np.sign(d) == last_sign:
                        keep += 1
                    else:
                        break
                recent = list(self._history)[-keep - 1:]
                self._history.clear()
                self._history.extend(recent)
                if len(self._history) >= 2:
                    fs = np.array([h[0] for h in self._history], float)
                    xs = np.array([h[1] for h in self._history], float)
                    ys = np.array([h[2] for h in self._history], float)
                    n2 = len(self._history)
                    self._px = np.polyfit(fs, xs, min(1, n2 - 1))
                    self._py = np.polyfit(fs, ys, min(2, n2 - 1))
                else:
                    self._px = self._py = None

    def predict(self, frame_idx):
        """Return (x, y) prediction, or None if the tracker is cold."""
        if self._px is None:
            return None
        if frame_idx - self._last_frame > self.max_age:
            return None
        return (float(np.polyval(self._px, frame_idx)), float(np.polyval(self._py, frame_idx)))

    @property
    def is_warm(self):
        return self._px is not None


def radius_px_to_hm(radius_px, frame_bgr, out_w):
    """Convert a search radius in original-frame pixels to heatmap pixels.

    get_transform() is an isotropic scale (no rotation) mapping a
    max(h, w)-side square onto an out_w-side square, so lengths scale by
    out_w / max(h, w).
    """
    h, w = frame_bgr.shape[:2]
    return radius_px * out_w / max(h, w)


def local_weighted_centroid(hm, cx_hm, cy_hm, r_hm, threshold):
    """Brightness-weighted centroid inside a disc of heatmap cells.

    Returns (cx_hm, cy_hm, peak_val) in HEATMAP coordinates, or None if the
    brightest cell in the disc doesn't clear `threshold`.
    """
    H, W = hm.shape
    y0, y1 = max(0, int(cy_hm - r_hm)), min(H, int(cy_hm + r_hm) + 1)
    x0, x1 = max(0, int(cx_hm - r_hm)), min(W, int(cx_hm + r_hm) + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2 = (xx - cx_hm) ** 2 + (yy - cy_hm) ** 2
    patch = hm[y0:y1, x0:x1].copy()
    patch[dist2 > r_hm ** 2] = 0.0
    peak = float(patch.max())
    if peak < threshold:
        return None
    w = patch.clip(0)
    wsum = w.sum()
    return float((xx * w).sum() / wsum), float((yy * w).sum() / wsum), peak


@torch.no_grad()
def compute_frame_heatmaps(model, frames_bgr):
    """One decoded heatmap per video frame (not per window), avoiding the
    triple-counting a naive sliding-window decode would give: each window
    only contributes its LAST output slot (the training notebook's
    evaluate_tracked() does the same, "to avoid triple-counting in
    sequence"). Window 0 additionally contributes its first two slots so
    frames 0 and 1 -- which no window's last slot ever reaches -- still get
    a heatmap.

    Returns {frame_idx: (hm (OUT_H,OUT_W) float32, trans_out, trans_inv)}.
    """
    n = len(frames_bgr)
    heatmaps = {}
    n_windows = max(0, n - FRAMES_IN + 1)
    for i in tqdm(range(n_windows), desc="inference"):
        window = frames_bgr[i:i + FRAMES_IN]
        trans_in = trans_out = trans_inv = None
        chans = []
        for bgr in window:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if trans_in is None:
                trans_in = get_transform(rgb, (INP_W, INP_H))
                trans_out = get_transform(rgb, (OUT_W, OUT_H))
                trans_inv = get_transform(rgb, (OUT_W, OUT_H), inv=1)
            w = cv2.warpAffine(rgb, trans_in, (INP_W, INP_H), flags=cv2.INTER_LINEAR)
            chans.append(IMNORM(Image.fromarray(w)))
        x = torch.cat(chans, 0)[None].to(DEVICE)
        logits = model(x)[SCALE0]
        hms = torch.sigmoid(logits)[0].cpu().numpy()  # (FRAMES_OUT, OUT_H, OUT_W)

        if i == 0:
            for s in range(FRAMES_OUT - 1):
                heatmaps[s] = (hms[s], trans_out, trans_inv)
        slot = FRAMES_OUT - 1
        heatmaps[i + slot] = (hms[slot], trans_out, trans_inv)
    return heatmaps


def track_online(model, post, frames, args):
    """Full-video pass with the physics-based OnlineTracker: predict from
    trajectory history, re-search the heatmap locally around the
    prediction, blend with (or fall back to) the standard global decode."""
    n = len(frames)
    heatmaps = compute_frame_heatmaps(model, frames)
    tracker = OnlineTracker(window=args.tracker_window, max_age=args.max_age,
                             max_speed_px=args.max_speed)

    rows = []
    for i in tqdm(range(n), desc="tracking"):
        if i not in heatmaps:
            rows.append({"Frame": i, "Visibility": 0, "X": "", "Y": "", "Score": "", "Source": "none"})
            continue
        hm, trans_out, trans_inv = heatmaps[i]
        inv_np = np.float32(trans_inv[:2])

        xys_g, sc_g = post._detect_blob_concomp(hm)
        if xys_g:
            k = int(np.argmax(sc_g))
            std_xy = affine_transform(xys_g[k].astype(np.float32), inv_np)
            std_score = sc_g[k]
            std_visi = True
        else:
            std_xy, std_score, std_visi = None, 0.0, False

        pred = tracker.predict(i)
        r_hm = radius_px_to_hm(args.search_radius, frames[i], OUT_W)

        if pred is not None:
            pred_x, pred_y = pred
            pred_hm = affine_transform(np.array([pred_x, pred_y], np.float32), trans_out)
            local = local_weighted_centroid(hm, pred_hm[0], pred_hm[1], r_hm, args.local_threshold)

            global_in_range = std_visi and math.hypot(std_xy[0] - pred_x, std_xy[1] - pred_y) <= args.search_radius

            if local is not None:
                lx_hm, ly_hm, local_peak = local
                local_orig = affine_transform(np.array([lx_hm, ly_hm], np.float32), inv_np)
                if global_in_range:
                    cand_x, cand_y, source, score = std_xy[0], std_xy[1], "global", std_score
                else:
                    cand_x, cand_y, source, score = local_orig[0], local_orig[1], "local", local_peak
                final_x = args.interp_alpha * cand_x + (1 - args.interp_alpha) * pred_x
                final_y = args.interp_alpha * cand_y + (1 - args.interp_alpha) * pred_y
                final_v = True
            elif global_in_range:
                final_x, final_y, final_v, source, score = std_xy[0], std_xy[1], True, "global", std_score
            elif not args.no_coast:
                # nothing seen anywhere near the prediction -- trust the
                # trajectory model for a few frames (see --max-age / --no-coast)
                final_x, final_y, final_v, source, score = pred_x, pred_y, True, "predicted", 0.0
            else:
                final_x = final_y = None
                final_v, source, score = False, "none", 0.0
        elif std_visi:
            final_x, final_y, final_v, source, score = std_xy[0], std_xy[1], True, "global", std_score
        else:
            final_x = final_y = None
            final_v, source, score = False, "none", 0.0

        rows.append({
            "Frame": i,
            "Visibility": int(final_v),
            "X": round(final_x, 2) if final_v else "",
            "Y": round(final_y, 2) if final_v else "",
            "Score": round(float(score), 4) if final_v else "",
            "Source": source,
        })
        if source in ("global", "local"):
            # only confirm the trajectory on real heatmap evidence -- if a
            # "predicted" (pure-extrapolation) frame confirmed the track too,
            # tracker.predict() would keep returning non-None forever (since
            # _last_frame keeps advancing), --max-age would never trigger,
            # and the tracker could drift indefinitely without ever being
            # checked against the model again.
            tracker.update(i, final_x, final_y)

    n_pred_only = sum(1 for r in rows if r["Source"] == "predicted")
    print(f"online tracker: search_radius={args.search_radius:.1f}px, max_age={args.max_age} frames, "
          f"{n_pred_only} frames filled by pure trajectory extrapolation (no heatmap evidence)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", help="path to source video, e.g. vidoes/3_3.mp4")
    ap.add_argument("--weights", default=None,
                     help="checkpoint path (default: wasb_pickleball_final.pth.zip if present, "
                          "else the zero-shot tennis checkpoint)")
    ap.add_argument("--score-threshold", type=float, default=0.5,
                     help="global blob score threshold (used directly by --tracker median, and "
                          "as the 'global' candidate gate by --tracker online)")
    ap.add_argument("--tracker", choices=["online", "median"], default="online",
                     help="'online': trajectory-predicting tracker with local heatmap re-search "
                          "(recommended). 'median': old per-window-argmax + median merge, no "
                          "temporal consistency")
    ap.add_argument("--search-radius", type=float, default=100.0,
                     help="online tracker: radius (original-frame px) around the predicted "
                          "position to trust a candidate or re-search the heatmap")
    ap.add_argument("--local-threshold", type=float, default=0.25,
                     help="online tracker: heatmap score threshold used INSIDE the local "
                          "search window (lower than --score-threshold on purpose, so a dim "
                          "blob that's still the brightest thing near the prediction counts)")
    ap.add_argument("--interp-alpha", type=float, default=0.75,
                     help="online tracker: blend weight between the detected candidate and the "
                          "trajectory prediction (1.0 = pure detection, 0.0 = pure prediction)")
    ap.add_argument("--max-age", type=int, default=3,
                     help="online tracker: max consecutive frames with no confirmed update "
                          "before the trajectory is considered cold and dropped")
    ap.add_argument("--max-speed", type=float, default=150.0,
                     help="online tracker: pixel jump between consecutive confirmed frames that "
                          "resets the trajectory history (bounce/serve/new rally)")
    ap.add_argument("--tracker-window", type=int, default=8,
                     help="online tracker: how many recent confirmed points the x/y polynomial "
                          "fit uses")
    ap.add_argument("--no-coast", action="store_true",
                     help="online tracker: mark a frame invisible when neither the global "
                          "detector nor the local heatmap re-search finds anything near the "
                          "prediction, instead of falling back to pure trajectory extrapolation")
    ap.add_argument("--out-dir", default=None, help="default: predictions/<video_stem>/")
    ap.add_argument("--save-all-frames", action="store_true",
                     help="save an annotated PNG for every frame, not just detections")
    args = ap.parse_args()

    stem = osp.splitext(osp.basename(args.video))[0]
    out_dir = args.out_dir or osp.join("predictions", stem)
    frames_dir = osp.join(out_dir, f"{stem}_annotated_frames")
    os.makedirs(frames_dir, exist_ok=True)
    csv_path = osp.join(out_dir, f"{stem}_pred.csv")
    video_out_path = osp.join(out_dir, f"{stem}_annotated.mp4")

    print(f"device: {DEVICE}")
    ckpt_path, ckpt_kind = resolve_checkpoint(args.weights)
    print(f"checkpoint: {ckpt_path} ({ckpt_kind}, {os.path.getsize(ckpt_path)/1e6:.1f} MB)")

    model = build_model(CFG).to(DEVICE)
    sd = load_state_dict_clean(ckpt_path)
    model.load_state_dict(sd)  # strict=True
    model.eval()
    print(f"strict load OK - {len(sd)} tensors")

    cfg = dict(CFG)
    cfg["detector"] = {"postprocessor": dict(CFG["detector"]["postprocessor"])}
    cfg["detector"]["postprocessor"]["score_threshold"] = args.score_threshold
    post = TracknetV2Postprocessor(cfg)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"cannot open {args.video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {args.video}  {W}x{H} @ {fps:.1f} fps, {n_frames} frames")

    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    n = len(frames)
    print(f"decoded {n} frames")

    if args.tracker == "online":
        rows = track_online(model, post, frames, args)
    else:
        # legacy strategy: best blob per window, then median across overlapping windows
        per_frame_window_top1 = defaultdict(list)
        n_windows = max(0, n - FRAMES_IN + 1)
        for i in tqdm(range(n_windows), desc="inference"):
            window = frames[i:i + FRAMES_IN]
            dets = predict_triplet(model, post, window)
            for s, cands in enumerate(dets):
                per_frame_window_top1[i + s].append(top1(cands))

        rows = []
        for i in range(n):
            x, y, score = merge_median(per_frame_window_top1.get(i, []))
            visible = x is not None
            rows.append({
                "Frame": i,
                "Visibility": int(visible),
                "X": round(x, 2) if visible else "",
                "Y": round(y, 2) if visible else "",
                "Score": round(score, 4) if visible else "",
                "Source": "global" if visible else "none",
            })

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)  # re-assert: a long-running job can outlive a temp dir
    df.to_csv(csv_path, index=False)
    n_visible = int(df["Visibility"].sum())
    print(f"csv saved: {csv_path}  ({n_visible}/{n} frames with a detection)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (W, H))
    radius = max(6, int(round(max(W, H) * 0.012)))
    for i, frame in enumerate(tqdm(frames, desc="annotating")):
        vis_frame = frame
        row = rows[i]
        if row["Visibility"]:
            x, y = int(round(row["X"])), int(round(row["Y"]))
            vis_frame = frame.copy()
            cv2.circle(vis_frame, (x, y), radius, (0, 0, 255), -1)  # solid red dot on the ball
        writer.write(vis_frame)
        if row["Visibility"] or args.save_all_frames:
            cv2.imwrite(osp.join(frames_dir, f"{i}.png"), vis_frame)
    writer.release()

    print(f"annotated video saved: {video_out_path}")
    print(f"annotated frames saved: {frames_dir} "
          f"({n_visible if not args.save_all_frames else n} images)")


if __name__ == "__main__":
    main()
