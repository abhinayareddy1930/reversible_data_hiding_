
import os, json, base64, io, types
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from flask import Flask, request, jsonify, send_from_directory
from skimage.metrics import structural_similarity as skssim


# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH = "best_vit_segmenter.pth"
IMG_SIZE   = 224
T1_DEFAULT = 0.33   # below → 3-bit tier  (lowest attention = safest)
T2_DEFAULT = 0.66   # below → 2-bit tier  |  above → 1-bit tier
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, static_folder="static", template_folder="templates")
os.makedirs("templates", exist_ok=True)


# ─── MODEL  (exact copy of Cell 5) ───────────────────────────────────────────
class ViTSegmenter(nn.Module):
    """
    ViT-B/16 backbone + 4-layer transposed-conv decoder with BatchNorm.
    Attention hook on last block → get_attention_map() after every forward().
    """

    def __init__(self):
        super().__init__()

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False)
        # Grayscale (1-channel) patch projection
        self.vit.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
        self.vit.head = nn.Identity()

        # Decoder: 14×14 → 28 → 56 → 112 → 224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, 2, 2),        # 0
            nn.BatchNorm2d(512), nn.ReLU(inplace=True), # 1, 2
            nn.ConvTranspose2d(512, 256, 2, 2),        # 3
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), # 4, 5
            nn.ConvTranspose2d(256, 64,  2, 2),        # 6
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True), # 7, 8
            nn.ConvTranspose2d(64,  1,   2, 2),        # 9
            nn.Sigmoid()                                # 10
        )

        self.last_attn = None
        self._install_attention_hook()

    def _install_attention_hook(self):
        """Patch last block's attention forward to save raw softmax weights."""
        seg_model = self

        def _patched_attn_forward(mod, x, attn_mask=None, **kwargs):
            B, N, C  = x.shape
            H        = mod.num_heads
            head_dim = C // H

            qkv  = mod.qkv(x).reshape(B, N, 3, H, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            attn = (q @ k.transpose(-2, -1)) * mod.scale
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)

            seg_model.last_attn = attn.detach()   # ← save raw attention

            attn = mod.attn_drop(attn)
            x    = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x    = mod.proj(x)
            x    = mod.proj_drop(x)
            return x

        last_blk = self.vit.blocks[-1].attn
        last_blk.forward = types.MethodType(_patched_attn_forward, last_blk)

    def get_attention_map(self) -> np.ndarray:
        """Return [224,224] float32 normalised CLS-attention map."""
        assert self.last_attn is not None, "Call model(tensor) first."
        attn     = self.last_attn.mean(dim=1)           # avg heads → [B,N,N]
        cls_attn = attn[0, 0, 1:].reshape(14, 14)       # CLS row → patches
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        up = F.interpolate(
            cls_attn.unsqueeze(0).unsqueeze(0),
            size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )
        return up.squeeze().cpu().numpy()

    def forward(self, x):
        feats   = self.vit.forward_features(x)          # [B, 197, 768]
        patches = feats[:, 1:, :]                       # drop CLS → [B,196,768]
        B       = patches.shape[0]
        patches = patches.permute(0, 2, 1).reshape(B, 768, 14, 14)
        return self.decoder(patches)


# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
model = ViTSegmenter().to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅  Model loaded from {MODEL_PATH}  |  device={device}")
else:
    print(f"⚠️   {MODEL_PATH} not found — place it next to app.py")


# ─── IMAGE HELPERS ────────────────────────────────────────────────────────────

def b64_to_img(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot decode image — unsupported format.")
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.uint8)


def img_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


# ─── BRAIN MASK  (exact copy of Cell 12) ─────────────────────────────────────

def get_brain_mask(img_uint8: np.ndarray) -> np.ndarray:
    """
    Binary mask (0/1) of the full brain ROI.
    Steps: Otsu threshold → 7×7 ellipse morphological closing
           → keep only the single largest connected component.
    """
    _, binary = cv2.threshold(
        img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    if n_labels <= 1:
        mask = (closed > 0).astype(np.uint8)
    else:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    # Fill any holes (black regions) inside the brain/skull to make the entire ROI white
    import scipy.ndimage
    mask = scipy.ndimage.binary_fill_holes(mask).astype(np.uint8)

    return mask


# ─── BIT CONVERSION  (exact copy of Cell 15) ─────────────────────────────────

def text_to_bits(text: str) -> str:
    """UTF-8 text → binary string."""
    return ''.join(format(b, '08b') for b in text.encode('utf-8'))


def bits_to_text(bits: str) -> str:
    """Binary string → UTF-8 text."""
    byte_list = [int(bits[i:i+8], 2) for i in range(0, len(bits) - 7, 8)]
    return bytes(byte_list).decode('utf-8', errors='replace')


# ─── EMBED  (exact copy of Cell 16) ──────────────────────────────────────────

def embed_multi_tier(
    image_uint8:   np.ndarray,
    brain_mask:    np.ndarray,
    attention_map: np.ndarray,
    secret_bits:   str,
    T1: float = T1_DEFAULT,
    T2: float = T2_DEFAULT,
) -> tuple:
    """
    Embed secret_bits using 3-tier attention-guided LSB substitution.

    Tier 1  (attention <  T1)  : 3 LSBs per pixel  (lowest attention → safest)
    Tier 2  (T1 ≤ attention < T2)  : 2 LSBs per pixel
    Tier 3  (attention ≥  T2)  : 1 LSB  per pixel  (highest attention)

    Pixels within each tier are sorted by ascending attention value
    (lowest attention gets embedded first).

    Returns
    -------
    stego     : np.ndarray uint8
    side_info : dict  {T1, T2, total_bits, embed_log: [[i,j,k,orig_lsb], ...]}
    """
    stego = image_uint8.copy().astype(np.int32)

    def _sorted_coords(mask):
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return coords
        order = np.argsort(attention_map[coords[:, 0], coords[:, 1]])
        return coords[order]

    t1c = _sorted_coords((attention_map <  T1) & (brain_mask == 1))
    t2c = _sorted_coords((attention_map >= T1) & (attention_map < T2) & (brain_mask == 1))
    t3c = _sorted_coords((attention_map >= T2) & (brain_mask == 1))

    embed_seq = (
        [(int(r), int(c), 3) for r, c in t1c] +
        [(int(r), int(c), 2) for r, c in t2c] +
        [(int(r), int(c), 1) for r, c in t3c]
    )

    total_cap = sum(k for _, _, k in embed_seq)
    if len(secret_bits) > total_cap:
        raise ValueError(
            f"Message too long: {len(secret_bits)} bits needed, "
            f"only {total_cap} available. "
            f"Max ≈ {total_cap // 8} bytes."
        )

    embed_log = []   # [[i, j, k, orig_lsb], ...]
    bit_ptr   = 0

    for (i, j, k) in embed_seq:
        if bit_ptr >= len(secret_bits):
            break

        orig_lsb  = int(image_uint8[i, j]) & ((1 << k) - 1)
        mask_bits = (1 << k) - 1
        chunk     = secret_bits[bit_ptr: bit_ptr + k].ljust(k, '0')
        new_lsbs  = int(chunk, 2)

        stego[i, j] = (int(image_uint8[i, j]) & ~mask_bits) | new_lsbs
        embed_log.append([i, j, k, orig_lsb])
        bit_ptr += k

    stego = np.clip(stego, 0, 255).astype(np.uint8)

    side_info = {
        "T1"         : float(T1),
        "T2"         : float(T2),
        "total_bits" : len(secret_bits),
        "embed_log"  : embed_log,
    }
    return stego, side_info


# ─── EXTRACT  (exact copy of Cell 17) ────────────────────────────────────────

def extract_and_recover(stego_uint8: np.ndarray, side_info: dict) -> tuple:
    """
    Extract hidden text and pixel-perfectly restore the original image.

    Returns
    -------
    recovered_img  : np.ndarray uint8  (identical to original, RCE = 0)
    extracted_text : str
    """
    embed_log  = side_info["embed_log"]
    total_bits = side_info["total_bits"]

    recovered = stego_uint8.copy().astype(np.int32)
    all_bits  = []

    for entry in embed_log:
        i, j, k, orig_lsb = int(entry[0]), int(entry[1]), int(entry[2]), int(entry[3])
        mask_bits = (1 << k) - 1

        embedded_val = int(stego_uint8[i, j]) & mask_bits
        all_bits.append(format(embedded_val, f'0{k}b'))

        recovered[i, j] = (int(stego_uint8[i, j]) & ~mask_bits) | orig_lsb

    recovered = np.clip(recovered, 0, 255).astype(np.uint8)

    extracted_bits = ''.join(all_bits)[:total_bits]
    extracted_text = bits_to_text(extracted_bits)

    return recovered, extracted_text


# ─── METRICS ─────────────────────────────────────────────────────────────────

def calc_psnr(orig: np.ndarray, stego: np.ndarray) -> float:
    mse = np.mean((orig.astype(np.float64) - stego.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return round(10 * np.log10(255.0 ** 2 / mse), 4)


def calc_ssim(orig: np.ndarray, stego: np.ndarray) -> float:
    return round(float(skssim(orig, stego, data_range=255)), 6)


def calc_cnr(img: np.ndarray, brain_mask: np.ndarray) -> float:
    """CNR = |mean_roi − mean_bg| / √(σ²_roi + σ²_bg)"""
    roi = img[brain_mask == 1].astype(np.float64)
    bg  = img[brain_mask == 0].astype(np.float64)
    bg_nz = bg[bg > 0] if bg[bg > 0].size > 0 else np.array([0.0])
    denom = np.sqrt(np.std(roi) ** 2 + np.std(bg_nz) ** 2) + 1e-8
    return round(float(abs(np.mean(roi) - np.mean(bg_nz)) / denom), 4)


def calc_mos(psnr_val: float) -> dict:
    """ITU-T MOS approximation from PSNR."""
    if psnr_val == float('inf') or psnr_val >= 40:
        return {"score": 5, "label": "Excellent"}
    if psnr_val >= 35:
        return {"score": 4, "label": "Good"}
    if psnr_val >= 30:
        return {"score": 3, "label": "Fair"}
    if psnr_val >= 25:
        return {"score": 2, "label": "Poor"}
    return {"score": 1, "label": "Bad"}


def calc_rce(orig: np.ndarray, recovered: np.ndarray) -> dict:
    """
    RCE = pixel differences between original and recovered image.
    With embed_log storing every original LSB, this is always 0.
    """
    diff  = int(np.sum(orig != recovered))
    total = orig.size
    return {
        "pixel_errors"  : diff,
        "rce_percent"   : round(diff / total * 100, 6),
        "is_lossless"   : diff == 0,
    }


def all_metrics(orig, stego, recovered, brain_mask, n_bits) -> dict:
    psnr = calc_psnr(orig, stego)
    ssim = calc_ssim(orig, stego)
    cnr  = calc_cnr(orig, brain_mask)
    mos  = calc_mos(psnr)
    rce  = calc_rce(orig, recovered)
    return {
        "PSNR"          : psnr,
        "SSIM"          : ssim,
        "CNR"           : cnr,
        "MOS_score"     : mos["score"],
        "MOS_label"     : mos["label"],
        "RCE_pixels"    : rce["pixel_errors"],
        "RCE_percent"   : rce["rce_percent"],
        "is_lossless"   : rce["is_lossless"],
        "BPP"           : round(n_bits / (IMG_SIZE * IMG_SIZE), 6),
        "bits_embedded" : n_bits,
    }


# ─── FLASK ROUTES ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/embed", methods=["POST"])
def api_embed():
    try:
        data        = request.get_json(force=True)
        img         = b64_to_img(data["image"])
        secret_text = data["secret_text"]
        T1          = float(data.get("T1", T1_DEFAULT))
        T2          = float(data.get("T2", T2_DEFAULT))

        if not secret_text:
            return jsonify({"status": "error", "message": "Secret text is empty."}), 400

        # ── ViT forward: seg + attention ──────────────────────────────────
        model.eval()
        tensor = (torch.tensor(img / 255.0, dtype=torch.float32)
                  .unsqueeze(0).unsqueeze(0).to(device))
        with torch.no_grad():
            _ = model(tensor)
        attn_map   = model.get_attention_map()

        # ── Brain mask ────────────────────────────────────────────────────
        brain_mask = get_brain_mask(img)

        # ── Capacity check ────────────────────────────────────────────────
        secret_bits = text_to_bits(secret_text)
        n1 = int(((attn_map <  T1) & (brain_mask == 1)).sum())
        n2 = int(((attn_map >= T1) & (attn_map <  T2) & (brain_mask == 1)).sum())
        n3 = int(((attn_map >= T2) & (brain_mask == 1)).sum())
        cap = n1 * 3 + n2 * 2 + n3 * 1

        if len(secret_bits) > cap:
            return jsonify({
                "status" : "error",
                "message": (f"Message too large: {len(secret_bits)} bits needed, "
                            f"only {cap:,} available. "
                            f"Max ≈ {cap // 8:,} characters.")
            }), 400

        # ── Embed ─────────────────────────────────────────────────────────
        stego, side_info = embed_multi_tier(
            img, brain_mask, attn_map, secret_bits, T1=T1, T2=T2
        )

        # ── Extract & verify ──────────────────────────────────────────────
        recovered, recovered_text = extract_and_recover(stego, side_info)
        text_ok = (recovered_text == secret_text)

        # ── Metrics ───────────────────────────────────────────────────────
        metrics = all_metrics(img, stego, recovered, brain_mask, len(secret_bits))

        # ── Attention visualisation ───────────────────────────────────────
        attn_vis   = (attn_map * 255).astype(np.uint8)
        attn_color = cv2.applyColorMap(attn_vis, cv2.COLORMAP_JET)

        # Tier overlay
        tier_vis = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        tier_vis[(attn_map <  T1) & (brain_mask == 1)] = [255, 80, 80]
        tier_vis[(attn_map >= T1) & (attn_map < T2) & (brain_mask == 1)] = [255, 200, 50]
        tier_vis[(attn_map >= T2) & (brain_mask == 1)] = [80, 200, 80]

        # Capacity breakdown
        capacity_info = {
            "tier1_px": n1, "tier1_bits": n1 * 3,
            "tier2_px": n2, "tier2_bits": n2 * 2,
            "tier3_px": n3, "tier3_bits": n3 * 1,
            "total_bits": cap,
            "total_chars": cap // 8,
        }

        return jsonify({
            "status"          : "success",
            "original_image"  : img_to_b64(img),
            "stego_image"     : img_to_b64(stego),
            "recovered_image" : img_to_b64(recovered),
            "attention_map"   : img_to_b64(attn_color),
            "tier_map"        : img_to_b64(tier_vis),
            "brain_mask"      : img_to_b64(brain_mask * 255),
            "metrics"         : metrics,
            "capacity"        : capacity_info,
            "text_verified"   : text_ok,
            "side_info"       : side_info,       # frontend saves this as .json
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/extract", methods=["POST"])
def api_extract():
    try:
        data      = request.get_json(force=True)
        stego     = b64_to_img(data["image"])
        side_info = data["side_info"]     # dict with embed_log, T1, T2, total_bits

        if not isinstance(side_info, dict) or "embed_log" not in side_info:
            return jsonify({
                "status" : "error",
                "message": "Invalid side_info — must include embed_log."
            }), 400

        recovered, extracted_text = extract_and_recover(stego, side_info)

        return jsonify({
            "status"         : "success",
            "extracted_text" : extracted_text,
            "recovered_image": img_to_b64(recovered),
            "n_bits"         : side_info["total_bits"],
            "n_pixels"       : len(side_info["embed_log"]),
            "T1"             : side_info.get("T1", T1_DEFAULT),
            "T2"             : side_info.get("T2", T2_DEFAULT),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/capacity", methods=["POST"])
def api_capacity():
    try:
        data = request.get_json(force=True)
        img  = b64_to_img(data["image"])
        T1   = float(data.get("T1", T1_DEFAULT))
        T2   = float(data.get("T2", T2_DEFAULT))

        model.eval()
        tensor = (torch.tensor(img / 255.0, dtype=torch.float32)
                  .unsqueeze(0).unsqueeze(0).to(device))
        with torch.no_grad():
            _ = model(tensor)
        attn_map   = model.get_attention_map()
        brain_mask = get_brain_mask(img)

        n1 = int(((attn_map <  T1) & (brain_mask == 1)).sum())
        n2 = int(((attn_map >= T1) & (attn_map <  T2) & (brain_mask == 1)).sum())
        n3 = int(((attn_map >= T2) & (brain_mask == 1)).sum())
        cap = n1 * 3 + n2 * 2 + n3 * 1

        return jsonify({
            "status"       : "success",
            "brain_pixels" : int(brain_mask.sum()),
            "tier1_px"     : n1, "tier1_bits": n1 * 3,
            "tier2_px"     : n2, "tier2_bits": n2 * 2,
            "tier3_px"     : n3, "tier3_bits": n3 * 1,
            "total_bits"   : cap,
            "total_chars"  : cap // 8,
            "exceeds_85k"  : cap >= 85000,
            "T1"           : T1, "T2": T2,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    print(f"Starting server on http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)