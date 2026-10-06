"""Discrete Wavelet Transform (DWT) 2D image codec with custom canonical Huffman coding.

Features:
- Arbitrary dimension padding to multiple of 2**levels (edge replicate).
- Forward 2D DWT using pywt.wavedec2 with selectable wavelets (default: 'bior4.4') and levels (default: 4).
- Uniform dead-zone quantization: q = sign(c) * floor(|c| / step).
  Base step = 2 ** ((100 - quality) / 12.5).
  Subband step = base_step * weight.
  Default weights: LL = 0.5, coarsest detail = 1.0, finer levels scaled by 1.4, HH extra 1.2x.
- Inverse DWT reconstruction with reconstruction bias (default: 0.25):
  value = sign(q) * (|q| + bias) * step for q != 0, and 0 for q == 0.
- LL subband entropy coding: 2D DPCM raster scan differences (left neighbor, first of row from above)
  coded as (size category + amplitude bits) with dedicated canonical Huffman table.
- Detail subbands entropy coding: raster scan per subband, (zero-run, size) symbols with EOB and ZRL (runs > 15),
  plus amplitude bits. Single shared table or optional per-level tables.
- Self-contained binary header with magic bytes ('MDWT'), image dimensions, wavelet metadata,
  quantization parameters, and serialized canonical Huffman code-length tables.
- Bit-exact reconstruction between encoder internal recon and standalone decoder.
"""

from __future__ import annotations

import struct
from collections import Counter
from typing import Sequence

import numpy as np
import pywt

from .entropy import (
    BitReader,
    BitWriter,
    average_code_length,
    build_canonical_decode_trie,
    build_huffman_code_lengths,
    canonical_codes,
    decode_amplitude,
    decode_huffman_symbol,
    deserialize_code_lengths,
    encode_amplitude,
    serialize_code_lengths,
    shannon_entropy_from_freqs,
)
from .io_utils import crop_to_shape, pad_to_multiple

# Magic identifier for MedComp DWT bitstream
MAGIC_DWT = b"MDWT"

# Known wavelet ID mappings for compact header representation
WAVELET_TO_ID: dict[str, int] = {
    "bior4.4": 1,
    "haar": 2,
    "bior2.2": 3,
    "db2": 4,
}
ID_TO_WAVELET: dict[int, str] = {v: k for k, v in WAVELET_TO_ID.items()}


def get_default_dwt_weights(levels: int = 4) -> dict[str, float]:
    """Generate default subband quantization weights for multi-level 2D DWT.

    Subband weight defaults:
    - LL = 0.5 (fine precision for low-frequency structural content)
    - Detail bands at coarsest level (level = levels): weight = 1.0 (HH = 1.2)
    - Each finer detail level scaled by factor of 1.4:
      scale = 1.4 ** (levels - level)
      cH, cV = 1.0 * scale
      cD (HH) = 1.2 * scale

    Args:
        levels: Number of decomposition levels.

    Returns:
        dict[str, float]: Mapping of subband identifier to multiplier weight.
    """
    weights: dict[str, float] = {"LL": 0.5}
    for j, lev in enumerate(range(levels, 0, -1)):
        scale = 1.4 ** j
        weights[f"cH_{lev}"] = float(1.0 * scale)
        weights[f"cV_{lev}"] = float(1.0 * scale)
        weights[f"cD_{lev}"] = float(1.2 * scale)
    return weights


def _resolve_step(
    base_step: float,
    subband_name: str,
    level: int | None,
    weights: dict[str, float] | None,
    default_weights: dict[str, float],
) -> float:
    """Resolve subband quantization step from base step and weight tables."""
    if weights is not None:
        # Check specific key first, then fallback
        if level is not None and f"{subband_name}_{level}" in weights:
            return float(base_step * weights[f"{subband_name}_{level}"])
        if subband_name in weights:
            return float(base_step * weights[subband_name])

    key = f"{subband_name}_{level}" if level is not None else subband_name
    return float(base_step * default_weights.get(key, default_weights.get(subband_name, 1.0)))


def dwt_encode_with_stats(
    img: np.ndarray,
    quality: int = 50,
    wavelet: str = "bior4.4",
    levels: int = 4,
    bias: float = 0.25,
    per_level_tables: bool = False,
    weights: dict[str, float] | None = None,
) -> tuple[bytes, np.ndarray, dict[str, float]]:
    """Compress a 2D grayscale image using 2D DWT, dead-zone quantization, and custom Huffman coding.

    Args:
        img: 2D uint8 numpy array.
        quality: Compression quality factor (1-100).
        wavelet: Wavelet name (default 'bior4.4').
        levels: Number of decomposition levels (default 4).
        bias: Reconstruction dead-zone bias (default 0.25).
        per_level_tables: If True, uses separate Huffman tables for each detail level.
        weights: Optional dictionary overriding default subband weights.

    Returns:
        tuple[bytes, np.ndarray, dict[str, float]]:
            - bitstream: Fully serialized compressed bytes including headers.
            - recon: Reconstructed uint8 image matching encoder dequantization.
            - stats: Detailed entropy and coding statistics.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(
            f"Expected 2D uint8 array, got shape={getattr(img, 'shape', None)}, "
            f"dtype={getattr(img, 'dtype', None)}"
        )
    if not (1 <= quality <= 100):
        raise ValueError(f"Quality must be in [1, 100], got {quality}")
    if levels < 1:
        raise ValueError(f"Levels must be >= 1, got {levels}")

    orig_h, orig_w = img.shape
    default_weights = get_default_dwt_weights(levels)

    # 1. Pad to multiple of 2**levels (edge replicate)
    multiple = 2 ** levels
    padded, orig_shape = pad_to_multiple(img, multiple=multiple, mode="edge")

    # 2. Level shift (subtract 128)
    shifted = padded.astype(np.float64) - 128.0

    # 3. Wavelet decomposition
    coeffs = pywt.wavedec2(shifted, wavelet=wavelet, level=levels, mode="symmetric")
    ll_coeff = coeffs[0]
    detail_coeffs = coeffs[1:]  # From coarsest (level = levels) to finest (level = 1)

    # 4. Quantization
    base_step = float(2.0 ** ((100.0 - float(quality)) / 12.5))

    # 4a. Quantize LL subband
    ll_step = _resolve_step(base_step, "LL", None, weights, default_weights)
    q_ll = np.int64(np.sign(ll_coeff) * np.floor(np.abs(ll_coeff) / ll_step))
    recon_ll = np.where(q_ll != 0, np.sign(q_ll) * (np.abs(q_ll) + bias) * ll_step, 0.0)

    # 4b. Quantize detail subbands
    q_details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    recon_details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for idx, (cH, cV, cD) in enumerate(detail_coeffs):
        lev = levels - idx
        s_cH = _resolve_step(base_step, "cH", lev, weights, default_weights)
        s_cV = _resolve_step(base_step, "cV", lev, weights, default_weights)
        s_cD = _resolve_step(base_step, "cD", lev, weights, default_weights)

        q_cH = np.int64(np.sign(cH) * np.floor(np.abs(cH) / s_cH))
        q_cV = np.int64(np.sign(cV) * np.floor(np.abs(cV) / s_cV))
        q_cD = np.int64(np.sign(cD) * np.floor(np.abs(cD) / s_cD))

        r_cH = np.where(q_cH != 0, np.sign(q_cH) * (np.abs(q_cH) + bias) * s_cH, 0.0)
        r_cV = np.where(q_cV != 0, np.sign(q_cV) * (np.abs(q_cV) + bias) * s_cV, 0.0)
        r_cD = np.where(q_cD != 0, np.sign(q_cD) * (np.abs(q_cD) + bias) * s_cD, 0.0)

        q_details.append((q_cH, q_cV, q_cD))
        recon_details.append((r_cH, r_cV, r_cD))

    # Compute internal encoder reconstruction
    recon_coeffs = [recon_ll] + [tuple(d) for d in recon_details]
    rec_padded = pywt.waverec2(recon_coeffs, wavelet=wavelet, mode="symmetric") + 128.0
    rec_cropped = crop_to_shape(rec_padded, (orig_h, orig_w))
    recon_img = np.clip(np.round(rec_cropped), 0, 255).astype(np.uint8)

    # 5. Entropy coding preparation: LL band 2D DPCM
    h_ll, w_ll = q_ll.shape
    ll_diffs: list[int] = []
    ll_tokens: list[tuple[int, int]] = []  # (category, amp_bits)
    ll_categories: list[int] = []

    for r in range(h_ll):
        for c in range(w_ll):
            val = int(q_ll[r, c])
            if r == 0 and c == 0:
                pred = 0
            elif c == 0:
                pred = int(q_ll[r - 1, 0])
            else:
                pred = int(q_ll[r, c - 1])
            diff = val - pred
            ll_diffs.append(diff)
            cat = abs(diff).bit_length()
            amp_bits = encode_amplitude(diff, cat)
            ll_tokens.append((cat, amp_bits))
            ll_categories.append(cat)

    ll_freqs = Counter(ll_categories)
    ll_lengths = build_huffman_code_lengths(ll_freqs)
    ll_codes = canonical_codes(ll_lengths)

    # 6. Entropy coding preparation: Detail bands
    # Detail tokens: list per subband of (sym, sz, amp_bits)
    # where sym = (run << 4) | sz, EOB = 0x00, ZRL = 0xF0
    def _tokenize_subband(arr: np.ndarray) -> tuple[list[tuple[int, int, int]], list[int]]:
        flat = arr.ravel()
        n = len(flat)
        toks: list[tuple[int, int, int]] = []
        syms: list[int] = []
        r = 0
        for i in range(n):
            v = int(flat[i])
            if v == 0:
                r += 1
            else:
                while r >= 16:
                    toks.append((0xF0, 0, 0))  # ZRL
                    syms.append(0xF0)
                    r -= 16
                sz = abs(v).bit_length()
                if sz > 15:
                    sz = 15  # Fallback safety
                sym = (r << 4) | sz
                amp = encode_amplitude(v, sz)
                toks.append((sym, sz, amp))
                syms.append(sym)
                r = 0
        if r > 0:
            toks.append((0x00, 0, 0))  # EOB
            syms.append(0x00)
        return toks, syms

    # Tokenize each detail level and subband
    level_tokens: list[list[list[tuple[int, int, int]]]] = []
    level_symbols: list[list[int]] = []
    all_detail_symbols: list[int] = []

    for q_cH, q_cV, q_cD in q_details:
        lev_toks: list[list[tuple[int, int, int]]] = []
        lev_syms: list[int] = []
        for band in (q_cH, q_cV, q_cD):
            t, s = _tokenize_subband(band)
            lev_toks.append(t)
            lev_syms.extend(s)
            all_detail_symbols.extend(s)
        level_tokens.append(lev_toks)
        level_symbols.append(lev_syms)

    # Build Huffman tables for detail bands
    detail_lengths_per_level: list[dict[int, int]] = []
    detail_codes_per_level: list[dict[int, tuple[int, int]]] = []
    global_detail_lengths: dict[int, int] = {}
    global_detail_codes: dict[int, tuple[int, int]] = {}

    if per_level_tables:
        for lev_syms in level_symbols:
            frq = Counter(lev_syms)
            lens = build_huffman_code_lengths(frq)
            detail_lengths_per_level.append(lens)
            detail_codes_per_level.append(canonical_codes(lens))
    else:
        g_frq = Counter(all_detail_symbols)
        global_detail_lengths = build_huffman_code_lengths(g_frq)
        global_detail_codes = canonical_codes(global_detail_lengths)

    # 7. Entropy stats
    ll_entropy = shannon_entropy_from_freqs(ll_freqs)
    ll_avg_len = average_code_length(ll_freqs, ll_lengths)

    if per_level_tables:
        tot_syms = sum(len(s) for s in level_symbols)
        if tot_syms > 0:
            detail_entropy = sum(
                shannon_entropy_from_freqs(Counter(s)) * len(s) for s in level_symbols
            ) / tot_syms
            detail_avg_len = sum(
                average_code_length(Counter(s), l) * len(s)
                for s, l in zip(level_symbols, detail_lengths_per_level)
            ) / tot_syms
        else:
            detail_entropy, detail_avg_len = 0.0, 0.0
    else:
        g_frq = Counter(all_detail_symbols)
        detail_entropy = shannon_entropy_from_freqs(g_frq)
        detail_avg_len = average_code_length(g_frq, global_detail_lengths)

    stats = {
        "ll_entropy": round(ll_entropy, 4),
        "ll_avg_code_length": round(ll_avg_len, 4),
        "detail_entropy": round(detail_entropy, 4),
        "detail_avg_code_length": round(detail_avg_len, 4),
    }

    # 8. Encode bitstream payload
    writer = BitWriter()

    # 8a. Write LL band
    for cat, amp_bits in ll_tokens:
        code, length = ll_codes[cat]
        writer.write_bits(code, length)
        if cat > 0:
            writer.write_bits(amp_bits, cat)

    # 8b. Write detail bands
    for lev_idx, lev_toks in enumerate(level_tokens):
        codes = detail_codes_per_level[lev_idx] if per_level_tables else global_detail_codes
        for sub_toks in lev_toks:
            for sym, sz, amp_bits in sub_toks:
                code, length = codes[sym]
                writer.write_bits(code, length)
                if sz > 0:
                    writer.write_bits(amp_bits, sz)

    payload = writer.pad_and_flush()

    # 9. Build header
    header = bytearray(MAGIC_DWT)
    w_id = WAVELET_TO_ID.get(wavelet.lower(), 0xFF)
    flags = 1 if per_level_tables else 0

    header.extend(
        struct.pack(
            ">HHBBBfB",
            orig_h,
            orig_w,
            w_id,
            levels,
            quality,
            float(bias),
            flags,
        )
    )

    if w_id == 0xFF:
        w_bytes = wavelet.encode("utf-8")
        header.extend(struct.pack(">B", len(w_bytes)))
        header.extend(w_bytes)

    # Serialize Huffman code-length tables
    header.extend(serialize_code_lengths(ll_lengths))

    if per_level_tables:
        for lens in detail_lengths_per_level:
            header.extend(serialize_code_lengths(lens))
    else:
        header.extend(serialize_code_lengths(global_detail_lengths))

    total_bitstream = bytes(header) + payload
    return total_bitstream, recon_img, stats


def dwt_encode(
    img: np.ndarray,
    quality: int = 50,
    wavelet: str = "bior4.4",
    levels: int = 4,
    bias: float = 0.25,
    per_level_tables: bool = False,
    weights: dict[str, float] | None = None,
) -> tuple[bytes, np.ndarray]:
    """Compress a 2D uint8 image using DWT and return (compressed_bytes, reconstructed_image)."""
    bitstream, recon, _ = dwt_encode_with_stats(
        img=img,
        quality=quality,
        wavelet=wavelet,
        levels=levels,
        bias=bias,
        per_level_tables=per_level_tables,
        weights=weights,
    )
    return bitstream, recon


def dwt_decode(
    data: bytes,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Decompress a complete DWT bitstream into a 2D uint8 image.

    Parses only the bytes (header + entropy bitstream) and reconstructs the image.

    Args:
        data: Compressed byte stream.
        weights: Optional subband weight overrides matching encoder.

    Returns:
        np.ndarray: Reconstructed 2D uint8 array matching the original dimensions.

    Raises:
        ValueError: If header magic is invalid or bitstream is corrupt.
    """
    if len(data) < 15:
        raise ValueError("Data stream too short to contain valid DWT header.")

    if data[:4] != MAGIC_DWT:
        raise ValueError(f"Invalid magic bytes: expected {MAGIC_DWT!r}, got {data[:4]!r}")

    offset = 4
    orig_h, orig_w, w_id, levels, quality, bias, flags = struct.unpack_from(
        ">HHBBBfB", data, offset
    )
    offset += struct.calcsize(">HHBBBfB")

    per_level_tables = bool(flags & 1)

    if w_id == 0xFF:
        (name_len,) = struct.unpack_from(">B", data, offset)
        offset += 1
        wavelet = data[offset : offset + name_len].decode("utf-8")
        offset += name_len
    else:
        wavelet = ID_TO_WAVELET.get(w_id, "bior4.4")

    # Deserialize LL Huffman table
    ll_lengths, consumed = deserialize_code_lengths(data, offset)
    offset += consumed

    # Deserialize Detail Huffman tables
    detail_lengths_per_level: list[dict[int, int]] = []
    global_detail_lengths: dict[int, int] = {}

    if per_level_tables:
        for _ in range(levels):
            lens, consumed = deserialize_code_lengths(data, offset)
            offset += consumed
            detail_lengths_per_level.append(lens)
    else:
        global_detail_lengths, consumed = deserialize_code_lengths(data, offset)
        offset += consumed

    # Reconstruct subband shapes from padded dimensions
    multiple = 2 ** levels
    pad_h = ((orig_h + multiple - 1) // multiple) * multiple
    pad_w = ((orig_w + multiple - 1) // multiple) * multiple

    dummy = np.zeros((pad_h, pad_w), dtype=np.float32)
    dummy_coeffs = pywt.wavedec2(dummy, wavelet=wavelet, level=levels, mode="symmetric")
    ll_shape = dummy_coeffs[0].shape
    detail_shapes = [(cH.shape, cV.shape, cD.shape) for cH, cV, cD in dummy_coeffs[1:]]

    # Pre-build decode tries
    ll_trie = build_canonical_decode_trie(ll_lengths)
    if per_level_tables:
        detail_tries = [build_canonical_decode_trie(lens) for lens in detail_lengths_per_level]
    else:
        g_trie = build_canonical_decode_trie(global_detail_lengths)

    # Initialize BitReader on bitstream payload
    reader = BitReader(data, byte_offset=offset)

    # 1. Decode LL band
    h_ll, w_ll = ll_shape
    n_ll = h_ll * w_ll
    q_ll = np.empty((h_ll, w_ll), dtype=np.int64)

    idx = 0
    for r in range(h_ll):
        for c in range(w_ll):
            cat = decode_huffman_symbol(reader, ll_trie)
            if cat > 0:
                amp_code = reader.read_bits(cat)
                diff = decode_amplitude(amp_code, cat)
            else:
                diff = 0

            if r == 0 and c == 0:
                pred = 0
            elif c == 0:
                pred = int(q_ll[r - 1, 0])
            else:
                pred = int(q_ll[r, c - 1])
            q_ll[r, c] = pred + diff
            idx += 1

    # 2. Decode Detail subbands
    def _decode_subband(target_shape: tuple[int, int], trie: object) -> np.ndarray:
        h_s, w_s = target_shape
        total_coeffs = h_s * w_s
        out_coeffs: list[int] = []
        while len(out_coeffs) < total_coeffs:
            sym = decode_huffman_symbol(reader, trie)
            if sym == 0x00:  # EOB
                out_coeffs.extend([0] * (total_coeffs - len(out_coeffs)))
                break
            if sym == 0xF0:  # ZRL (16 zeros)
                out_coeffs.extend([0] * 16)
                continue
            r = sym >> 4
            sz = sym & 0x0F
            out_coeffs.extend([0] * r)
            if sz > 0:
                amp_code = reader.read_bits(sz)
                val = decode_amplitude(amp_code, sz)
                out_coeffs.append(val)
            else:
                out_coeffs.append(0)

        if len(out_coeffs) > total_coeffs:
            out_coeffs = out_coeffs[:total_coeffs]
        return np.array(out_coeffs, dtype=np.int64).reshape((h_s, w_s))

    decoded_details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for lev_idx, (cH_shape, cV_shape, cD_shape) in enumerate(detail_shapes):
        trie = detail_tries[lev_idx] if per_level_tables else g_trie
        q_cH = _decode_subband(cH_shape, trie)
        q_cV = _decode_subband(cV_shape, trie)
        q_cD = _decode_subband(cD_shape, trie)
        decoded_details.append((q_cH, q_cV, q_cD))

    # 3. Dequantization
    default_weights = get_default_dwt_weights(levels)
    base_step = float(2.0 ** ((100.0 - float(quality)) / 12.5))

    ll_step = _resolve_step(base_step, "LL", None, weights, default_weights)
    recon_ll = np.where(q_ll != 0, np.sign(q_ll) * (np.abs(q_ll) + bias) * ll_step, 0.0)

    recon_details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for idx, (q_cH, q_cV, q_cD) in enumerate(decoded_details):
        lev = levels - idx
        s_cH = _resolve_step(base_step, "cH", lev, weights, default_weights)
        s_cV = _resolve_step(base_step, "cV", lev, weights, default_weights)
        s_cD = _resolve_step(base_step, "cD", lev, weights, default_weights)

        r_cH = np.where(q_cH != 0, np.sign(q_cH) * (np.abs(q_cH) + bias) * s_cH, 0.0)
        r_cV = np.where(q_cV != 0, np.sign(q_cV) * (np.abs(q_cV) + bias) * s_cV, 0.0)
        r_cD = np.where(q_cD != 0, np.sign(q_cD) * (np.abs(q_cD) + bias) * s_cD, 0.0)

        recon_details.append((r_cH, r_cV, r_cD))

    # 4. Wavelet reconstruction
    recon_coeffs = [recon_ll] + [tuple(d) for d in recon_details]
    rec_padded = pywt.waverec2(recon_coeffs, wavelet=wavelet, mode="symmetric") + 128.0
    rec_cropped = crop_to_shape(rec_padded, (orig_h, orig_w))
    return np.clip(np.round(rec_cropped), 0, 255).astype(np.uint8)
