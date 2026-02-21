import os
import sys
import io
import math
import traceback
from collections import defaultdict

import numpy as np
import pydicom
from pydicom.misc import is_dicom as pydicom_is_dicom
from PIL import Image

import hashlib
import inspect

# --- Compatibility patch for some Python/OpenSSL builds on 3.8 ---
# ReportLab may call hashlib.md5(..., usedforsecurity=False)
# but older/alternate OpenSSL backends reject that kwarg.
if "usedforsecurity" not in inspect.signature(hashlib.md5).parameters:
    _orig_md5 = hashlib.md5

    def _md5_compat(data=b"", **kwargs):
        kwargs.pop("usedforsecurity", None)
        return _orig_md5(data, **kwargs)

    hashlib.md5 = _md5_compat
# --- End patch ---

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from tqdm import tqdm


def is_dicom_file(path: str) -> bool:
    """
    Robust DICOM detection even without extension.
    Tries pydicom's is_dicom, falls back to checking DICM magic word.
    """
    try:
        if pydicom_is_dicom(path):
            return True
    except Exception:
        pass

    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def has_pixels(ds: pydicom.dataset.Dataset) -> bool:
    """
    True if dataset contains pixel data usable for imaging.
    """
    return ("PixelData" in ds) or ("FloatPixelData" in ds) or ("DoubleFloatPixelData" in ds)


def first_value(x):
    # WindowCenter/Width can be MultiValue
    if isinstance(x, pydicom.multival.MultiValue) or isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


def apply_window_gray(arr: np.ndarray, ds: pydicom.dataset.Dataset) -> np.ndarray:
    """
    Apply modality LUT (RescaleSlope/Intercept) + windowing to a *grayscale* 2D array.
    Output uint8 0..255.
    """
    arr = arr.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if wc is not None and ww is not None:
        wc = first_value(wc)
        ww = first_value(ww)
        lo = wc - (ww / 2.0)
        hi = wc + (ww / 2.0)
        arr = np.clip(arr, lo, hi)

    amin = float(np.min(arr))
    amax = float(np.max(arr))
    if math.isclose(amax, amin):
        out = np.zeros_like(arr, dtype=np.uint8)
    else:
        out = (arr - amin) / (amax - amin)
        out = (out * 255.0).clip(0, 255).astype(np.uint8)

    photo = getattr(ds, "PhotometricInterpretation", "").upper()
    if photo == "MONOCHROME1":
        out = 255 - out

    return out


def sort_key(ds: pydicom.dataset.Dataset):
    """
    Prefer geometric ordering when available; fall back to InstanceNumber.
    """
    ipp = getattr(ds, "ImagePositionPatient", None)
    inst = getattr(ds, "InstanceNumber", None)

    if ipp is not None and len(ipp) >= 3:
        # Z position is usually the 3rd coordinate
        try:
            return (0, float(ipp[2]), float(inst) if inst is not None else 0.0)
        except Exception:
            pass

    if inst is not None:
        try:
            return (1, float(inst))
        except Exception:
            pass

    # Last resort: SOPInstanceUID / filename order
    return (2, str(getattr(ds, "SOPInstanceUID", "")))


def dicom_to_png_pages(ds: pydicom.dataset.Dataset) -> list:
    """
    Returns a list of PNG byte strings (one per page/frame).
    Handles:
      - 2D grayscale
      - 3D RGB color (rows, cols, 3)
      - multi-frame grayscale (frames, rows, cols)
      - multi-frame RGB (frames, rows, cols, 3)
    """
    arr = ds.pixel_array

    photo = getattr(ds, "PhotometricInterpretation", "").upper()
    spp = int(getattr(ds, "SamplesPerPixel", 1))
    planar = int(getattr(ds, "PlanarConfiguration", 0))

    pages = []

    def to_png_bytes(pil_img: Image.Image) -> bytes:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    # --- Case 1: 2D grayscale (rows, cols)
    if arr.ndim == 2:
        img8 = apply_window_gray(arr, ds)
        pil = Image.fromarray(img8, mode="L")
        pages.append(to_png_bytes(pil))
        return pages

    # --- Case 2: single-frame color (rows, cols, 3)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        # If planar configuration is 1 (rare), pydicom typically already outputs interleaved;
        # but we handle just in case.
        pil = Image.fromarray(arr.astype(np.uint8))
        # You can keep color; for CT/MRI it’s usually not needed, so convert to RGB for stability
        if pil.mode not in ("RGB", "RGBA"):
            pil = pil.convert("RGB")
        pages.append(to_png_bytes(pil))
        return pages

    # --- Case 3: multi-frame grayscale (frames, rows, cols)
    if arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[-1] not in (3, 4):
        for i in range(arr.shape[0]):
            frame = arr[i]
            img8 = apply_window_gray(frame, ds)
            pil = Image.fromarray(img8, mode="L")
            pages.append(to_png_bytes(pil))
        return pages

    # --- Case 4: multi-frame color (frames, rows, cols, 3/4)
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        for i in range(arr.shape[0]):
            frame = arr[i].astype(np.uint8)
            pil = Image.fromarray(frame)
            if pil.mode not in ("RGB", "RGBA"):
                pil = pil.convert("RGB")
            pages.append(to_png_bytes(pil))
        return pages

    raise ValueError(f"Unhandled pixel array shape: {arr.shape}, PhotometricInterpretation={photo}, SamplesPerPixel={spp}, PlanarConfiguration={planar}")


def write_series_pdf(datasets, out_pdf_path: str):
    """
    Write one multi-page PDF where each page fits the image size exactly (no rescaling).
    """
    if not datasets:
        return

    # Get first page from first dataset (handles multiframe too)
    first_pages = dicom_to_png_pages(datasets[0])
    first_img = Image.open(io.BytesIO(first_pages[0]))
    w_px, h_px = first_img.size
    page_size = (w_px, h_px)

    c = canvas.Canvas(out_pdf_path, pagesize=page_size, pageCompression=1)

    for ds in tqdm(datasets, desc=os.path.basename(out_pdf_path)):
        try:
            png_pages = dicom_to_png_pages(ds)
            for png_bytes in png_pages:
                img_reader = ImageReader(io.BytesIO(png_bytes))
                c.drawImage(img_reader, 0, 0, width=w_px, height=h_px)
                c.showPage()
        except Exception:
            print(f"  [WARN] Skipping object due to error. SOP={getattr(ds, 'SOPInstanceUID', 'unknown')}")
            # traceback.print_exc()
            continue

    c.save()


def main(input_dir: str, output_dir: str = None):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir or input_dir)

    print(f"Scanning: {input_dir}")
    files = []
    for root, _, fnames in os.walk(input_dir):
        for fn in fnames:
            path = os.path.join(root, fn)
            if is_dicom_file(path):
                files.append(path)

    if not files:
        print("No DICOM files found.")
        return

    print(f"Found {len(files)} DICOM-looking files. Reading headers...")

    series = defaultdict(list)
    skipped_non_image = 0
    skipped_read_errors = 0

    # Read minimal first (no pixels) to group quickly
    for path in tqdm(files, desc="Indexing"):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            uid = getattr(ds, "SeriesInstanceUID", None)
            if not uid:
                continue
            series[uid].append(path)
        except Exception:
            skipped_read_errors += 1
            continue

    if not series:
        print("No SeriesInstanceUID found in files.")
        return

    print(f"Detected {len(series)} series. Now converting each series to PDF...")

    os.makedirs(output_dir, exist_ok=True)

    for uid, paths in series.items():
        datasets = []
        for path in paths:
            try:
                ds = pydicom.dcmread(path, force=True)
                if not has_pixels(ds):
                    skipped_non_image += 1
                    continue
                # Some objects have NumberOfFrames>1; this script treats them as one image per file.
                # If you have true multi-frame DICOM, tell me and I'll expand frames.
                datasets.append(ds)
            except Exception:
                skipped_read_errors += 1
                continue

        if not datasets:
            print(f"[INFO] Series {uid}: no image slices (skipped).")
            continue

        # Sort slices
        datasets.sort(key=sort_key)

        out_name = f"Series_{uid}.pdf"
        out_path = os.path.join(output_dir, out_name)

        print(f"\n[OK] Series {uid}: {len(datasets)} image slices -> {out_path}")
        write_series_pdf(datasets, out_path)

    print("\nDone.")
    if skipped_non_image:
        print(f"Skipped non-image DICOM objects (no PixelData): {skipped_non_image}")
    if skipped_read_errors:
        print(f"Skipped files due to read/decode errors: {skipped_read_errors}")
    print("If you see decode errors, install the optional decoders (pylibjpeg/gdcm).")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python dicom_to_pdf.py <input_dir> [output_dir]")
        sys.exit(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) == 3 else None
    main(in_dir, out_dir)

