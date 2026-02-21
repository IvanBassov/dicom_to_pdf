# DICOM to PDF Converter

A robust Python tool for converting large collections of DICOM files into **one PDF per imaging series**, preserving full diagnostic resolution.

Designed for real-world hospital exports — including DICOM files **without extensions** — and capable of handling large CT/MRI datasets (1,000+ slices).

---

## Why This Exists

A foreign hospital provided imaging data on a CD containing over **1,500 DICOM files**. None of them had file extensions.

The receiving U.S. insurance company refused to accept:

- The physical CD (“security reasons”)
- Email attachments
- Cloud uploads

The only options they offered were:

- Fax the images  
- Upload them as a PDF  

Faxing diagnostic CT/MRI scans in 2026 is not a serious option.

Various commercial DICOM tools were tested. Most failed to recognize extensionless files. None handled reliable bulk conversion into organized PDFs — especially one PDF per imaging series.

So the only practical solution was to write a script.

This tool was built to:

- Detect DICOM files even without `.dcm`
- Automatically group them into imaging series
- Preserve diagnostic resolution
- Export clean, organized multi-page PDFs
- Work offline
- Avoid proprietary software

It is now made available publicly in case others encounter the same bureaucratic dead end.

---

## Features

- Recursively scans a directory for DICOM files (extension not required)
- Detects valid DICOM objects
- Groups images by `SeriesInstanceUID`
- Skips non-image DICOM objects (e.g., Structured Reports, Presentation States)
- Sorts slices correctly using:
  - `ImagePositionPatient` (preferred)
  - `InstanceNumber` (fallback)
- Handles:
  - 2D grayscale images
  - RGB color images
  - Multi-frame DICOM datasets
- Applies:
  - Rescale Slope / Intercept (modality LUT)
  - Window Center / Width (if present)
- Preserves full pixel resolution (no downscaling)
- Exports one multi-page PDF per detected DICOM series

---

## Requirements

- Python 3.8+

### Required Python Packages

Install required dependencies:

```bash
pip install pydicom pillow numpy reportlab tqdm
```

### Optional (Recommended) Decoders

Some CT/MRI datasets use compressed transfer syntaxes (JPEG, JPEG2000, RLE).  
If you encounter pixel decode errors, install:

```bash
pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg gdcm
```
---

## Usage
### Convert a directory (PDFs created in the same directory)
```bash
python dicom_to_pdf.py /path/to/dicom_folder
```

### Convert a directory and specify output directory
```bash
python dicom_to_pdf.py /path/to/dicom_folder /path/to/output_folder
```
---

## Output

For each detected imaging series, the program creates:

`Series_<SeriesInstanceUID>.pdf`

Each PDF contains all slices of that series, in correct anatomical order.

Non-image DICOM objects are automatically skipped.

---

## Notes

- Full diagnostic resolution is preserved (no image downscaling).
- PDF stream compression is enabled (does not affect resolution).
- Large CT/MRI series may produce large PDFs — this is expected.
- Multi-frame DICOM files are expanded into individual PDF pages.
- The tool is intended for submission and archival workflows.

---

## Limitations

- This tool is not a replacement for a medical-grade DICOM viewer.
- Interactive window/level adjustments are not supported in PDF output.
- The script assumes standard DICOM compliance.

---

## Practical Context

DICOM is a powerful medical imaging standard.

Administrative systems often are not.

This tool bridges that gap — converting raw imaging data into a format that bureaucratic systems can accept, without degrading image fidelity.

If you find yourself being asked to “just upload a PDF,” this script exists for you.

---

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.
