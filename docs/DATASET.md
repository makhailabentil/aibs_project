# CASIA-WebFace Dataset

**Reference:** [3] in project proposal.

## Setup checklist (get everything running)

1. **Clone the repo** and open a terminal in the project root (`aibs_project/`).
2. **Install the dataset helper** (once):  
   `pip install kagglehub`
3. **Run the setup script:**  
   `python scripts/get_dataset.py --kaggle`
4. **Confirm layout:**  
   - `data/casia_webface/` should contain `train.rec`, `train.idx`, `train.lst`, `property` (record format; no loose .jpg files).  
   - `data/eval/` should contain evaluation bins (e.g. `lfw.bin`, `agedb_30.bin`).  
   Use an MXNet/RecordIO loader in your code to read images from `train.rec`, or extract to folders (see below).

5. **Optional: get folder-per-identity .jpg**  
   If you want loose `.jpg` files (one folder per identity) instead of using the record file in code, run:  
   `python scripts/extract_rec_to_folders.py`  
   Output: `data/casia_webface_extracted/` (e.g. `0000045/001.jpg`, `0000099/086.jpg`). Full extraction takes about 30–60 minutes and needs **about 2.5–3 GB** extra disk space (on top of the existing `train.rec`). Use `--limit 5000` to test with fewer images first.

---

## Getting folder-per-identity .jpg (optional)

If any team member prefers working with loose image files instead of the packed `train.rec`:

1. **Prerequisite:** Run the main dataset setup first (`python scripts/get_dataset.py --kaggle`) so `data/casia_webface/train.rec` and `train.lst` exist.
2. **From the project root**, run:
   ```bash
   python scripts/extract_rec_to_folders.py
   ```
3. **Output:** `data/casia_webface_extracted/` with one folder per identity and `.jpg` files inside (e.g. `0000045/001.jpg`, `0000045/002.jpg`).
4. **Time and disk:** Full extraction (~494k images) usually takes 30–60 minutes and needs **about 2.5–3 GB** extra disk space (the extracted .jpg files take roughly the same space as `train.rec`). Ensure you have enough free space before running. To try a smaller set first:  
   `python scripts/extract_rec_to_folders.py --limit 5000`

---

## Overview

| Property | Value |
|----------|--------|
| **Identities** | ~10,575 |
| **Images** | ~494,414 face images |
| **Source** | Collected from the Internet |
| **Structure** | One folder per person, multiple images per identity |
| **Variation** | Pose, expression, illumination, resolution (real-world conditions) |

## Suitability for This Project

- Provides **identity labels** and **multiple images per identity**, which is required for face verification (Face ID).
- Not oriented toward attribute classification (e.g. gender, race, age); focus is on identity.

---

## How to Get the Dataset (For Each Team Member)

The dataset is **not** in this repo (too large for GitHub). Each person runs the setup script locally.

**Quick start (Kaggle):** `pip install kagglehub` then from project root: `python scripts/get_dataset.py --kaggle`

### Option A: Kaggle (recommended, one command)

1. **Install the helper** (once):
   ```bash
   pip install kagglehub
   ```

2. **Optional: Kaggle API**  
   If the dataset asks for login, set up [Kaggle API](https://github.com/Kaggle/kaggle-api): create an API token in your Kaggle account (Account → Create New Token), then place `kaggle.json` in your user folder as described in the Kaggle API docs.

3. **From the project root** (`aibs_project/`), run:
   ```bash
   python scripts/get_dataset.py --kaggle
   ```
   This downloads from Kaggle (`debarghamitraroy/casia-webface`), extracts, and copies into `data/casia_webface/` and `data/eval/` (evaluation bins).

4. **To save disk space** (link instead of copy; dataset stays in Kaggle cache):
   ```bash
   python scripts/get_dataset.py --kaggle --symlink
   ```
   On Windows, creating symlinks often requires administrator rights; if the command fails, use the default (copy) without `--symlink`.

**Note:** The Kaggle version may be a subset or in **record format** (e.g. `train.rec`, `train.idx`, `train.lst`). The script detects this and copies those files into `data/casia_webface/`. To get **folder-per-identity .jpg** from the record file, run `python scripts/extract_rec_to_folders.py` (writes to `data/casia_webface_extracted/`). For other sources (archive or pre-extracted folder), use Option B or C below.

---

### Option B: Local archive (zip or tar.gz)

If you already have the dataset as a zip or tar.gz (e.g. from the official CASIA site):

1. From the **project root**, run:
   ```bash
   python scripts/get_dataset.py --archive "C:\path\to\your\casia_webface.zip"
   ```
   (Use your actual path; on Mac/Linux use `/path/to/archive.tar.gz`.)

2. The script unpacks into `data/casia_webface/`.

---

### Option C: Already extracted folder

If you have a folder where each subfolder is one identity (with images inside):

1. From the **project root**, run:
   ```bash
   python scripts/get_dataset.py --extracted "C:\path\to\casia_webface_folder"
   ```
   To symlink instead of copy (saves space, folder must stay in place):
   ```bash
   python scripts/get_dataset.py --extracted "C:\path\to\casia_webface_folder" --symlink
   ```

---

### After running the script

- The script writes (or links) into **`data/casia_webface/`** and **`data/eval/`** (evaluation bins at the root of `data/`), and runs a quick layout check when the layout is folder-per-identity.
- It prints the path when done. Use **`data/casia_webface/`** as the training dataset root in your code; use **`data/eval/`** for evaluation bins (e.g. LFW, AgeDB-30).

**Skip validation** (e.g. for a small subset):
```bash
python scripts/get_dataset.py --kaggle --no-validate
```

---

## Expected Layout After Setup

**If you used Option A (Kaggle),** you get record format (no loose .jpg files):

```
data/
├── casia_webface/
│   ├── property
│   ├── train.idx
│   ├── train.lst
│   └── train.rec
└── eval/
    ├── agedb_30.bin
    ├── lfw.bin
    └── ... (other .bin evaluation sets)
```

Images are packed inside `train.rec`; load them in code with an MXNet/RecordIO (or compatible) reader.

**If you also ran `extract_rec_to_folders.py`** (optional), you get folder-per-identity .jpg in a separate tree:

```
data/
├── casia_webface/          # record format (above)
├── casia_webface_extracted/
│   ├── 0000045/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   ├── 0000099/
│   │   └── ...
│   └── ...
└── eval/
```

Use `data/casia_webface_extracted/` as the dataset root when you want to work with loose .jpg files.

**If you used Option B or C** (archive or extracted folder with folder-per-identity), you get:

```
data/
└── casia_webface/
    ├── id_0001/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── id_0002/
    │   └── ...
    └── ...
```

The script accepts both layouts where the archive root is directly identity folders, or where there is one top-level folder containing them.

---

## Usage in This Project

- **Training/validation:** Use for training the face embedding/verification model (Option I or II) and for threshold calibration.
- **Held-out split:** Reserve a subset of identities for verification pairs and clean benchmark (Task 2); do not use this split for training or threshold tuning.
- **Adversarial evaluation:** Generate verification pairs from the same or a consistent split when evaluating PGD, C&W, and transfer attacks (Tasks 3–4).

## Data Splits (To Define)

- [ ] Define train / validation / test (held-out) identity splits.
- [ ] Document split IDs or paths (e.g. in `data/splits/` or in code) so all tasks use the same splits.

## License and Citation

Use and cite the dataset according to the terms and citation provided by the CASIA-WebFace authors.
