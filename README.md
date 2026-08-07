# PlexAutoImport

An automatic way to move all your memories off your camera and onto your storage location.

Photos and videos are tagged with AI-detected keywords (categories, and optionally the names
of people recognised from your reference photos), then filed into `Photos from <year>` folders.

## Requirements & Setup

This project uses `PyExifTool` to read and write metadata seamlessly across different image
formats (.jpg, .png, .webp, .heic).

Because `PyExifTool` is a wrapper, you must install the system-level `exiftool` utility on
your host machine:

### Debian/Ubuntu
```bash
sudo apt-get update && sudo apt-get install -y libimage-exiftool-perl
```

### macOS (via Homebrew)
```bash
brew install exiftool
```

Then install the Python dependencies:

```bash
pip install -r requirements.txt              # importer runtime
pip install -r aisorter/requirements.txt     # training + model export (dev machine only)
```

## Usage

All commands run from the repository root.

### Import media

Interactive mode is the default. Run without location arguments and answer the prompts:

```bash
python main.py
```

Locations supplied on the command line skip their corresponding prompts:

```bash
python main.py <src> <dest> <category_model> \
  [--face-detector-model <onnx>] \
  [--references <faces_dir>] \
  [--face-identifier-model <onnx>] \
  [--event-threshold <count>]
```

- `src` — folder containing media to import
- `dest` — destination root; files land in `Photos from <year>` subfolders
- `category_model` — category classifier ONNX model
- `--face-detector-model` — optional local face detector model; blank uses the cache/download default
- `--references` — optional directory of per-person reference face folders
  (`references/Alice/*.jpg`, `references/Bob/*.jpg`) to enable name tagging
- `--face-identifier-model` — optional local face identifier model; prompted only when references are enabled
- `--event-threshold` — same-day files needed to create an event folder (default: `15`)
- `--config` — JSON file supplying any input not given on the command line
- `--no-interactive` — fail instead of prompting when a required location is missing
- `--verbose` — print all logs to the terminal instead of showing the progress bar
- `--log-file` — change the persistent log path (default: `plex_auto_import.log`)

#### Config file

`--config` reads inputs from a JSON file so repeat imports need no long command line:

```bash
python main.py --config import-config.json
```

```json
{
  "src": "/Volumes/CAMERA/DCIM",
  "dest": "/Volumes/media/Photos",
  "model": "aisorter/open_weights/category_95.onnx",
  "references": "references",
  "face_identifier_model": "~/.cache/aisorter/face_identifier.onnx",
  "event_threshold": 20,
  "interactive": false
}
```

Precedence is command line, then config file, then built-in defaults; anything still
missing is prompted for unless `--no-interactive` is set. Supported keys mirror the
argument names: `src`, `dest`, `model`, `face_detector_model`, `references`,
`face_identifier_model`, `family_group`, `family_dest`, `event`, `event_threshold`,
`interactive`, `verbose`, and `log_file`. Relative paths resolve against the config
file's own directory, `~` is expanded, and `null` means "not set". Unknown keys and
wrong value types are rejected with exit code `2`.

All runs write INFO-and-higher logs to the log file, including category classification,
face detection, face identification, and metadata-write results for each processed image.
Without `--verbose`, the terminal shows interactive prompts, the import progress bar, and a
final summary with image/video counts, elapsed time, entities processed per second, and images
processed per second. Entities count imported files, while images include still images and every
successfully analyzed frame extracted from videos.
After importing, each populated `Photos from <year>` folder is automatically grouped into
`Event on <date>` folders using the configured event threshold.
Years and event dates come from the file's capture date: EXIF/QuickTime metadata first, then
a `YYYYMMDD_*` filename prefix, then the file's modification time.
Files that fail AI processing are still moved (untagged). Non-media files are left in the
source folder, as are files whose destination already exists; the exit code is non-zero when
any file failed or was left behind because its destination already existed.

On Windows, `run.bat` starts interactive import mode; `events.bat` is a thin wrapper whose
placeholder paths must be edited before use.

### Group photos into events

```bash
python events.py <src> <threshold>
```

Bundles same-day photos into `Event on <date>` folders when at least `threshold`
photos share a capture date. Dates are derived the same way as the importer:
EXIF/QuickTime metadata, then a `YYYYMMDD_*` filename prefix, then modification time.

## AI sorter (`aisorter/`)

The `aisorter` package contains the inference pipeline and training tooling.
Run its entry points as modules from the repository root:

```bash
python -m aisorter.pipeline --input <dir> [--model <onnx>] [--references <dir>]  # tag a directory
python -m aisorter.classify                    # interactive sort-and-review tool
python -m aisorter.models.train --data <dir>   # train a category model
python -m aisorter.onnx_exporter --model <pth> # export a checkpoint to ONNX
```

Face detection/identification models are downloaded once to `~/.cache/aisorter/`.
Shared configuration (file extensions, thresholds, model URLs) lives in the
`constants/` package.

See `aisorter/photo_ai_sorter_strategy.md` for the pipeline design.
