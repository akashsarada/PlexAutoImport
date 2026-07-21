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

```bash
python main.py <src> <dest> <model> [--references <faces_dir>]
```

- `src` — folder containing media to import
- `dest` — destination root; files land in `Photos from <year>` subfolders
- `model` — path to the category classifier ONNX model
- `--references` — optional directory of per-person reference face folders
  (`references/Alice/*.jpg`, `references/Bob/*.jpg`) to enable name tagging

Files that fail AI processing are still moved (untagged); the exit code is
non-zero when any file failed.

On Windows, `run.bat` and `events.bat` are thin wrappers — edit the
placeholder paths inside before use.

### Group photos into events

```bash
python events.py <src> <threshold>
```

Bundles consecutive same-day photos (date-prefixed filenames such as
`20190614_123456.jpg`) into `Event on <date>` folders when at least
`threshold` photos share a date.

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
