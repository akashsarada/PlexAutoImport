# PlexAutoImport
An automatic way to move all your memories off your camera and onto your storage location

## Requirements & Setup

This project uses `PyExifTool` to read and write metadata seamlessly across different image formats (.jpg, .png, .webp, .heic).

Because `PyExifTool` is a wrapper, you must install the system-level `exiftool` utility on your host machine:

### Debian/Ubuntu
```bash
sudo apt-get update && sudo apt-get install -y libimage-exiftool-perl
```

### macOS (via Homebrew)
```bash
brew install exiftool
```
