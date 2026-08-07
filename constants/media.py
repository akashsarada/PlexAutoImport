"""File-type and video-sampling constants."""

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".dng")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v")
DEFAULT_EVENT_THRESHOLD = 15
DEFAULT_FAMILY_GROUP = "Family"

# Videos shorter than this sample 3 spread frames; longer ones sample every n/3 seconds.
DURATION_THRESHOLD_SECONDS = 9
