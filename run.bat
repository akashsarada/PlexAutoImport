@echo off
title Plex Automatic Importer
REM Usage: replace the placeholders below.
REM   <src>   source folder containing media to import
REM   <dest>  destination root (files land in "Photos from <year>" subfolders)
REM   <model> path to the category classifier ONNX model
python main.py "<src>" "<dest>" "<model>"
