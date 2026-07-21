@echo off
title Plex Automatic Event Sorter
REM Usage: replace the placeholders below.
REM   <src>       folder of date-prefixed photos to group into events
REM   <threshold> number of photos on one day required to form an event
python events.py "<src>" "<threshold>"
