@echo on
title Plex Automatic Event Sorter
REM The Python script requires 2 arguments to run, the source path and the threshold
REM Replace src with the folder to be sorted
REM Replace threshold with the int of the number of photos required to be considered an event
python events.py ["C:\Data\Applications\GitHub\PlexAutoImport\test"] ["15"]
