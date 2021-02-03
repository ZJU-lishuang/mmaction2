#!/usr/bin/env bash

cd ../../../

PYTHONPATH=. python tools/data/build_file_list.py jhmdb data/jhmdb/Frames/ --level 2 --format rawframes --rgb-prefix 0 --shuffle
echo "Filelist for rawframes generated."

cd tools/data/hmdb51/
