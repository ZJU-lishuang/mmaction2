#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/ava_custom/video/ ../../data/ava_custom/rawframes/ --task rgb --level 1 --mixed-ext
echo "Genearte raw frames (RGB only)"

cd ava/
