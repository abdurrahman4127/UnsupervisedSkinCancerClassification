#!/bin/bash

URL="https://raw.githubusercontent.com/RaphaelMeudec/deblur-gan/master/generator.h5"
OUTPUT="generator.h5"

wget --progress=bar:force "$URL" -O "$OUTPUT"

if [[ -f "$OUTPUT" ]]; then
    echo "Downloaded $OUTPUT successfully."
else
    echo "Failed to download the file."
fi