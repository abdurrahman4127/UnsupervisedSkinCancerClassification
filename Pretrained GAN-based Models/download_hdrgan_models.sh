#!/bin/bash

URL="https://raw.githubusercontent.com/nonu116/HDR-GAN/master/pretrained/model.meta"
OUTPUT="model.meta"

wget --progress=bar:force "$URL" -O "$OUTPUT"

if [[ -f "$OUTPUT" ]]; then
    echo "Downloaded $OUTPUT successfully."
else
    echo "Failed to download the file."
fi