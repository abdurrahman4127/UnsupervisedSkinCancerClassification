#!/bin/bash

# URLs for model and config URLs
declare -A MODEL_URLS=(
    ["biggan-deep-128"]="https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-pytorch_model.bin"
    ["biggan-deep-256"]="https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin"
    ["biggan-deep-512"]="https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-pytorch_model.bin"
)

declare -A CONFIG_URLS=(
    ["biggan-deep-128"]="https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-config.json"
    ["biggan-deep-256"]="https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-config.json"
    ["biggan-deep-512"]="https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-config.json"
)

mkdir -p bigGAN_models bigGAN_configs

download_file() {
    local url=$1
    local output=$2
    echo "Downloading $output..."
    wget --progress=bar:force:noscroll "$url" -O "$output"
    echo "Download complete: $output"
}

# download models
for model in "${!MODEL_URLS[@]}"; do
    download_file "${MODEL_URLS[$model]}" "models/$model-pytorch_model.bin"
done

# download configs
for config in "${!CONFIG_URLS[@]}"; do
    download_file "${CONFIG_URLS[$config]}" "configs/$config-config.json"
done

echo "All downloads complete."