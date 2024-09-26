#!/bin/bash

# to download the desired model, pass its name as an argument; like this:
# ./ download_srgan_model.sh SRGAN_x2-SRGAN_ImageNet

FILE=$1

FILE_PATH=./srgan_results/pretrained_models
mkdir -p "$FILE_PATH"

if [ "$FILE" == "SRGAN_x2-SRGAN_ImageNet" ]; then
  URL="https://huggingface.co/goodfellowliu/SRGAN-PyTorch/resolve/main/SRGAN_x2-SRGAN_ImageNet.pth.tar"
  wget "$URL" -P "$FILE_PATH"
elif [ "$FILE" == "SRGAN_x4-SRGAN_ImageNet" ]; then
  URL="https://huggingface.co/goodfellowliu/SRGAN-PyTorch/resolve/main/SRGAN_x4-SRGAN_ImageNet.pth.tar"
  wget "$URL" -P "$FILE_PATH"
elif [ "$FILE" == "SRGAN_x8-SRGAN_ImageNet" ]; then
  URL="https://huggingface.co/goodfellowliu/SRGAN-PyTorch/resolve/main/SRGAN_x8-SRGAN_ImageNet.pth.tar"
  wget "$URL" -P "$FILE_PATH"
elif [ "$FILE" == "SRResNet_x2-SRGAN_ImageNet" ]; then
  URL="https://huggingface.co/goodfellowliu/SRGAN-PyTorch/resolve/main/SRResNet_x2-SRGAN_ImageNet.pth.tar"
  wget "$URL" -P "$FILE_PATH"
elif [ "$FILE" == "SRResNet_x4-SRGAN_ImageNet" ]; then
  URL="https://huggingface.co/goodfellowliu/SRGAN-PyTorch/resolve/main/SRResNet_x4-SRGAN_ImageNet.pth.tar"
  wget "$URL" -P "$FILE_PATH"
elif [ "$FILE" == "SRResNet_x8-SRGAN_ImageNet" ]; then
  URL="https://huggingface.co/goodfellowliu/SRGAN-PyTorch/resolve/main/SRResNet_x8-SRGAN_ImageNet.pth.tar"
  wget "$URL" -P "$FILE_PATH"
elif [ "$FILE" == "DiscriminatorForVGG_x2-SRGAN_ImageNet" ]; then
  URL="https://huggingface.co/goodfellowliu/SRGAN-PyTorch/resolve/main/DiscriminatorForVGG_x2-SRGAN_ImageNet.pth.tar"
  wget "$URL" -P "$FILE_PATH"
elif [ "$FILE" == "DiscriminatorForVGG_x4-SRGAN_ImageNet" ]; then
  URL="https://huggingface.co/goodfellowliu/SRGAN-PyTorch/resolve/main/DiscriminatorForVGG_x4-SRGAN_ImageNet.pth.tar"
  wget "$URL" -P "$FILE_PATH"
elif [ "$FILE" == "DiscriminatorForVGG_x8-SRGAN_ImageNet" ]; then
  URL="https://huggingface.co/goodfellowliu/SRGAN-PyTorch/resolve/main/DiscriminatorForVGG_x8-SRGAN_ImageNet.pth.tar"
  wget "$URL" -P "$FILE_PATH"
else
  echo "Available arguments are: SRGAN_x2-SRGAN_ImageNet, SRGAN_x4-SRGAN_ImageNet, SRGAN_x8-SRGAN_ImageNet, SRResNet_x2-SRGAN_ImageNet, SRResNet_x4-SRGAN_ImageNet, SRResNet_x8-SRGAN_ImageNet, DiscriminatorForVGG_x2-SRGAN_ImageNet, DiscriminatorForVGG_x4-SRGAN_ImageNet, DiscriminatorForVGG_x8-SRGAN_ImageNet"
  exit 1
fi