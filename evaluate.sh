#!/bin/bash
Model=(resnet152 resnet101 resnet50 densenet121 densenet201 vgg16 senet154 vgg19_bn inceptionv3 inceptionv4 inceptionresnetv2)
#Model=(resnet152 resnet101 resnet50 densenet121 densenet201 vgg16 senet154 vgg19_bn inceptionv3 inceptionv4 inceptionresnetv2)
#AdvPath=()

clear
for path in "${AdvPath[@]}"
do
  for model in "${Model[@]}"
  do
    CUDA_VISIBLE_DEVICES=0 python -u evaluate.py --input_dir=$path --arch=$model
  done
done