# Diverse Gradient Method
 
This repository contains the code for "Improving the Transferability of Adversarial Examples with Diverse Gradients".

## Method

We propose a Diverse Gradient Method (DGM) to craft transferable adversarial examples.
## Environment

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.6.2
- torch = 1.5.0
- torchvision = 0.6.0
- advertorch = 0.2.3
- pretrainedmodels = 0.7.4

Additionally, we reproduced DI, TI, MI and PI in Pytorch in *attack_method.py*.
## Run the code

1. Download the dataset from [Dataset](https://drive.google.com/file/d/1PqpPTCIvzRRfbOhiGgQP72MXo43cw8nr/view?usp=sharing) and extract images to the path `./SubImageNetVal/`  


2. Training derived model by knowledge distillation, diverse gradient information, For ResNet-152 as the source model,
    ```bash
    python train_kd.py --save_root './result' --img_root './SubImageNetVal/' --T 20 --note 'T20_resnet152' --arch 'resnet152' 
    ```
   Or you can directly download our trained weight files form [derived model weight files](https://drive.google.com/file/d/1hHYUwQw9POYczCRLXURyNtcXT4qrac2u/view?usp=sharing)  

    
3. Generate adversarial examples and save them into path `./adv_path/`. For ResNet-152 as the source model,
    ```bash
    python attack_distillation.py --input_dir './SubImageNetVal/' --output_dir './adv_path/' --attack_method 'pgd' --ensemble 1 --snet_dir './result/T20_resnet152/checkpoint.pth.tar'
    ```
   --snet_dir is the weight file path, you can directily download or training by step 2.  


4. Evaluate the transferability of generated adversarial examples in `./adv_path/`. 
    ```bash
    AdvPath="/data/xfl2/adv_sp_res50_pgd/" bash evaluate.sh
    AdvPath="./adv_30/" bash evaluate.sh
    ```
python attack_distillation at.py --input_dir './dataset/SubImageNetVal/' --output_dir './resultruanying/' --attack_method 'pgd' --snet_dir './atfgsm/'checkpoint.pth.tar'

#### Pretrained models

All pretrained models in our paper can be found online:

- For undenfended models, we use pretrained models in [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch);

- For denfended models, they are trained by [Ensemble Adversarial Training](https://arxiv.org/abs/1705.07204) [1],   
  and pretrained results can be found in [Tensorflow version](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models).
  or in [Pytroch version](https://github.com/ylhz/tf_to_pytorch_model)

## Reference

[1] Florian Tramèr, Alexey Kurakin, Nicolas Papernot, Ian Goodfellow, Dan Boneh, Patrick McDaniel. Ensemble Adversarial Training: Attacks and Defenses. In ICLR, 2018.
