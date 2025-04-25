


## Environment

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.7.6
- torch = 1.7.1
- torchvision = 0.8.2
- advertorch = 0.2.3
- pretrainedmodels = 0.7.4

Additionally, we reproduced DI, TI, MI and PI in Pytorch in *attack_method.py*.

## Run the code

1. Download the dataset from [Dataset](https://drive.google.com/file/d/1PqpPTCIvzRRfbOhiGgQP72MXo43cw8nr/view?usp=sharing)

2. Train the model to generate a perturbed dataset, For ResNet-152 as the source model,
    ```bash
    python train_kd.py --save_root './datamodel' --img_s './SubImageNetVal/' --T 20 --note 'T20_resnet152' --arch 'resnet152' 
    ```
   

    
3. Generate perturbation dataset and save them into path `./epsdata/`. For ResNet-152 as the source model,
    ```bash
    python get_epsdata.py --input_dir './SubImageNetVal/' --output_dir './epsdata/' --attack_method 'fgsm' --epsilon=18 --ensemble 0 --snet_dir './datamodel/T20_resnet152/checkpoint.pth.tar'
    ```
   --snet_dir is the weight file path, you can directily download or training by step 2.  

4. Train the final model, For ResNet-152 as the source model,
    ```bash
    python train_kd.py --save_root './result' --img_s './epsdata/' --T 20 --note 'T20_resnet152' --arch 'resnet152' 
    ```

5. Generate adversarial samples
    ```bash
    python get_adv.py --input_dir './SubImageNetVal/' --output_dir './adv/' --attack_method 'pgd' --epsilon=16 --ensemble 1 --snet_dir1 './result/T20_T20_resnet152/checkpoint.pth.tar'
    ```
    Set different T values in the second step for the third and fourth steps to obtain an additional seven models

4. Evaluate the transferability of generated adversarial examples in `./adv/`. 
    ```bash
    AdvPath="./adv/" bash evaluate.sh
    
    ```


#### Pretrained models

All pretrained models in our paper can be found online:

- For undenfended models, we use pretrained models in [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch);

- For denfended models, they are trained by [Ensemble Adversarial Training](https://arxiv.org/abs/1705.07204) [1],   
  and pretrained results can be found in [Tensorflow version](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models).
  or in [Pytroch version](https://github.com/ylhz/tf_to_pytorch_model)

## Reference

[1] Florian Tramèr, Alexey Kurakin, Nicolas Papernot, Ian Goodfellow, Dan Boneh, Patrick McDaniel. Ensemble Adversarial Training: Attacks and Defenses. In ICLR, 2018.
