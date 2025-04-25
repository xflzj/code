import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
from torchvision import transforms
import time

import pretrainedmodels
from utils_sgm import register_hook_for_resnet, register_hook_for_densenet
from utils_data import SubsetImageNet, save_images
from utils import load_pretrained_model
from attack_method import FGSMAttack,PGDAttack

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Attack Evaluation')#

parser.add_argument('--input_dir', default='', help='the path of original dataset')

parser.add_argument('--output_dir', default='', help='the path of the saved dataset')
parser.add_argument('--attack_method', default='fgsm', type=str,choices=['fgsm','pgd'])
parser.add_argument('--batch_size', type=int, default=10, metavar='N',help='input batch size for adversarial attack')

parser.add_argument('--ensemble', default=0, type=int)
parser.add_argument("--DI", type=int, default=0)
parser.add_argument("--TI", type=int, default=0)
parser.add_argument('--gamma', default=2, type=float)
parser.add_argument("--amplification", type=float, default=0.0, help="To amplifythe step size.")
parser.add_argument("--pi_size", type=int, default=3, help="k size")

parser.add_argument('--ensemble_flag', default='0', type=str)
parser.add_argument('--arch', default='resnet152',help='source model for black-box attack evaluation',choices=model_names)
parser.add_argument('--snet_dir', default='', help='the path of snet')
####################
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--epsilon', default=18, type=float, help='perturbation')
parser.add_argument('--num_steps', default=10, type=int, help='perturb number of steps')
parser.add_argument('--step_size', default=2, type=float, help='perturb step size') 
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--print_freq', default=10, type=int)

parser.add_argument("--prob", type=float, default=0.0, help="probability of using diverse inputs.")
parser.add_argument("--image_width", type=int, default=224, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=224, help="Height of each input images.")
parser.add_argument("--image_resize", type=int, default=247, help="Resize of each input images.")

args = parser.parse_args()
# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

def generate_adversarial_example(model, data_loader, adversary, img_path):
    """
    evaluate model by black-box attack
    """
    model.eval()

    for batch_idx, (inputs, true_class, idx) in enumerate(data_loader):

        inputs, true_class = \
            inputs.to(device), true_class.to(device)

        inputs_adv = adversary.perturb(inputs, true_class)
        save_images(inputs_adv.detach().cpu().numpy(), img_list=img_path,
                    idx=idx, output_dir=args.output_dir)
        # assert False
        if batch_idx % args.print_freq == 0:
            print('generating: [{0}/{1}]'.format(batch_idx, len(data_loader)))

import torch.nn.functional as F

class   Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        self.normalize = len(models)

    def forward(self, x):
        output = 0
        for i in range(len(self.models)):
            output += F.softmax(self.models[i](x), dim=1)
        output /= self.normalize
        return torch.log(output)

def main():
    print(args.output_dir)
    begin=time.time()
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    data_set = SubsetImageNet(root=args.input_dir, transform=transform_test)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    # create models
    net = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')  # densenet 121
    model = nn.Sequential(Normalize(mean=net.mean, std=net.std), net)
    model = model.to(device)
    checkpoint = torch.load(args.snet_dir)
    
    #print(checkpoint.keys())
    load_pretrained_model(model, checkpoint['snet'])
    model.eval()



    epsilon = args.epsilon / 255.0 # 16
    if args.step_size < 0:
        step_size = epsilon / args.num_steps  # 16/10
        print("<0")
    else:
        step_size = args.step_size / 255.0  # 2

    # using our method - Skip Gradient Method (SGM)
    if args.gamma < 1.0:
        print('using SGM Method')
        if args.arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            register_hook_for_resnet(model, arch=args.arch, gamma=args.gamma)
            if args.ensemble==True:
                register_hook_for_resnet(model2, arch=args.arch, gamma=args.gamma)
        elif args.arch in ['densenet121', 'densenet169', 'densenet201']:
            register_hook_for_densenet(model, arch=args.arch, gamma=args.gamma)
            if args.ensemble == True:
                register_hook_for_densenet(model2, arch=args.arch, gamma=args.gamma)
        else:
            raise ValueError('Current code only supports resnet/densenet. '
                             'You can extend this code to other architectures.')

    if args.momentum > 0.0:
        print('-- momentum --')

    if args.DI == True:
        print("input diversity load")
        args.prob = 0.7
    if args.ensemble == True:
        print('-- ensemble loss --')
        loss = nn.NLLLoss(reduction='mean').to(device)
        if args.TI == True:
            if args.attack_method == 'fgsm':
                print('using FGSM_TIM attack  -- 1')
                adversary = FGSMAttack(predict=ensemble, loss_fn=loss,
                                              eps=epsilon, clip_min=0.0, clip_max=1.0, targeted=False,ti_size=15,image_width=224, image_resize=247, prob=args.prob)
            if args.attack_method == 'pgd':
                print('using PGD_TIM attack   --2')
                adversary = PGDAttack(model=ensemble, epsilon=epsilon,num_steps=args.num_steps,step_size=step_size,
                                      random_start=False,image_width=224, image_resize=247, prob=args.prob, momentum=args.momentum,
                                      ti_size=15, loss_fn=loss, targeted=False,pi_amplification=args.amplification,pi_size=args.pi_size)
        else:
            if args.attack_method=='fgsm':
                print('using linf FGSM attack --3 ')  #
                print('fgsm ensemble')
                adversary = FGSMAttack(predict=ensemble, loss_fn=loss,
                                     eps=epsilon, clip_min=0.0, clip_max=1.0, targeted=False, ti_size=1,image_width=224, image_resize=247, prob=args.prob)
            if args.attack_method=='pgd':
                print('using linf PGD attack  --4')
                adversary = PGDAttack(model=ensemble, epsilon=epsilon, num_steps=args.num_steps, step_size=step_size,
                                      random_start=False, image_width=224, image_resize=247, prob=args.prob, momentum=args.momentum,
                                      ti_size=1, loss_fn=loss, targeted=False,pi_amplification=args.amplification,pi_size=args.pi_size)

    else:
        print('no  ensemble')
        loss = nn.CrossEntropyLoss(reduction="sum")
        if args.TI==True:

            if args.attack_method == 'fgsm':
                print('using FGSM_TIM attack  --5')
                adversary = FGSMAttack(predict=model, loss_fn=loss,
                                              eps=epsilon, clip_min=0.0, clip_max=1.0, targeted=False,ti_size=15, image_width=224, image_resize=247,prob=args.prob)
            if args.attack_method == 'pgd':
                print('using PGD_TIM attack  --6')
                adversary = PGDAttack(model=model, epsilon=epsilon,num_steps=args.num_steps,step_size=step_size,
                                      random_start=False,image_width=224, image_resize=247, prob=args.prob, momentum=args.momentum,
                                      ti_size=15, loss_fn=loss, targeted=False,pi_amplification=args.amplification,pi_size=args.pi_size)

        else:
            if args.attack_method == 'fgsm':
                print('using linf FGSM attack --7')
                adversary = FGSMAttack(predict=model, loss_fn=loss,
                                     eps=epsilon, clip_min=0.0, clip_max=1.0, targeted=False, ti_size=1, image_width=224, image_resize=247,prob=args.prob)
            if args.attack_method == 'pgd':
                print('using linf PGD attack --8')
                adversary = PGDAttack(model=model, epsilon=epsilon, num_steps=args.num_steps, step_size=step_size,
                                      random_start=False, image_width=224, image_resize=247, prob=args.prob,momentum=args.momentum,
                                      ti_size=1, loss_fn=loss, targeted=False,pi_amplification=args.amplification,pi_size=args.pi_size)

    generate_adversarial_example(model=model, data_loader=data_loader,
                                 adversary=adversary, img_path=data_set.img_path)

    end=time.time()
    print("time:",end-begin)
if __name__ == '__main__':
   
    main()
