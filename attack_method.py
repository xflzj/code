import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable as V
from advertorch.attacks import GradientSignAttack
from advertorch.utils import batch_multiply,clamp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern).cuda()
    return stack_kern, kern_size // 2

def project_noise(x, stack_kern, padding_size):
    # x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = F.conv2d(x, stack_kern, padding = (padding_size, padding_size), groups=3)
    return x

def input_diversity(img, image_width, image_resize):
    rnd = torch.randint(image_width, image_resize, ())
    rescaled = F.interpolate(img, size = [rnd, rnd], mode = 'bilinear', align_corners=True)
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    pad_list = (pad_left, pad_right, pad_top, pad_bottom)
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [image_resize, image_resize])
    # return padded if torch.rand(()) < prob else img
    return padded

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def clip_by_tensor(t, t_min, t_max):
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def transition_invariant_conv(size=15):
    kernel = gkern(size, 3).astype(np.float32)
    padding = size // 2
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)

    conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=size, stride=1, groups=3,
              padding=padding, bias=False)
    conv = conv.to(device)
    conv.weight.data = conv.weight.new_tensor(data=stack_kernel)

    return conv

class PGDAttack(object):
    def __init__(self, model, epsilon, num_steps, step_size, random_start=False,
                 image_width=224, image_resize=247, prob=0.0, momentum=0.0,
                 ti_size=1, loss_fn=None, targeted=False,pi_amplification=0.0,pi_size=3):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.random_start = random_start
        self.loss_fn = nn.CrossEntropyLoss()
        self.image_width = image_width
        self.image_resize = image_resize
        self.prob = prob
        self.momentum = momentum
        self.targeted = targeted
        self.ti_size = ti_size
        self.ti_conv = transition_invariant_conv(ti_size)
        self.pi_amplification=pi_amplification
        # self.alpha_beta=(epsilon/num_steps)*pi_amplification
        self.alpha_beta=step_size*pi_amplification
        self.stack_kern, self.padding_size = project_kern(pi_size)
        if pi_amplification!=10.0:
            self.gamma=self.alpha_beta*0.5
        else:
            self.gamma = self.alpha_beta
        kernel = gkern(15, 3).astype(np.float32)
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        self.c = torch.nn.Conv2d(3, 3, (15, 15), stride=1, padding=7, bias=False, groups=3)
        self.c.weight.data = torch.from_numpy(stack_kernel).cuda()

        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

    def perturb(self, X_nat, y_nat):
        """
        :param X_nat: a Float Tensor
        :param y: a Long Tensor
        :return:
        """
        images_min = clip_by_tensor(X_nat - self.epsilon , 0.0, 1.0)
        images_max = clip_by_tensor(X_nat + self.epsilon , 0.0, 1.0)
        out = self.model(X_nat)
        err = (out.data.max(1)[1] != y_nat.data).float().sum()
        X_pgd = Variable(X_nat.data, requires_grad=True)
        grad = X_pgd.new_zeros(X_nat.size(), requires_grad=True)
        amplification=0.0
        for k in range(self.num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            # print(k)L
            with torch.enable_grad():
                # print(self.prob)
                if np.random.uniform(low=0.0, high=1.0) <= self.prob:
                    X_div = input_diversity(X_pgd,
                                            image_width=self.image_width,
                                            image_resize=self.image_resize)
                else:
                    X_div = X_pgd

                loss = self.loss_fn(self.model(X_div), y_nat)
                if self.targeted:
                    loss = - loss
            loss.backward()

            cur_grad = X_pgd.grad.data
            if self.ti_size > 1:
                # print(self.ti_size)
                cur_grad = self.c(cur_grad)

            cur_grad = cur_grad / torch.mean(torch.abs(cur_grad), dim=[1, 2, 3], keepdim=True)
            grad = self.momentum * grad + cur_grad
            eta = self.step_size * grad.sign()

            if self.pi_amplification>1.0:
                amplification += self.alpha_beta * torch.sign(grad)
                cut_noise = torch.clamp(abs(amplification) - self.epsilon, 0, 10000.0) * torch.sign(amplification)
                projection = self.gamma * torch.sign(project_noise(cut_noise, self.stack_kern, self.padding_size))
                amplification += projection

                X_pgd = X_pgd + self.alpha_beta * torch.sign(grad) + projection
                X_pgd = clip_by_tensor(X_pgd, images_min, images_max)
                X_pgd = V(X_pgd, requires_grad=True)

            else:
                X_pgd.data = X_pgd.data + eta
                eta = torch.clamp(X_pgd.data - X_nat.data, -self.epsilon, self.epsilon)
                X_pgd.data = X_nat.data + eta
                X_pgd.data = torch.clamp(X_pgd, 0.0, 1.0)
        return X_pgd





class FGSMAttack(GradientSignAttack):
    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0.,
                 clip_max=1., targeted=False,ti_size=1,image_width=224, image_resize=247, prob=0.0):
        """
        Create an instance of the GradientSignAttack.
        """
        super(GradientSignAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.ti_size = ti_size
        self.ti_conv = transition_invariant_conv(ti_size)
        self.eps = eps
        self.targeted = targeted
        self.prob=prob
        self.image_width = image_width
        self.image_resize = image_resize
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """


        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        with torch.enable_grad():
            if np.random.uniform(low=0.0, high=1.0) <= self.prob:
                X_div = input_diversity(xadv,
                                        image_width=self.image_width,
                                        image_resize=self.image_resize)
            else:
                X_div = xadv

        outputs = self.predict(X_div)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()

        if self.ti_size>1:
            tim=self.ti_conv(xadv.grad.data.cuda())
        else:
            tim=xadv.grad.data.cuda()

        grad_sign = tim.sign()

        xadv = xadv + batch_multiply(self.eps, grad_sign)

        xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv.detach()

