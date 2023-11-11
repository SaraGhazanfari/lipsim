import torch
import torch.nn as nn

from lipsim.core.attack.square_attack import Attack


class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]
    Distance Measure : L2
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, steps=50, overshoot=0.02, device='cuda'):
        super().__init__("DeepFool", model)
        self.steps = steps
        self.overshoot = overshoot
        self.supported_mode = ["default"]
        self.device = device

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        adv_images, target_labels = self.forward_return_target_labels(images, labels)
        return adv_images

    def forward_return_target_labels(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        batch_size = len(images)
        correct = torch.tensor([True] * batch_size)
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx: idx + 1].clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                if not correct[idx]:
                    continue
                early_stop, pre, adv_image = self._forward_indiv(
                    adv_images[idx], labels[idx]
                )
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()
        return adv_images, target_labels

    def _forward_indiv(self, image, label):
        image.requires_grad = True
        fs = self.get_logits(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image)

        ws = self._construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (
                torch.abs(f_prime[hat_L])
                * w_prime[hat_L]
                / (torch.norm(w_prime[hat_L], p=2) ** 2)
        )

        target_label = hat_L if hat_L < label else hat_L + 1

        adv_image = image + (1 + self.overshoot) * delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx + 1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
# import numpy as np
# from torch.autograd import Variable
# import torch as torch
# import copy
#
# from autoattack.other_utils import zero_gradients
#
#
# def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):
#     """
#        :param image: Image of size HxWx3
#        :param net: network (input: images, output: values of activation **BEFORE** softmax).
#        :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
#        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
#        :param max_iter: maximum number of iterations for deepfool (default = 50)
#        :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
#     """
#     is_cuda = torch.cuda.is_available()
#     #
#     # if is_cuda:
#     #     print("Using GPU")
#     #     image = image.cuda()
#     #     net = net.cuda()
#     # else:
#     #     print("Using CPU")
#
#     f_image = net(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
#     # I = (np.array(f_image)).flatten().argsort()[::-1]
#     #
#     # I = I[0:num_classes]
#     label = I[0]
#
#     input_shape = image.cpu().numpy().shape
#     pert_image = copy.deepcopy(image)
#     w = np.zeros(input_shape)
#     r_tot = np.zeros(input_shape)
#
#     loop_i = 0
#
#     x = Variable(pert_image[None, :], requires_grad=True)
#     fs = net(x)
#     # fs_list = [fs[0, I[k]] for k in range(num_classes)]
#     k_i = label
#
#     while k_i == label and loop_i < max_iter:
#
#         pert = np.inf
#         fs[0, I[0]].backward(retain_graph=True)
#         grad_orig = x.grad.data.cpu().numpy().copy()
#
#         for k in range(1, num_classes):
#             zero_gradients(x)
#
#             fs[0, I[k]].backward(retain_graph=True)
#             cur_grad = x.grad.data.cpu().numpy().copy()
#
#             # set new w_k and new f_k
#             w_k = cur_grad - grad_orig
#             f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
#
#             pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
#
#             # determine which w_k to use
#             if pert_k < pert:
#                 pert = pert_k
#                 w = w_k
#
#         # compute r_i and r_tot
#         # Added 1e-4 for numerical stability
#         r_i = (pert + 1e-4) * w / np.linalg.norm(w)
#         r_tot = np.float32(r_tot + r_i)
#
#         if is_cuda:
#             pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
#         else:
#             pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)
#
#         x = Variable(pert_image, requires_grad=True)
#         fs = net(x)
#         k_i = np.argmax(fs.data.cpu().numpy().flatten())
#
#         loop_i += 1
#
#     r_tot = (1 + overshoot) * r_tot
#
#     return r_tot, loop_i, label, k_i, pert_image
