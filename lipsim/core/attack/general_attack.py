import torch
from advertorch.attacks import L2PGDAttack, LinfPGDAttack, CarliniWagnerL2Attack
from torch import nn

from autoattack import AutoAttack


class GeneralAttack:
    def __init__(self, config):
        self.config = config

    def generate_attack(self, img_ref, img_0, img_1, target=None, target_model=None, is_dist_attack=False):
        attack_method, attack_norm = self.config.attack.split('-')

        if attack_method == 'AA':
            img_ref = self.generate_auto_attack(attack_norm, img_0, img_1, img_ref, is_dist_attack, target,
                                                target_model)

        if attack_method == 'CW':
            img_ref = self.generate_carlini_attack(img_ref, target, target_model)

        elif attack_method == 'PGD':
            img_ref = self.generate_pgd_attack(attack_norm, img_ref, target_model)

        return img_ref

    def generate_pgd_attack(self, attack_norm, img_ref, target_model):
        if attack_norm == 'L2':
            adversary = L2PGDAttack(target_model, loss_fn=nn.MSELoss(), eps=self.config.eps, nb_iter=1000,
                                    rand_init=True, targeted=False, eps_iter=0.01, clip_min=0.0, clip_max=1.0)

        else:  # attack_type == 'Linf':
            adversary = LinfPGDAttack(target_model, loss_fn=nn.MSELoss(), eps=self.config.eps, nb_iter=50,
                                      eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1., targeted=False)
        img_ref = adversary(img_ref, target_model(img_ref))
        return img_ref

    def generate_carlini_attack(self, img_ref, target, target_model):
        adversary = CarliniWagnerL2Attack(target_model, 2, confidence=0, targeted=False, learning_rate=0.01,
                                          binary_search_steps=9, max_iterations=10000, abort_early=True,
                                          initial_const=0.001, clip_min=0.0, clip_max=1.0, loss_fn=None)
        img_ref = adversary(img_ref, target.long())
        return img_ref

    def generate_auto_attack(self, attack_norm, img_0, img_1, img_ref, is_dist_attack, target, target_model):
        adversary = AutoAttack(target_model, norm=attack_norm, eps=self.config.eps, version='standard',
                               device='cuda')
        adversary.attacks_to_run = ['apgd-ce']
        if is_dist_attack:
            img_ref = adversary.run_standard_evaluation(torch.stack((img_ref, img_ref), dim=1), target.long(),
                                                        bs=img_ref.shape[0])
        else:
            img_ref = adversary.run_standard_evaluation(torch.stack((img_ref, img_0, img_1), dim=1), target.long(),
                                                        bs=img_ref.shape[0])
        img_ref = img_ref[:, 0, :, :].squeeze(1)
        return img_ref
