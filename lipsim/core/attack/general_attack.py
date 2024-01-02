import torch
from advertorch.attacks import L2PGDAttack, LinfPGDAttack, CarliniWagnerL2Attack, MomentumIterativeAttack, L1PGDAttack
from torch import nn

from autoattack import AutoAttack
from lipsim.core.attack.deepfool_attack import DeepFool
from lipsim.core.attack.square_attack import Square


class GeneralAttack:
    def __init__(self, config):
        self.config = config

    def generate_attack(self, img_ref, img_0, img_1, target=None, target_model=None, is_dist_attack=False):
        attack_method, attack_norm = self.config.attack.split('-')
        if attack_method == 'AA':  # AutoAttack
            img_adv = self.generate_auto_attack(attack_norm, img_0, img_1, img_ref, is_dist_attack, target,
                                                target_model)

        if attack_method == 'CW':  # CarliniWagner
            img_adv = self.generate_carlini_attack(img_ref, target, target_model)

        elif attack_method == 'PGD':
            img_adv = self.generate_pgd_attack(attack_norm, img_ref, target_model)

        elif attack_method == 'SQ':  # Square
            attack = Square(target_model, norm='L2', eps=self.config.eps, n_queries=5000, n_restarts=1,
                            p_init=.8, seed=0, verbose=False, loss='margin', resc_schedule=True)
            img_adv = attack.perturb(torch.stack((img_ref, img_0, img_1), dim=1), target.long())

        elif attack_method == 'DF':  # DeepFool
            attack = DeepFool(target_model, steps=50, overshoot=0.02)
            img_adv = attack(img_ref, target.long())

        elif attack_method == 'MI':  # MomentumIterativeAttack
            img_adv = self.generate_MIA_attack(img_ref, target, target_model, attack_norm)

        return img_adv

    def generate_MIA_attack(self, img_ref, target, target_model, attack_norm):
        if attack_norm == 'L2':
            attack = MomentumIterativeAttack(target_model, loss_fn=None, eps=self.config.eps, nb_iter=100,
                                             decay_factor=1.0, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
                                             targeted=False, ord=2)
        else:
            attack = MomentumIterativeAttack(target_model, loss_fn=None, eps=self.config.eps, nb_iter=100,
                                             decay_factor=1.0, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
                                             targeted=False, ord=float('inf'))
        img_adv = attack(img_ref, target.long())
        return img_adv

    def generate_pgd_attack(self, attack_norm, img_ref, target_model):
        if attack_norm == 'L2':
            adversary = L2PGDAttack(target_model, loss_fn=nn.CrossEntropyLoss(), eps=self.config.eps, nb_iter=1000,
                                    rand_init=True, targeted=False, eps_iter=0.01, clip_min=0.0, clip_max=1.0)

        elif attack_norm == 'Linf':
            adversary = LinfPGDAttack(target_model, loss_fn=nn.CrossEntropyLoss(), eps=self.config.eps, nb_iter=50,
                                      eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1., targeted=False)

        else:
            adversary = L1PGDAttack(target_model, loss_fn=None, eps=10.0, nb_iter=50, eps_iter=0.01, rand_init=True,
                                    clip_min=0.0, clip_max=1.0, targeted=False)
        img_ref = adversary(img_ref, target_model(img_ref))

        return img_ref

    def generate_carlini_attack(self, img_ref, target, target_model):
        adversary = CarliniWagnerL2Attack(target_model, 2, confidence=0, targeted=False, learning_rate=0.01,
                                          binary_search_steps=9, max_iterations=100, abort_early=True,
                                          initial_const=0.001, clip_min=0.0, clip_max=1.0, loss_fn=None)
        img_adv = adversary(img_ref, target.long())
        return img_adv

    def generate_auto_attack(self, attack_norm, img_0, img_1, img_ref, is_dist_attack, target, target_model):
        adversary = AutoAttack(target_model, norm=attack_norm, eps=self.config.eps, version='standard',
                               device='cuda')
        # adversary.attacks_to_run = ['apgd-ce']
        if is_dist_attack:
            img_ref = adversary.run_standard_evaluation(torch.stack((img_ref, img_ref), dim=1), target.long(),
                                                        bs=img_ref.shape[0])
        else:
            img_ref = adversary.run_standard_evaluation(torch.stack((img_ref, img_0, img_1), dim=1), target.long(),
                                                        bs=img_ref.shape[0])
        img_ref = img_ref[:, 0, :, :].squeeze(1)
        return img_ref
