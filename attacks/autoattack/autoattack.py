import os
import json
import time
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adv_utils import Logger

# Attacks
from .autopgd import APGDAttack
from .fab import FABAttack
from .square import SquareAttack
import numpy as np

def calc_dice(pred, y):
    return np.sum([y * pred]) * 2.0 / (np.sum(pred) + np.sum(y))


class AutoAttack():
    def __init__(self, model, dice_thresh, n_target_classes,
                 eps=.3, seed=None, verbose=True, attacks_to_run=[],
                 device='cuda', n_iter=20,
                 visualizations=False):
        self.model = model
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.device = device
        self.n_iter = n_iter
        self.visualizations = visualizations
        # Dice loss
        self.dice_thresh = dice_thresh
        # Dataloader
        self.classes = n_target_classes

        self.apgd = APGDAttack(
            self.model, dice_thresh, n_restarts=5, n_iter=n_iter,
            verbose=False, eps=self.epsilon, eot_iter=1,
            rho=.75, seed=self.seed, device=self.device)

        self.fab = FABAttack(
            self.model, dice_thresh, n_target_classes=n_target_classes,
            n_restarts=5, n_iter=n_iter, eps=self.epsilon, seed=self.seed,
            verbose=False, device=self.device)

        self.square = SquareAttack(
            self.model, dice_thresh, p_init=0.8, n_queries=self.n_iter,
            eps=self.epsilon, n_restarts=1, seed=self.seed,
            verbose=False, device=self.device, resc_schedule=False)

        
    def get_seed(self):
        return time.time() if self.seed is None else self.seed



    @torch.no_grad()
    def run_standard_evaluation(self, x, y):
        # print('Including {}'.format(', '.join(self.attacks_to_run)))

        output = self.model(x.to(self.device))
        output = torch.argmax(F.softmax(output, 1), dim=1)
        dice = calc_dice(output.cpu().numpy(), y.cpu().numpy())

        if dice < self.dice_thresh:
            return  x


        for attack in self.attacks_to_run:
            # print(attack)
            if attack == 'apgd-ce':
                self.apgd.loss = 'ce'
                self.apgd.dice_thresh = self.dice_thresh
                self.apgd.seed = self.get_seed()
                try:
                    _, adv_curr = self.apgd.perturb(x, y)
                except RuntimeError as e:
                    print(e)
                    adv_curr = x

            elif attack == 'apgd-dlr':
                self.apgd.loss = 'dlr'
                self.apgd.dice_thresh = self.dice_thresh
                self.apgd.seed = self.get_seed()
                _, adv_curr = self.apgd.perturb(x, y)

            elif attack == 'fab':
                self.fab.n_iter = 5
                self.fab.seed = self.get_seed()
                self.fab.dice_thresh = self.dice_thresh
                adv_curr = self.fab.perturb(x, y)

            elif attack == 'square':
                self.square.seed = self.get_seed()
                self.square.dice_thresh = self.dice_thresh
                adv_curr = self.square.perturb(x, y)

            elif attack == 'pgd':
                epsilon = torch.tensor(
                    self.epsilon, device=self.device).unsqueeze(0).expand(
                    x.shape[1]).view(x.shape[1], 1, 1, 1)
                delta = attack_pgd(
                    self.model, x, y, epsilon, self.device, epsilon / 3.,
                    iters=self.n_iter)
                adv_curr = x + delta

            else:
                raise ValueError('Attack not supported')

            # print("print inside autoattack: ", x.shape, adv_curr.shape)
            if len(adv_curr.shape) > len(x.shape):
                adv_curr = adv_curr.squeeze(0)
            output = self.model(adv_curr)
            output = torch.argmax(F.softmax(output, 1), dim=1)
            dice_score = calc_dice(output.cpu().numpy(), y.cpu().numpy())

            if self.verbose:
                print("attack {} reduced dice: {} => {}".format(attack, dice, dice_score))
            dice = dice_score

            if dice_score < self.dice_thresh:
                return adv_curr
            

            x = adv_curr
            
        return x

   
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def attack_pgd(model, X, y, eps, rank, alpha=1./255., iters=5, restarts=5):
    # eps: magnitude of the attack
    # alpha: step size
    def clamp(X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    max_loss = torch.zeros(y.shape[0], 1, 1, 1).to(rank)
    max_delta = torch.zeros_like(X).to(rank)
    for _ in range(restarts):
        with torch.enable_grad():
            delta = torch.zeros_like(X).to(rank)
            # Delta for each channel
            for i in range(X.shape[1]):
                # print(delta.shape)
                delta[:, i, ...].uniform_(-eps[i][0][0][0].item(),
                                              eps[i][0][0][0].item())
            delta.requires_grad = True
            for _ in range(iters):
                output = model(X + delta)
                loss = F.cross_entropy(output, y.long())
                loss.backward()
                grad = delta.grad.detach()
                d = clamp(delta + alpha * torch.sign(grad), -eps, eps)
                d = clamp(d, 0 - X, 1 - X)
                delta.data = d
                delta.grad.zero_()

        # with torch.no_grad():
        final = model(X + delta)
        all_loss = F.cross_entropy(final, y.long(), reduction='none')
        idx = (all_loss >= max_loss).unsqueeze(1).repeat(
            1, X.shape[1], 1, 1, 1)
        # print(idx.shape, delta.shape)
        idx = idx.squeeze(0)
        max_delta[idx] = delta.detach()[idx]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta
