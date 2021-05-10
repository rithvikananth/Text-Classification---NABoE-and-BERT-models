import torch
from torch.optim.optimizer import Optimizer
import math


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                gradient = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p.data)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p.data)

                e_average, e_average_sq = state['exp_avg'], state['exp_avg_sq']
                beta_1, beta_2 = group['betas']

                state['step'] = state['step'] + 1

                e_average_sq.mul_(beta_2).addcmul_(1 - beta_2, gradient, gradient)
                e_average.mul_(beta_1).add_(1 - beta_1, gradient)

                dnm = e_average_sq.sqrt().add_(group['eps'])
                add_1 = beta_1 ** state['step']
                bias_correct_1 = 1 - add_1
                add_2 = beta_2 ** state['step']
                bias_correct_2 = 1 - add_2
                if group['warmup'] > state['step']:
                    schedule_learning_rate = state['step'] * group['lr'] / group['warmup']
                    schedule_learning_rate = 1e-8 + schedule_learning_rate
                else:
                    schedule_learning_rate = group['lr']

                step_s = schedule_learning_rate * math.sqrt(bias_correct_2) / bias_correct_1

                if group['weight_decay'] != 0:
                    input_1 = -group['weight_decay'] * schedule_learning_rate
                    p.data.add_(input_1, p.data)

                p.data.addcdiv_(-step_s, e_average, dnm)

        return loss

