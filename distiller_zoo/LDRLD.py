import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import numpy as np

def ADW(index1, index2, gamma):
    t = index1 + 1
    t_prime = index2 + 1
    a = 2.0
    weight = 1.0 / (1.50 + abs(t - t_prime)) * a * np.exp(-gamma * (t + t_prime))
    return weight
def calculate_recursion_combinations(teacher_logits, student_logits, T, indices,all_index_combinations):

    if not indices:
        return 0

    current_indices = indices[0]
    remaining_indices = indices[1:]
    current_index_combinations=all_index_combinations[0]
    remaining_index_combinations =all_index_combinations[1:]

    gamma = 0.05
    if len(current_index_combinations)==2:
        weight = ADW(current_index_combinations[0], current_index_combinations[1], gamma)
    else:
        weight = 1.0

    mask = torch.ones_like(teacher_logits).scatter_(1, current_indices, 0).bool()


    selected_teacher_logits = teacher_logits / T - 1000.0 * mask
    selected_student_logits = student_logits / T - 1000.0 * mask

    teacher_probs = F.softmax(selected_teacher_logits, dim=1)
    student_probs = F.log_softmax(selected_student_logits, dim=1)

    loss = weight*F.kl_div(student_probs, teacher_probs, size_average=False) * (T**2) / teacher_logits.size()[0]

    # recursion operation
    next_loss = calculate_recursion_combinations(teacher_logits, student_logits, T, remaining_indices,remaining_index_combinations)

    return loss + next_loss


def selective_softmax_loss(student_logits, teacher_logits, T, k):

    _, top_indices = student_logits.topk(k, 1, True, True)
    all_index_combinations = list(combinations(range(k), 2)) + [tuple(range(k))]


    index_combinations = [top_indices[:, list(comb)].cuda() for comb in all_index_combinations]
    index_finally = index_combinations[-1]

    #local logit combination
    total_loss = calculate_recursion_combinations(teacher_logits, student_logits, T, index_combinations,all_index_combinations)
    ldrld = total_loss / len(index_combinations)

    # remaining non-target knowledge
    not_mask = torch.zeros_like(teacher_logits).scatter_(1, index_finally, 1).bool()
    not_selected_teacher_logits = teacher_logits / T - 1000.0 * not_mask
    not_selected_student_logits = student_logits / T - 1000.0 * not_mask
    not_teacher_probs = F.softmax(not_selected_teacher_logits, dim=1)
    not_student_probs = F.log_softmax(not_selected_student_logits, dim=1)
    rntk = F.kl_div(not_student_probs, not_teacher_probs, size_average=False) * (T ** 2) / teacher_logits.size()[0]

    return ldrld, rntk

class LDRLD(nn.Module):

    def __init__(self, alpha, beta, k, temperature, ce_loss_weight):
        super(LDRLD, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.temperature = temperature
        self.ce_loss_weight = ce_loss_weight

    def forward(self, logits_student, logits_teacher, target):
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd, not_loss_kd = selective_softmax_loss(logits_student, logits_teacher, self.temperature, self.k)
        loss_ldrld = self.alpha * loss_kd + self.beta * not_loss_kd

        losses = loss_ce + loss_ldrld

        return losses
