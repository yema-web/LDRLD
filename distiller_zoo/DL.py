from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def top_k_mask(logits,maxk=2):
    pred_s_value, pred_s_index = logits.topk(maxk, 1, True, True)
    mask = torch.ones_like(logits).scatter_(1, pred_s_index, 0).bool()
    return mask

def top_k_not_mask(logits,maxk=2):
    pred_s_value, pred_s_index = logits.topk(maxk, 1, True, True)
    mask = torch.zeros_like(logits).scatter_(1, pred_s_index, 1).bool()
    return mask

def recursive_decouple(t, mask,depth, results):
    if depth == 0:
        results.append((t, t))
        return t, t
    tmask = top_k_mask(t,2)
    ntmask = top_k_not_mask(t,2)
    t / self.T - 1000.0 * tmask
    t1 = (t * mask).sum(dim=1, keepdim=True)
    t2 = t * ~mask
    results.append((t1, t2))
    part2_1, part2_2 = recursive_decouple(t2, mask,depth - 1, results)
    return t1, torch.cat([part2_1, part2_2], dim=1)


def recursive_decouple(student_logits, teacher_logits, depth, student_results, teacher_results):
    if depth == 0:
        student_results.append((student_logits, student_logits))
        teacher_results.append((teacher_logits, teacher_logits))
        return student_logits, student_logits, teacher_logits, teacher_logits

    topk_mask = top_k_mask(teacher_logits, depth)
    ntopk_mask = top_k_not_mask(teacher_logits, depth)
    student_part1 = student_logits - 1000.0 * topk_mask
    student_part2 = student_logits - 1000.0 * ntopk_mask
    teacher_part1 = teacher_logits - 1000.0 * topk_mask
    teacher_part2 = teacher_logits - 1000.0 * ntopk_mask

    student_results.append((student_part1, student_part2))
    teacher_results.append((teacher_part1, teacher_part2))

    #depth = max(2, depth - 1)  # 动态调整 top-k 的值

    part2_1_student, part2_2_student, part2_1_teacher, part2_2_teacher = recursive_decouple(
        student_part2, teacher_part2, depth-1, student_results, teacher_results
    )

    return student_part1, torch.cat([part2_1_student, part2_2_student], dim=1), teacher_part1, torch.cat(
        [part2_1_teacher, part2_2_teacher], dim=1)


class WDKD(nn.Module):
    def __init__(self):
        super(WDKD, self).__init__()

    def forward(self, logits_student, logits_teacher, target, alpha, beta, temperature):
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        student_results = []
        teacher_results = []
        depth = 2
        nt = logits_student - 1000.0 * gt_mask
        tt = logits_teacher - 1000.0 * gt_mask

        recursive_decouple(nt,tt, depth, student_results, teacher_results)
        #recursive_decouple(logits_teacher - 1000.0 * gt_mask, teacher_logits, k, depth, student_results, teacher_results)

        tckd_loss_ = 0
        nckd_loss_ = 0
        for i in range(len(student_results)):
            student_part1, student_part2 = student_results[i]
            teacher_part1, teacher_part2 = teacher_results[i]

            #log_pred_student1 = torch.log(student_part1)
            tckd_loss1 = F.kl_div(F.log_softmax(student_part1/temperature,dim=1), F.softmax(teacher_part1/temperature,dim=1), size_average=False)* (temperature ** 2)/ target.shape[0]
            #log_pred_student2 = torch.log(student_part2)
            nckd_loss1 = F.kl_div(F.log_softmax(student_part2/temperature,dim=1), F.softmax(teacher_part2/temperature,dim=1), size_average=False)* (temperature ** 2)/ target.shape[0]

            tckd_loss_ = tckd_loss_ + tckd_loss1
            nckd_loss_ = nckd_loss_ + nckd_loss1
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nnckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )

        #print(beta * (nckd_loss_+tckd_loss_))
        return alpha * tckd_loss + beta * (0.4*nckd_loss_+0.6*nnckd_loss)