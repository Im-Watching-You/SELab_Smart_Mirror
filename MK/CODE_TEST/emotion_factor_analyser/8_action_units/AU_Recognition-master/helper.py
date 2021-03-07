import shutil
import torch
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterList(object):
    def __init__(self, length):
        self.val = [0] * length
        self.avg = [0] * length
        self.sum = [0] * length
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = (np.array(val) * n + np.array(self.sum)).tolist()
        self.count += n
        self.avg = (np.array(self.sum) / self.count).tolist()


def accuracy(output, target):
    pred = (output >= 0.5).float()
    acc = pred.eq(target).sum(dim=0).float().div(output.size(0)).mul_(100)
    return acc.tolist()


def f1score_helper(output, target, eps=1e-5):
    # pred = (output >= 0.5).float()
    # tp = (pred * target).sum(dim=0)
    # p = pred.sum(dim=0)
    # precision = tp / (p+eps)
    # t = target.sum(dim=0)
    # recall = tp / (t+eps)
    # f1 = 2 * precision * recall / (precision + recall)
    # return f1.cpu().tolist()
    pred = (output >= 0.5).astype(np.float32)
    tp = np.sum(pred * target, axis=0)
    p = np.sum(pred, axis=0)
    precision = tp / (p + eps)
    t = np.sum(target, axis=0)
    recall = tp / (t + eps)
    f1 = 2 * recall * precision / (precision + recall)
    return f1


def adjust_learning_rate(optimizer, epoch, init_lr, steps=50):
    """Sets the learning rate to the initial LR decayed by 0.5 every steps epochs"""
    lr = init_lr * (0.5 ** (epoch // steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='au_model'):
    torch.save(state, filename+'.pth')
    if is_best:
        shutil.copyfile(filename+'.pth', filename+'_best.pth')
