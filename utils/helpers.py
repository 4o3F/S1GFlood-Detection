import torch
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from utils.dataloaders import train_path, test_path, FloodDetection
from torch.autograd import Variable
from networks import DAMNet_New

def get_loaders(opt):
    train_full_load, val_full_load = train_path(opt.dataset_dir)
    train_dataset = FloodDetection(
        train_full_load,
        flag='train',
        aug=opt.augmentation,
        include_water=True,
        load_water_labels=opt.water_loss_weight > 0,
    )
    val_dataset = FloodDetection(
        val_full_load,
        flag='val',
        aug=False,
        include_water=True,
        load_water_labels=opt.water_loss_weight > 0,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers,pin_memory=True)

    return train_loader, val_loader


def get_test_loaders(opt, batch_size=None):
    if not batch_size:
        batch_size = opt.batch_size

    test_full_load = test_path(opt.dataset_dir)
    test_dataset = FloodDetection(test_full_load, flag='test', aug=False)

    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=opt.num_workers)

    return test_loader


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))

        target = target.view(-1, 1)
        logpt = F.log_softmax(input,dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def dice_loss(logits, true, eps=1e-7):
    num_classes = logits.shape[1]
    labels = true.squeeze(1) if true.ndim == logits.ndim else true
    if labels.ndim != logits.ndim - 1:
        raise ValueError(
            'segmentation targets must have shape [N,H,W] or [N,1,H,W]'
        )

    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1, device=true.device)[labels]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes, device=true.device)[labels]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)

    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, probas.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    score = (2. * intersection / (cardinality + eps)).mean()
    return 1 - score


def hybrid_loss(predictions, target):
    loss = 0
    focal = FocalLoss(gamma=0, alpha=None)
    if isinstance(predictions, torch.Tensor):
        predictions = (predictions,)

    for prediction in predictions:
        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss

def get_criterion(opt):
    if opt.loss_function == 'hybrid':
        criterion = hybrid_loss
    if opt.loss_function == 'dice':
        criterion = dice_loss
    if opt.loss_function == 'bce':
        criterion = nn.CrossEntropyLoss()

    return criterion


def initialize_metrics():
    return {
        'losses': [],
        'change_losses': [],
        'water_losses': [],
        'weighted_water_losses': [],
        'water_supervised_samples': [],
        'water_supervision_fraction': [],
        'overall_accuracy': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': [],
        'learning_rate': [],
        '_batch_sizes': [],
    }


def get_mean_metrics(metric_dict):
    supervised_counts = metric_dict['water_supervised_samples']
    supervised_samples = int(np.sum(supervised_counts))
    total_samples = int(np.sum(metric_dict['_batch_sizes']))
    water_loss_sum = sum(
        loss * count
        for loss, count in zip(
            metric_dict['water_losses'],
            supervised_counts,
        )
    )
    weighted_water_loss_sum = sum(
        loss * count
        for loss, count in zip(
            metric_dict['weighted_water_losses'],
            supervised_counts,
        )
    )

    metrics = {
        key: np.mean(values)
        for key, values in metric_dict.items()
        if not key.startswith('_')
        and key not in {
            'water_losses',
            'weighted_water_losses',
            'water_supervised_samples',
            'water_supervision_fraction',
        }
    }
    metrics['water_losses'] = (
        water_loss_sum / supervised_samples
        if supervised_samples
        else 0.0
    )
    metrics['weighted_water_losses'] = (
        weighted_water_loss_sum / supervised_samples
        if supervised_samples
        else 0.0
    )
    metrics['water_supervised_samples'] = supervised_samples
    metrics['water_supervision_fraction'] = (
        supervised_samples / total_samples
        if total_samples
        else 0.0
    )
    return metrics


def set_metrics(
    metric_dict,
    total_loss,
    overall_accuracy,
    report,
    lr,
    *,
    change_loss,
    water_loss,
    water_loss_weight,
    water_supervised_samples,
    batch_size,
):
    metric_dict['losses'].append(total_loss.item())
    metric_dict['change_losses'].append(change_loss.item())
    metric_dict['water_losses'].append(water_loss.item())
    metric_dict['weighted_water_losses'].append(
        water_loss_weight * water_loss.item()
    )
    metric_dict['water_supervised_samples'].append(water_supervised_samples)
    metric_dict['_batch_sizes'].append(batch_size)
    metric_dict['learning_rate'].append(lr)
    metric_dict['overall_accuracy'].append(overall_accuracy.item())
    metric_dict['precisions'].append(report[0])
    metric_dict['recalls'].append(report[1])
    metric_dict['f1_scores'].append(report[2])

    return metric_dict

def set_test_metrics(metric_dict, overall_accuracy, report):
    metric_dict['overall_accuracy'].append(overall_accuracy.item())
    metric_dict['precisions'].append(report[0])
    metric_dict['recalls'].append(report[1])
    metric_dict['f1_scores'].append(report[2])

    return metric_dict

def load_model(opt, device):
    model = DAMNet_New(opt, input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8).to(device)

    return model