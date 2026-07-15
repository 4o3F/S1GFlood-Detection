import torch
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from utils.dataloaders import train_path, test_path, FloodDetection
from utils.parser import parser_with_args
from torch.autograd import Variable
from networks import DAMNet_New


parser, metadata = parser_with_args()
opt = parser.parse_args()

def get_loaders(opt):
    train_full_load, val_full_load = train_path(opt.dataset_dir)
    train_dataset = FloodDetection(train_full_load, flag='train', aug=opt.augmentation)
    val_dataset = FloodDetection(val_full_load, flag='val', aug=False)

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
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1, device=true.device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)

    else:
        true_1_hot = torch.eye(num_classes, device=true.device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


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
    metrics = {'losses': [], 'overall_accuracy': [],'precisions': [],'recalls': [],'f1_scores': [],'learning_rate': []}
    return metrics

def get_mean_metrics(metric_dict):
    return {k: np.mean(v) for k, v in metric_dict.items()}


def set_metrics(metric_dict, cd_loss, overall_accuracy, report, lr):
    metric_dict['losses'].append(cd_loss.item())
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