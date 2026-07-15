import os
import torch
import random
import logging
import datetime
import numpy as np
from tqdm import tqdm
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from utils.parser import parser_with_args
from sklearn.metrics import precision_recall_fscore_support
from utils.helpers import (load_model, get_loaders, set_metrics, get_criterion,
                           get_mean_metrics, initialize_metrics)


parser, metadata = parser_with_args()
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=42)

def get_scheduler(optimizer, opt, lr_policy):
    if lr_policy == 'linear':
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - epoch / float(opt.epochs + 1))
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,step_size=opt.epochs // 3,gamma=0.1)
    else:
        raise NotImplementedError(f'LR policy [{lr_policy}] is not implemented')
    return scheduler

opt = parser.parse_args()
opt.epochs = 100
opt.batch_size = 8
opt.loss_function = "hybrid"
opt.dataset_dir = os.path.join(os.path.abspath(opt.dataset_dir), '')
run_name = f"{opt.dataset}_{opt.backbone}_{opt.mode}"
save_path = f".tmp/{run_name}"
writer = SummaryWriter(f".tmp/log/{run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

model = load_model(opt, dev)
train_loader, val_loader = get_loaders(opt)
best_metrics = {'f1_scores': -1, 'recalls': -1, 'precisions': -1}

criterion = get_criterion(opt)
if opt.backbone == 'vitae':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99, weight_decay=0.0005)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)

scheduler = get_scheduler(optimizer, opt, 'linear')
total_step = -1

for epoch in range(opt.epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()

    model.train()
    logging.info('Starting training phase')
    batch_iter = 0
    tbar = tqdm(train_loader)
    for img1, img2, labels, fname in tbar:
        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1
   
        img1 = img1.float().to(dev)
        img2 = img2.float().to(dev)
        labels = labels.long().to(dev)

        optimizer.zero_grad()
        preds = model(img1, img2)

        cd_loss = criterion(preds, labels)
        loss = cd_loss
        loss.backward()
        optimizer.step()

        _, preds = torch.max(preds, 1)

        Overall_acc = ((preds.squeeze().byte() == labels.squeeze().byte()).sum() / (labels.size()[0] * (opt.patch_size**2)))
        cd_train_report = precision_recall_fscore_support(labels.data.cpu().numpy().flatten(),
                            preds.data.cpu().numpy().flatten(),
                            average='binary',
                            pos_label=1,
                            zero_division=0
                            )

        train_metrics = set_metrics(train_metrics,
                                    cd_loss,
                                    Overall_acc,
                                    cd_train_report,
                                    scheduler.get_last_lr())

        mean_train_metrics = get_mean_metrics(train_metrics)

        for k, v in mean_train_metrics.items():
            writer.add_scalars(str(k), {'train': v}, total_step)

        del img1, img2, labels

    scheduler.step()
    logging.info(f"EPOCH {epoch} TRAIN METRICS: {mean_train_metrics}")


    model.eval()
    with torch.no_grad():
        for img1, img2, labels, fname in val_loader:
            img1 = img1.float().to(dev)
            img2 = img2.float().to(dev)
            labels = labels.long().to(dev)

            preds = model(img1, img2)
            cd_loss = criterion(preds, labels)
            _, preds = torch.max(preds, 1)
            Overall_acc = ((preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (opt.patch_size**2)))
            
            cd_val_report = precision_recall_fscore_support(labels.data.cpu().numpy().flatten(),
                                 preds.data.cpu().numpy().flatten(),
                                 average='binary',
                                 pos_label=1,
                                 zero_division=0
                                 )

            val_metrics = set_metrics(val_metrics,
                                      cd_loss,
                                      Overall_acc,
                                      cd_val_report,
                                      scheduler.get_last_lr())

            mean_val_metrics = get_mean_metrics(val_metrics)

            for k, v in mean_train_metrics.items():
                writer.add_scalars(str(k), {'val': v}, total_step)

            del img1, img2, labels


        logging.info(f"EPOCH {epoch} VALIDATION METRICS: {mean_val_metrics}")
        improved = (
            mean_val_metrics['precisions'] > best_metrics['precisions'] or
            mean_val_metrics['recalls']    > best_metrics['recalls']    or
            mean_val_metrics['f1_scores']  > best_metrics['f1_scores']
            )
        if improved:
            logging.info('Updating the model')
            metadata['validation_metrics'] = mean_val_metrics
            os.makedirs(save_path, exist_ok=True)
            torch.save(model, f"{save_path}/checkpoint_epoch_{epoch}.pth")
            best_metrics = mean_val_metrics

        print(f'Epoch {epoch} completed.')
writer.close()  
print('Training complete.')