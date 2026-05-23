from tqdm import tqdm
import torch.utils.data
from utils.parser import parser_with_args
from utils.helpers import get_test_loaders
from sklearn.metrics import confusion_matrix

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser, metadata = parser_with_args()
opt = parser.parse_args()
opt.dataset_dir = './S1GFloods_example/'
test_loader = get_test_loaders(opt)
model = torch.load(opt.path, map_location='cpu')
model.to(dev)

CM = {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}
model.eval()

with torch.no_grad():
    tbar = tqdm(test_loader)
    for img1, img2, labels, fname in tbar:

        img1 = img1.float().to(dev)
        img2 = img2.float().to(dev)
        labels = labels.long().to(dev)

        preds = model(img1, img2)
        _, preds = torch.max(preds, 1)

        TN, FP, FN, TP = confusion_matrix(labels.data.cpu().numpy().flatten(),
                        preds.data.cpu().numpy().flatten(),labels=[0,1]).ravel()

        CM['TN'] += TN
        CM['FP'] += FP
        CM['FN'] += FN
        CM['TP'] += TP

TN, FP, FN, TP = CM['TN'], CM['FP'], CM['FN'], CM['TP']
P = TP / (TP + FP)
R = TP / (TP + FN)
F1 = 2 * P * R / (R + P)

print('Precision: {}\n Recall: {}\n F1-Score: {}'.format(P, R, F1))