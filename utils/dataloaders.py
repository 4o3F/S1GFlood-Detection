import os
import torch.utils.data as data
from PIL import Image
from utils import transforms as trans


_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def _list_images(directory):
    return sorted(
        name for name in os.listdir(directory)
        if not name.startswith('.') and name.lower().endswith(_IMAGE_EXTENSIONS)
    )


def train_path(data_dir):
    train_data = _list_images(data_dir + 'train/A/')
    valid_data = _list_images(data_dir + 'val/A/')

    train_label_paths = []
    val_label_paths = []
    for img in train_data:
        train_label_paths.append(data_dir + 'train/GT/' + img)
    for img in valid_data:
        val_label_paths.append(data_dir + 'val/GT/' + img)


    train_data_path = []
    val_data_path = []

    for img in train_data:
        train_data_path.append([data_dir + 'train/', img])
    for img in valid_data:
        val_data_path.append([data_dir + 'val/', img])

    train_dataset = {}
    val_dataset = {}
    for t in range(len(train_data)):
        train_dataset[t] = {'image': train_data_path[t], 'label': train_label_paths[t]}
    for t in range(len(valid_data)):
        val_dataset[t] = {'image': val_data_path[t], 'label': val_label_paths[t]}

    return train_dataset, val_dataset


def test_path(data_dir):
    test_data = _list_images(data_dir + 'test/A/')

    test_label_paths = []
    for img in test_data:
        test_label_paths.append(data_dir + 'test/GT/' + img)

    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir + 'test/', img])

    test_dataset = {}
    for t in range(len(test_data)):
        test_dataset[t] = {'image': test_data_path[t], 'label': test_label_paths[t]}

    return test_dataset

def FloodChange(img_path, label_path, aug):
    dir = img_path[0]
    name = img_path[1]

    img1 = Image.open(dir + 'A/' + name)
    img2 = Image.open(dir + 'B/' + name)
    label = Image.open(label_path)
    sample = {'image': (img1, img2), 'label': label}

    if aug:
        sample = trans.train_transforms(sample)
    else:
        sample = trans.test_transforms(sample)

    return sample['image'][0], sample['image'][1], sample['label'], name


class FloodDetection(data.Dataset):
    def __init__(self, full_load, flag = 'train', aug=False):
        self.full_load = full_load
        self.loader = FloodChange
        self.aug = aug

    def __getitem__(self, index):
        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']
        return self.loader(img_path, label_path, self.aug)

    def __len__(self):
        return len(self.full_load)