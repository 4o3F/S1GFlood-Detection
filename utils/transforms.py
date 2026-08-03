import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms as transforms


_ALLOWED_MASK_VALUES = {0, 1, 255}


def _temporal_targets(sample):
    if 'targets' in sample:
        return sample['targets'], False
    return {'change': sample['label']}, True


def _temporal_sample(sample, images, targets, legacy):
    result = dict(sample)
    result['image'] = images
    if legacy:
        result.pop('targets', None)
        result['label'] = targets['change']
    else:
        result.pop('label', None)
        result['targets'] = targets
    return result


def _transform_targets(targets, operation):
    return {name: operation(mask) for name, mask in targets.items()}


def _mask_to_tensor(mask):
    array = np.asarray(mask)
    values = set(np.unique(array).tolist())
    if not values.issubset(_ALLOWED_MASK_VALUES):
        raise ValueError(
            f'mask must contain only {{0,1,255}}, found {sorted(values)}'
        )
    return torch.from_numpy(np.asarray(array > 0, dtype=np.float32)).float()


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, sample):
        img1, img2 = sample['image']
        targets, legacy = _temporal_targets(sample)
        if any(img1.size != mask.size or img2.size != mask.size for mask in targets.values()):
            raise ValueError('image and target sizes must match before resize')

        images = (
            img1.resize(self.size, Image.BILINEAR),
            img2.resize(self.size, Image.BILINEAR),
        )
        targets = _transform_targets(
            targets,
            lambda mask: mask.resize(self.size, Image.NEAREST),
        )
        return _temporal_sample(sample, images, targets, legacy)


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'label': mask}


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img, 'label': mask}


class ToTensor(object):
    def __call__(self, sample):
        img1, img2 = sample['image']
        targets, legacy = _temporal_targets(sample)
        images = (
            torch.from_numpy(
                np.asarray(img1, dtype=np.float32).transpose((2, 0, 1))
            ).float(),
            torch.from_numpy(
                np.asarray(img2, dtype=np.float32).transpose((2, 0, 1))
            ).float(),
        )
        tensor_targets = {
            name: _mask_to_tensor(mask)
            for name, mask in targets.items()
        }
        return _temporal_sample(sample, images, tensor_targets, legacy)


class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1, img2 = sample['image']
        targets, legacy = _temporal_targets(sample)
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            targets = _transform_targets(
                targets,
                lambda mask: mask.transpose(Image.FLIP_TOP_BOTTOM),
            )
        return _temporal_sample(sample, (img1, img2), targets, legacy)


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1, img2 = sample['image']
        targets, legacy = _temporal_targets(sample)
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            targets = _transform_targets(
                targets,
                lambda mask: mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return _temporal_sample(sample, (img1, img2), targets, legacy)


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img1, img2 = sample['image']
        targets, legacy = _temporal_targets(sample)
        rotate_degree = random.uniform(-self.degree, self.degree)
        images = (
            img1.rotate(rotate_degree, Image.BILINEAR),
            img2.rotate(rotate_degree, Image.BILINEAR),
        )
        targets = _transform_targets(
            targets,
            lambda mask: mask.rotate(rotate_degree, Image.NEAREST),
        )
        return _temporal_sample(sample, images, targets, legacy)


class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img1, img2 = sample['image']
        targets, legacy = _temporal_targets(sample)
        if random.random() < 0.75:
            rotate_degree = random.choice(self.degree)
            img1 = img1.transpose(rotate_degree)
            img2 = img2.transpose(rotate_degree)
            targets = _transform_targets(
                targets,
                lambda mask: mask.transpose(rotate_degree),
            )
        return _temporal_sample(sample, (img1, img2), targets, legacy)


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)

        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img1, img2 = sample['image']
        targets, legacy = _temporal_targets(sample)
        if random.random() < 0.5:
            img1 = img1.filter(ImageFilter.GaussianBlur(radius=random.random()))
            img2 = img2.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return _temporal_sample(sample, (img1, img2), targets, legacy)


train_transforms = transforms.Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomFixRotate(),
    ToTensor(),
])

test_transforms = transforms.Compose([
    ToTensor(),
])
