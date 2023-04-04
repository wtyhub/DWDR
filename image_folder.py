import torch.utils.data as Data
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from os.path import join as ospj
from PIL import Image
import numpy as np
import os
import torch
from collections import defaultdict
import random

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images
#getting one image of a folder.
def make_dataset_one(dir, class_to_idx, extensions, reverse=False):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            index = 0
            for fname in sorted(fnames, reverse=reverse):
                index += 1
                if has_file_allowed_extension(fname, extensions) and index == 36:
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    break

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class customData(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, rotate = 0, pad = 0):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform 
        self.loader = loader
        self.rotate = rotate
        self.pad = pad
    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = self.imgs[index]
            img = self.loader(path)
            img = transforms.functional.rotate(img,self.rotate)
            if self.pad > 0:
                img = transforms.functional.resize(img,(256,256),interpolation=3)
                img = transforms.functional.pad(img,(self.pad,0,0,0))
                img = transforms.functional.five_crop(img,(256,256))[0]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.imgs)

class customData_one(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, rotate = 0, pad = 0, reverse=False):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset_one(root, class_to_idx, IMG_EXTENSIONS, reverse)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform 
        self.loader = loader
        self.rotate = rotate
        self.pad = pad
    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = self.imgs[index]
            img = self.loader(path)
            img = transforms.functional.rotate(img,self.rotate)
            if self.pad > 0:
                img = transforms.functional.resize(img,(256,256),interpolation=3)
                img = transforms.functional.pad(img,(self.pad,0,0,0))
                img = transforms.functional.five_crop(img,(256,256))[0]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.imgs)

def make_pair_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, target, class_to_idx[target])
                    images.append(item)
    return images


class SatData(Data.Dataset):
    def __init__(self, root, transform = None, d_transform = None, loader = default_loader, view='/drone/', dataset='university'):
        if dataset == 'cvact':
            sat_root = root + '/satview_polish/'
        else:
            sat_root = root + '/satellite/'
        classes, class_to_idx = find_classes(sat_root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_pair_dataset(sat_root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.d_transform = d_transform
        self.loader = loader
        self.view = view
    def _get_pair_sample(self, root, _cls):
        img_path = []
        folder_root = root + str(_cls)
        assert os.path.isdir(folder_root), 'no pair drone image'
        for file_name in os.listdir(folder_root):
            img_path.append(folder_root + '/' + file_name)
        rand = np.random.permutation(len(img_path))
        tmp_index = rand[0]
        result_path = img_path[tmp_index]
        return result_path

    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, _cls, target = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            d_root = self.root + self.view
            d_path = self._get_pair_sample(d_root, _cls)
            d_img = self.loader(d_path)
            if self.d_transform is not None:
                d_img = self.d_transform(d_img)
            return img, d_img, target

    def __len__(self):
        return len(self.imgs)

class DroneData(Data.Dataset):
    def __init__(self, root, transform = None, s_transform = None, loader = default_loader, view='/drone/', dataset='university'):
        drone_root = root + view
        classes, class_to_idx = find_classes(drone_root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_pair_dataset(drone_root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.s_transform = s_transform
        self.loader = loader
        self.dataset = dataset
    def _get_pair_sample(self, root, _cls):
        img_path = []
        folder_root = root + str(_cls)
        assert os.path.isdir(folder_root), 'no pair sat image'
        for file_name in os.listdir(folder_root):
            img_path.append(folder_root + '/' + file_name)
        rand = np.random.permutation(len(img_path))
        tmp_index = rand[0]
        result_path = img_path[tmp_index]
        return result_path

    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, _cls, target = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.dataset == 'cvact':
                s_root = self.root + '/satview_polish/'
            else:
                s_root = self.root + '/satellite/'
            s_path = self._get_pair_sample(s_root, _cls)
            s_img = self.loader(s_path)
            if self.s_transform is not None:
                s_img = self.s_transform(s_img)
            return img, s_img, target

    def __len__(self):
        return len(self.imgs)

# class AugSatData(Data.Dataset):
#     def __init__(self, root, transform = None, d_transform = None, loader = default_loader, view='/drone/', dataset='university'):
#         if dataset == 'cvact':
#             sat_root = root + '/satview_polish/'
#         else:
#             sat_root = root + '/satellite/'
#         classes, class_to_idx = find_classes(sat_root)
#         IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
#         imgs = make_pair_dataset(sat_root, class_to_idx, IMG_EXTENSIONS)
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

#         self.root = root
#         self.imgs = imgs
#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.transform = transform
#         self.d_transform = d_transform
#         self.loader = loader
#         self.view = view
#     def _get_pair_sample(self, root, _cls):
#         img_path = []
#         folder_root = root + str(_cls)
#         assert os.path.isdir(folder_root), 'no pair drone image'
#         for file_name in os.listdir(folder_root):
#             img_path.append(folder_root + '/' + file_name)
#         rand = np.random.permutation(len(img_path))
#         tmp_index = rand[0]
#         result_path = img_path[tmp_index]
#         return result_path

#     def __getitem__(self, index):
#             """
#             index (int): Index
#         Returns:tuple: (image, target) where target is class_index of the target class.
#             """
#             path, _cls, target = self.imgs[index]
#             img = self.loader(path)
#             if self.transform is not None:
#                 img1 = self.transform[0](img)
#                 img2 = self.transform[1](img)
#             d_root = self.root + self.view
#             d_path = self._get_pair_sample(d_root, _cls)
#             d_img = self.loader(d_path)
#             if self.d_transform is not None:
#                 d_img = self.d_transform(d_img)
#             return img1, img2, d_img, target

#     def __len__(self):
#         return len(self.imgs)

# class AugDroneData(Data.Dataset):
#     def __init__(self, root, transform = None, s_transform = None, loader = default_loader, view='/drone/', dataset='university'):
#         drone_root = root + view
#         classes, class_to_idx = find_classes(drone_root)
#         IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
#         imgs = make_pair_dataset(drone_root, class_to_idx, IMG_EXTENSIONS)
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

#         self.root = root
#         self.imgs = imgs
#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.transform = transform
#         self.s_transform = s_transform
#         self.loader = loader
#         self.dataset = dataset
#     def _get_pair_sample(self, root, _cls):
#         img_path = []
#         folder_root = root + str(_cls)
#         assert os.path.isdir(folder_root), 'no pair sat image'
#         for file_name in os.listdir(folder_root):
#             img_path.append(folder_root + '/' + file_name)
#         rand = np.random.permutation(len(img_path))
#         tmp_index = rand[0]
#         result_path = img_path[tmp_index]
#         return result_path

#     def __getitem__(self, index):
#             """
#             index (int): Index
#         Returns:tuple: (image, target) where target is class_index of the target class.
#             """
#             path, _cls, target = self.imgs[index]
#             img = self.loader(path)
#             if self.transform is not None:
#                 img1 = self.transform[0](img)
#                 img2 = self.transform[1](img)
#             if self.dataset == 'cvact':
#                 s_root = self.root + '/satview_polish/'
#             else:
#                 s_root = self.root + '/satellite/'
#             s_path = self._get_pair_sample(s_root, _cls)
#             s_img = self.loader(s_path)
#             if self.s_transform is not None:
#                 s_img = self.s_transform(s_img)
#             return img1, img2, s_img, target

#     def __len__(self):
#         return len(self.imgs)

class AugSatData(Data.Dataset):
    def __init__(self, root, transform = None, d_transform = None, loader = default_loader, view='/drone/', dataset='university'):
        if dataset == 'cvact':
            sat_root = root + '/satview_polish/'
        else:
            sat_root = root + '/satellite/'
        classes, class_to_idx = find_classes(sat_root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_pair_dataset(sat_root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.d_transform = d_transform
        self.loader = loader
        self.view = view
    def _get_pair_sample(self, root, _cls):
        img_path = []
        folder_root = root + str(_cls)
        assert os.path.isdir(folder_root), 'no pair drone image'
        for file_name in os.listdir(folder_root):
            img_path.append(folder_root + '/' + file_name)
        rand = np.random.permutation(len(img_path))
        tmp_index = rand[0]
        result_path = img_path[tmp_index]
        return result_path

    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, _cls, target = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img1 = self.transform[0](img)
                img2 = self.transform[1](img)
            d_root = self.root + self.view
            d_path = self._get_pair_sample(d_root, _cls)
            d_img = self.loader(d_path)
            if self.d_transform is not None:
                d_img1 = self.d_transform[0](d_img)
                d_img2 = self.d_transform[1](d_img)
            return img1, img2, d_img1, d_img2, target

    def __len__(self):
        return len(self.imgs)

class AugDroneData(Data.Dataset):
    def __init__(self, root, transform = None, s_transform = None, loader = default_loader, view='/drone/', dataset='university'):
        drone_root = root + view
        classes, class_to_idx = find_classes(drone_root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_pair_dataset(drone_root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.s_transform = s_transform
        self.loader = loader
        self.dataset = dataset
    def _get_pair_sample(self, root, _cls):
        img_path = []
        folder_root = root + str(_cls)
        assert os.path.isdir(folder_root), 'no pair sat image'
        for file_name in os.listdir(folder_root):
            img_path.append(folder_root + '/' + file_name)
        rand = np.random.permutation(len(img_path))
        tmp_index = rand[0]
        result_path = img_path[tmp_index]
        return result_path

    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, _cls, target = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img1 = self.transform[0](img)
                img2 = self.transform[1](img)
            if self.dataset == 'cvact':
                s_root = self.root + '/satview_polish/'
            else:
                s_root = self.root + '/satellite/'
            s_path = self._get_pair_sample(s_root, _cls)
            s_img = self.loader(s_path)
            if self.s_transform is not None:
                s_img1 = self.s_transform[0](s_img)
                s_img2 = self.s_transform[1](s_img)
            return img1, img2, s_img1, s_img2, target

    def __len__(self):
        return len(self.imgs)








def make_dataset_selectID(dir, class_to_idx, extensions):
    images = defaultdict(list)
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images[class_to_idx[target]].append(item)
    return images


class ImageFolder_selectID(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs.keys()) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = random.choice(self.imgs[index])
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            placeholder = torch.zeros_like(img)
            return img, placeholder, target

    def __len__(self):
        return len(self.imgs)

class ImageFolder_expandID(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        imgs = imgs * 3
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            placeholder = torch.zeros_like(img)
            return img, placeholder, target

    def __len__(self):
        return len(self.imgs)

