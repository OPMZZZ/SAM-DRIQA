from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
import random
import numpy as np
import pandas as pd
from config import MyConfig
import matplotlib.pyplot as plt
from PIL import ImageOps
import pydicom
from torch.nn import functional as F
def mask_maker(img):
    # copy_img = img[:]
    # copy_img = (copy_img - img.min()) / (img.max() - img.min()) * 255
    # otsu.load_image(copy_img[i, 0])
    # kThresholds = otsu.calculate_k_thresholds(3)
    # # print(kThresholds)
    # for j in range(len(kThresholds)):
    #     tmp = np.zeros_like(copy_img[0])
    #     tmp[copy_img[j] > kThresholds[j]] = 1
    #     copy_img[j] = tmp
    width, height = img.shape[-2:]

    # 创建一个全零的640x640数组
    mask = np.zeros((width, height), dtype=np.uint8)

    # 掩码的宽度和高度
    mask_width = 400
    mask_height = height

    # 掩码的起始和结束水平索引（图像中间）
    start_x = (height - mask_width) // 2

    # 在全零数组上画掩码
    mask[:, start_x:start_x + mask_width] = 1
    # plt.imshow(mask)
    # plt.show()

    return mask


def dice(pred, gt, smooth=1e-5):
    intersection = (2 * pred * gt).sum()

    return intersection.item()


class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_excel, test, is_train=True, is_d=True):
        self.is_d = is_d
        self.config = MyConfig().config
        self.is_train = is_train
        self.images_dir = images_dir
        self.test_path = test
        self.images0 = sorted(os.listdir(os.path.join(images_dir, "0")), key=lambda x: int(x.split(".")[0]))
        self.images1 = sorted(os.listdir(os.path.join(images_dir, "1")), key=lambda x: int(x.split(".")[0]))
        self.index = self.get_index()
        # self.labels1 = pd.read_excel(labels_excel, header=None).loc[:,
        #                [1, 2, 3, 4] + [x for x in range(9, 17)]].values.tolist()
        # self.labels2 = pd.read_excel(labels_excel, header=None).loc[:,
        #                [5, 6, 7, 8] + [x for x in range(9, 17)]].values.tolist()
        # self.labels = [(x, y) for x, y in zip(self.labels1, self.labels2)]
        self.labels = pd.read_excel(labels_excel, header=None).loc[:, 1:self.config.output_size[0]].values.tolist()

        self.tx = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] > 3 else x),
            torchvision.transforms.Lambda(lambda x: self.Normalize(x)),
            # torchvision.transforms.Lambda(lambda x: self.Padding2Square(x)),
            # torchvision.transforms.Grayscale(),

            # torchvision.transforms.Resize((self.config.input_size[1:])),
            # torchvision.transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        i = self.index[i]
        if self.config.is_dicom and self.is_d:
            i0 = pydicom.read_file(os.path.join(self.images_dir, "0", self.images0[i])).pixel_array.astype(np.float32)

            i1 = pydicom.read_file(os.path.join(self.images_dir, "1", self.images1[i])).pixel_array.astype(np.float32)
        else:
            i0 = Image.open(os.path.join(self.images_dir, "0", self.images0[i]))
            i1 = Image.open(os.path.join(self.images_dir, "1", self.images1[i]))

        label = torch.tensor(self.labels[i]).float()

        seed = self.config.random_seed  # make a seed with numpy generator

        # apply this seed to img tranfsorms
        torch.manual_seed(seed)
        # plt.imshow(i0, cmap='gray')
        # plt.colorbar()
        # plt.show()
        i0 = self.tx(i0)

        i1 = self.tx(i1)

        # i0 = self.img_aug(i0)
        # i1 = self.img_aug(i1)
        sample = self.threeFoldScale(i0, i1)
        sample['score_tag'] = self.label_change(label)
        sample['score'] = label
        sample['name'] = self.images0[i]

        return sample

    def get_index(self):
        # 假设self.images0已经定义并且包含文件名
        total_images = len(self.images0)
        sample_size = int(total_images * 0.2)  # 计算20%的样本大小

        # 设置随机种子以确保结果的可重复性
        random.seed(self.config.random_seed)

        # 随机选择20%的索引作为test_index
        test_index = random.sample(range(total_images), sample_size)
        if self.is_train:
            index = [int(x.split(".")[0]) - 1 for x in self.images0]
            index = [x for x in index if x not in test_index]
            return index
        return test_index

    def threeFoldScale(self, img0, img1):
        # img0 = img0.expand(3, -1, -1)
        # img1 = img1.expand(3, -1, -1)
        return {'d_img_org': (img0, img1)}

    def Normalize(self, img):
        if self.config.is_dicom:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = img.expand(3, -1, -1)
        else:
            g = torchvision.transforms.Grayscale()
            img = g(img)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = img.expand(3, -1, -1)

        # img = img.expand(3, -1, -1)
        return img

    def RandomPad(self, img):
        r_pad = torchvision.transforms.Pad(random.randint(0, 500))

        return r_pad(img)

    def Padding2Square(self, img):
        max_one = max(img.size())

        pad_height = max_one - img.size(1)
        pad_width = max_one - img.size(2)

        pad_transform = torchvision.transforms.Pad((pad_width // 2, pad_height // 2, pad_width // 2, pad_height // 2))

        img = pad_transform(img)

        return img

    def label_change(self, label):
        c = 16
        l = torch.zeros((c, 12))
        for index, v in enumerate(label):
            l[index, int(v.item())] = 1
        return l

    # def img_aug(self, img):
    #     ia.seed(self.config.random_seed)
    #     img = img.numpy()
    #     img = img.astype(np.uint8)
    #     img = np.transpose(img, (1, 2, 0))
    #     if self.is_train:
    #         seq = iaa.Sequential([
    #             iaa.Dropout([0, 0.05]),
    #                         ])
    #         img = seq(images=img)
    #     img = np.transpose(img, (2, 0, 1))
    #     img = torch.from_numpy(img)
    #     return img


class chest_dataset(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_excel, is_train=True, is_d=True):
        self.is_d = is_d
        self.config = MyConfig().config
        self.is_train = is_train
        self.images_dir = images_dir
        self.images0 = sorted(os.listdir(os.path.join(images_dir, "0")), key=lambda x: int(x.split(".")[0]))
        self.images1 = sorted(os.listdir(os.path.join(images_dir, "1")), key=lambda x: int(x.split(".")[0]))
        self.index = self.get_index()
        # self.labels1 = pd.read_excel(labels_excel, header=None).loc[:,
        #                [1, 2, 3, 4] + [x for x in range(9, 17)]].values.tolist()
        # self.labels2 = pd.read_excel(labels_excel, header=None).loc[:,
        #                [5, 6, 7, 8] + [x for x in range(9, 17)]].values.tolist()
        # self.labels = [(x, y) for x, y in zip(self.labels1, self.labels2)]
        self.labels = pd.read_excel(labels_excel, header=None).loc[:, 1:self.config.output_size[0]].values.tolist()

        self.tx = torchvision.transforms.Compose([
            # torchvision.transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] > 3 else x),
            torchvision.transforms.Lambda(lambda x: self.Normalize(x)),
            torchvision.transforms.Lambda(lambda x: self.Padding2Square(x)),
            torchvision.transforms.Lambda(
                lambda x: self.RandomPad(x) if self.is_train else x),
            torchvision.transforms.Resize((self.config.input_size[1:])),
            torchvision.transforms.RandomHorizontalFlip(),
        ])

        self.togrey = torchvision.transforms.Grayscale()
        self.normalize = torchvision.transforms.Lambda(lambda x: self.Normalize(x))
        if self.config.is_dicom and self.is_d:
            self.image0_list = [np.array(pydicom.read_file(os.path.join(self.images_dir, "0", self.images0[i])).pixel_array.astype(np.float32))
                                for i in range(len(self.images0))]

            self.image1_list = [np.array(pydicom.read_file(os.path.join(self.images_dir, "1", self.images1[i])).pixel_array.astype(np.float32))
                                for i in range(len(self.images0))]
        else:
            self.image0_list = [np.array(self.togrey(Image.open(os.path.join(self.images_dir, "0", self.images0[i]))))
                                for i in range(len(self.images0))]
            self.image1_list = [np.array(self.togrey(Image.open(os.path.join(self.images_dir, "1", self.images1[i]))))
                                for i in range(len(self.images0))]
        # self.seg0_list = [np.load(os.path.join(self.images_dir, 'seg', "0", self.images0[i][:-4], 'mask.npy')) for i in range(len(self.images0))]
        # self.seg1_list = [np.load(os.path.join(self.images_dir, 'seg', "1", self.images1[i][:-4], 'mask.npy')) for i in range(len(self.images0))]

    def __len__(self):
        return len(self.index)


    def fliter(self, seg):
        tmp = np.zeros_like(seg[0]).astype(np.float32)
        num = list(range(seg.shape[0]))
        random.shuffle(num)
        seg = seg[num]
        for i in range(seg.shape[0]):
            tmp[seg[i] == 1] = i + 1
        return tmp[np.newaxis]
    def __getitem__(self, i):
        i = self.index[i]
        seg0 = [np.array(Image.open(os.path.join(self.images_dir,'seg', "0", self.images0[i][:-4], item))) for item in os.listdir(os.path.join(self.images_dir,'seg', "0", self.images0[i][:-4])) if item.endswith('jpg')][:3]
        seg1 = [np.array(Image.open(os.path.join(self.images_dir,'seg', "1", self.images0[i][:-4], item))) for item in os.listdir(os.path.join(self.images_dir,'seg', "1", self.images0[i][:-4]))if item.endswith('jpg')][:3]
        # random.shuffle(seg0)
        # random.shuffle(seg1)
        # seg0 = self.seg0_list[i]
        # seg1 = self.seg1_list[i]
        # seg0 = self.fliter(seg0)
        # seg1 = self.fliter(seg1)
        seed = self.config.random_seed  # make a seed with numpy generator

        torch.manual_seed(seed)
        seg0 = np.stack(seg0, axis=0)
        seg1 = np.stack(seg1, axis=0)
        # seg0 = self.normalize(seg0)
        # seg1 = self.normalize(seg1)
        i0 = self.image0_list[i]
        i1 = self.image1_list[i]

        label = torch.tensor(self.labels[i]).float()
        i0 = np.concatenate((seg0, i0[np.newaxis]), axis=0)
        i1 = np.concatenate((seg1, i1[np.newaxis]), axis=0)

        # plt.imshow(i0, cmap='gray')
        # plt.colorbar()
        # plt.show()
        i0 = self.tx(torch.from_numpy(i0)).float()
        # for im in i0:
        #     plt.imshow(im, cmap='gray')
        #     plt.colorbar()
        #     plt.show()
        i1 = self.tx(torch.from_numpy(i1)).float()

        # i0 = self.img_aug(i0)
        # i1 = self.img_aug(i1)
        sample = self.threeFoldScale(i0, i1)
        sample['score_tag'] = self.label_change(label)
        sample['score'] = label
        sample['name'] = self.images0[i]

        return sample

    def get_index(self):
        # 假设self.images0已经定义并且包含文件名
        total_images = len(self.images0)
        sample_size = int(total_images * 0.2)  # 计算20%的样本大小

        # 设置随机种子以确保结果的可重复性
        random.seed(self.config.random_seed)

        # 随机选择20%的索引作为test_index
        test_index = random.sample(range(total_images), sample_size)
        if self.is_train:
            index = [int(x.split(".")[0]) - 1 for x in self.images0]
            index = [x for x in index if x not in test_index]
            return index
        return test_index

    def threeFoldScale(self, img0, img1):
        return {'d_img_org': (img0, img1)}

    def Normalize(self, img):

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        # img = img.expand(3, -1, -1)
        return img

    def RandomPad(self, img):
        r_pad = torchvision.transforms.Pad(random.randint(0, 250))
        torchvision.transforms.Resize((640, 640)),
        return r_pad(img)

    def Padding2Square(self, img):
        max_one = max(img.size())

        pad_height = max_one - img.size(1)
        pad_width = max_one - img.size(2)

        pad_transform = torchvision.transforms.Pad((pad_width // 2, pad_height // 2, pad_width // 2, pad_height // 2))

        img = pad_transform(img)

        return img

    def label_change(self, label):
        c = 16
        l = torch.zeros((c, 12))
        for index, v in enumerate(label):
            l[index, int(v.item())] = 1
        return l
