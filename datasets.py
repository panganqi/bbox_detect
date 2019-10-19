import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        # with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
        #     self.images = json.load(j)
        # with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
        #     self.objects = json.load(j)
        with open(os.path.join(data_folder,'/annot/'+split+'.json'),'r') as j:
            self.json_file = json.load(j)

        # assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        item_i = self.json_file[i]
        image = Image.open(self.data_folder+'/images/'+item_i['image'], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        # objects = self.objects[i]
        center = item_i['center']
        scale = item_i['scale']
        bbox  = [center[0] - 100*scale, center[1]-100*scale, center[0] + 100*scale ,center[1] + 100*scale]
        boxes = torch.FloatTensor(bbox)  # (n_objects, 4)
        labels = torch.LongTensor([1])  # (n_objects)
        difficulties = torch.ByteTensor([0])  # (n_objects)

        # Discard difficult objects, if desired
        # if not self.keep_difficult:
        #     boxes = boxes[1 - difficulties]
        #     labels = labels[1 - difficulties]
        #     difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.json_file)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
