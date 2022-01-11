import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from abc import abstractmethod
from torchvision.datasets import ImageFolder
import configs

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append("../")


def identity(x):
    return x


class SimpleDataset:
    def __init__(self, transform, datasetpath,
                 target_transform=identity, dataset=ImageFolder,
                 transfer_class_list=None):
        self.transform = transform
        self.target_transform = target_transform
        self.transfer_class_list = transfer_class_list

        self.meta = {}
        self.meta['image_names'] = []
        self.meta['image_labels'] = []
        self.meta['image_classes'] = []
        d = dataset(datasetpath)
        self.meta['image_classes'] = d.classes
        for sample in d.samples:
            self.meta['image_names'].append(sample[0])
            self.meta['image_labels'].append(sample[1])

    def __getitem__(self, i):
        img = self.transform(Image.open(
            self.meta['image_names'][i]).convert('RGB'))
        # set target to index of the class
        target = self.meta['image_labels'][i]

        # if transform, map indexes between datasets
        if self.transfer_class_list is not None:
            class_name = self.meta['image_classes'][target]
            target = self.transfer_class_list.index(class_name)

        target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, batch_size, transform, datasetpath,
                 class_num, dataset=ImageFolder, transfer_class_list=None):

        self.sub_meta = {}
        self.cl_list = range(class_num)
        self.transfer_class_list = transfer_class_list
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        d = dataset(datasetpath)
        self.image_classes = d.classes
        for sample in d.samples:
            self.sub_meta[sample[1]].append(sample[0])

        # for key, item in self.sub_meta.items():
        #    print (len(self.sub_meta[key]))

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(
                self.sub_meta[cl], cl, self.image_classes[cl],
                transform=transform,
                transfer_class_list=self.transfer_class_list)
            self.sub_dataloader.append(torch.utils.data.DataLoader(
                sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)


class SubDataset:
    def __init__(self, sub_meta, cl,  cl_name, transform=transforms.ToTensor(),
                 target_transform=identity, transfer_class_list=None):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        self.transfer_class_list = transfer_class_list
        self.cl_name = cl_name

    def __getitem__(self, i):

        img = self.transform(Image.open(
            self.sub_meta[i]).resize((256, 256)).convert('RGB'))

        target = self.cl

        # if transform, map indexes between datasets
        if self.transfer_class_list is not None:
            target = self.transfer_class_list.index(self.cl_name)

        target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


class TransformLoader:
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[
                                      0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomResizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([int(self.image_size*1.15),
                           int(self.image_size*1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ['ToTensor', 'Normalize']
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

    # changed to match exactly to:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    # def get_composed_transform(self, aug=False):
    #     normalize = transforms.Normalize(
    #         mean=self.normalize_param["mean"],
    #         std=self.normalize_param["std"])
    #     if aug:
    #         transform = transforms.Compose([
    #             transforms.RandomResizedCrop(self.image_size),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])
    #     else:
    #         transform = transforms.Compose([
    #             transforms.Resize((int(self.image_size*1.15),
    #                              int(self.image_size*1.15))),
    #             transforms.CenterCrop(self.image_size),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])
    #     return transform


class DataManager(object):
    @ abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size,
                 datasetpath, dataset=ImageFolder, transfer_class_list=None):
        """[summary]

        Args:
            image_size ([type]): [description]
            batch_size ([type]): [description]
            datasetpath ([type]): [description]
            dataset ([type], optional): [description]. Defaults to ImageFolder.
            transfer_class_list ([type], optional): [list of source dataset
                to map to target classes indexes]. Defaults to None.
        """
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.datasetpath = datasetpath
        self.dataset = dataset
        self.transfer_class_list = transfer_class_list

    def get_data_loader(self, aug):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(
            transform, self.datasetpath, dataset=self.dataset,
            transfer_class_list=self.transfer_class_list)

        data_loader_params = dict(
            batch_size=self.batch_size, shuffle=True,
            num_workers=configs.num_workers, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, **data_loader_params)

        return data_loader


class SetDataManager(DataManager):
    def __init__(self, image_size, datasetpath, dataset=ImageFolder,
                 n_support=5, n_query=2, class_num=5, n_eposide=100,
                 transfer_class_list=None):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        self.datasetpath = datasetpath
        self.class_num = class_num
        self.dataset = dataset
        self.transfer_class_list = transfer_class_list

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug, n_way):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform,
                             self.datasetpath, self.class_num,
                             dataset=self.dataset,
                             transfer_class_list=self.transfer_class_list)
        sampler = EpisodicBatchSampler(
            len(dataset), n_way, self.n_eposide)
        data_loader_params = dict(
            batch_sampler=sampler,
            num_workers=configs.num_workers, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, **data_loader_params)
        return data_loader


if __name__ == '__main__':
    pass
