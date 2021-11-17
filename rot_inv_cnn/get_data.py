import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        assert len(data) == len(labels)
        self.data = data.copy()
        try:
            self.targets = labels.clone()
        except:
            self.targets = labels.copy()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L' if img.shape[-1] != 3 else 'RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def duplicate_rotation(data_set, transform):
    images = data_set.data
    labels = data_set.targets

    rotate_datapoints_list = lambda l, k: list(map(lambda datapoint: np.rot90(datapoint, k), l))
    images_90 = rotate_datapoints_list(images, 1)
    images_180 = rotate_datapoints_list(images, 2)
    images_270 = rotate_datapoints_list(images, 3)

    duplicated_dataset = torch.utils.data.ConcatDataset([data_set,
                                                         SimpleDataset(images_90, labels, transform),
                                                         SimpleDataset(images_180, labels, transform),
                                                         SimpleDataset(images_270, labels, transform)])
    assert len(duplicated_dataset) == 4 * len(data_set)
    return duplicated_dataset


def main(use_cuda, train_batch_size, test_batch_size, duplicate_train=False, duplicate_test=False,
         dataset_type='mnist'):
    mnist_data_folder = 'code/rot_inv_cnn/data'
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if dataset_type.lower() == 'mnist':
        dataset_type = datasets.MNIST
        data_transform = transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ])
        n_input_channels = 1
        data_width = 28
    elif dataset_type.lower() == 'fashion_mnist':
        dataset_type = datasets.FashionMNIST
        data_transform = transforms.ToTensor()
        n_input_channels = 1
        data_width = 28
    elif dataset_type.lower() == 'cifar10':
        dataset_type = datasets.CIFAR10
        data_transform = transforms.ToTensor()
        n_input_channels = 3
        data_width = 32
    else:
        raise ValueError('dataset %s not supported' % dataset_type.lower())

    print('Using %s with%s duplicate of training samples' % (dataset_type, 'out' if not duplicate_train else ''))

    train_val_set = dataset_type(mnist_data_folder, train=True, transform=data_transform, download=True)
    if duplicate_train:
        train_val_set = duplicate_rotation(train_val_set, data_transform)

    val_indexes = np.random.choice(list(range(0, len(train_val_set))), int(.2 * len(train_val_set)), replace=False)
    train_indexes = [i for i in range(0, len(train_val_set)) if i not in val_indexes]

    train_set = torch.utils.data.Subset(train_val_set, train_indexes)
    val_set = torch.utils.data.Subset(train_val_set, val_indexes)
    # train_size = int(.8 * len(train_val_set))
    # train_set = torch.utils.data.Subset(train_val_set, list(range(0, train_size)))
    # val_set = torch.utils.data.Subset(train_val_set, list(range(train_size, len(train_val_set))))

    test_set = dataset_type(mnist_data_folder, train=False, transform=data_transform, download=True)
    if duplicate_test:
        test_set = duplicate_rotation(test_set, data_transform)

    print('Sizes', 'train', len(train_set), 'val', len(val_set), 'test', len(test_set))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, train_set, val_set, test_set, n_input_channels, data_width


if __name__ == '__main__':
    *_, test_set = main(True, 10, 10)
    duplicated_dataset = duplicate_rotation(test_set)
    print(len(duplicated_dataset))
