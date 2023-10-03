import utils

import torch
import torchvision 

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def get_train_loader():
    transformer = transforms.Compose([
        transforms.Resize((utils.IMG_SIZE, utils.IMG_SIZE)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(utils.TRAIN_PATH, transform=transformer)
    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=utils.BATCH_SIZE,
        shuffle=utils.SHUFFLE,
        num_workers=utils.NUM_WORKERS,
        pin_memory=utils.PIN_MEMMORY
    )

    return train_loader, class_names

def get_test_loader():
    transformer = transforms.Compose([
        transforms.Resize((utils.IMG_SIZE, utils.IMG_SIZE)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(utils.TEST_PATH, transform=transformer)
    class_names = train_dataset.classes

    test_loader = DataLoader(
        train_dataset,
        batch_size=utils.BATCH_SIZE,
        shuffle=utils.SHUFFLE,
        num_workers=utils.NUM_WORKERS,
        pin_memory=utils.PIN_MEMMORY
    )

    return test_loader, class_names


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    train_loader, class_list = get_test_loader()
    image_batch, label_batch = next(iter(train_loader))
    train_img, train_lbl = image_batch[0], label_batch[0]

    print(train_img.shape, train_lbl)

    # Plot image with matplotlib
    plt.imshow(train_img.permute(1, 2, 0)) 
    plt.title(class_list[train_lbl])
    plt.axis(False)
    plt.show()