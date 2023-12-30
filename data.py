""""Script with functions for loading the corrupted version of the data. It assumes that 
we have a folder in which the train and test images are saved using the convention 
test_images_0 or train_images_0 and their corresponding labels are test_targets_0 or 
train_targets_0.  """

import torch 
from torch.utils.data import TensorDataset, DataLoader
import os 


def load(folder) : 
    
    files = os.listdir(folder)

    train_images = []
    test_images = []

    train_targets = []
    test_targets = []

    for i in files : 
        if "train_images" in i : 
            train_images.append(torch.load(folder + i))
            train_targets.append(torch.load(folder + i.replace("images", "target")))

        elif "test_images" in i : 
            test_images.append(torch.load(folder + i))
            test_targets.append(torch.load(folder + i.replace("images","target")))
    train_images, train_targets, test_images, test_targets = torch.concatenate(train_images), torch.concatenate(train_targets), torch.concatenate(test_images), torch.concatenate(test_targets)

    print("Found a total of {} train images and {} test images. ".format(train_images.shape[0], test_images.shape[0]))

    # Create a TensorDataset
    train_images = train_images.unsqueeze(1)
    dataset_train = TensorDataset(train_images, train_targets)
    # Specify batch size for DataLoader
    batch_size = 64
    # Create a DataLoader
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    test_images = test_images.unsqueeze(1)
    dataset_test = TensorDataset(test_images, test_targets)
    test_loader = DataLoader(dataset_test)
    return train_loader, test_loader


