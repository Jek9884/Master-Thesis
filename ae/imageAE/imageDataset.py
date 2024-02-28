from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, images_list, transform=None):
        self.images_list = images_list
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = self.images_list[idx]

        if self.transform:
            image = self.transform(image)

        #image = torch.tensor(image, dtype=torch.float32)
        image = image.float()

        return image