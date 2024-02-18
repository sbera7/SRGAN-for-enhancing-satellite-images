import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.root_dir = root_dir
        self.high_res = sorted(os.listdir(root_dir))
        # self.data = list(zip(self.high_res, [index] * len(self.high_res)))

    def __len__(self):
        return len(self.high_res)

    def __getitem__(self, index):
        img_file = self.high_res

        image = np.array(Image.open(os.path.join(self.root_dir, img_file[index])))
        image = config.both_transforms(image=image)["image"]
        high_res = config.highres_transform(image=image)["image"]
        low_res = config.lowres_transform(image=image)["image"]
        return low_res, high_res

def test():
    dataset = MyImageFolder(root_dir="new_data/")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()