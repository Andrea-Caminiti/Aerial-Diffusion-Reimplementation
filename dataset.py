import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from util import group_captions_images
import cv2
from os.path import join

class Flickr8KDataset(Dataset):
    
    def __init__(self, path2img: str, path2captions:str) -> None:
        self.path2img = path2img
        self.path2captions = path2captions
        self.captionDF = pd.read_csv(path2captions, sep=',')

    def __len__(self) -> int:
        return len(self.captionDF)
    
    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.captionDF.iloc[index]['image'], self.captionDF.iloc[index]['caption']
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR) # resizing
        img = np.array(img).astype(np.float32) / 255.0 # image of floats  between 0 and 1
        H = 256
        W = 256 
        original = np.float32([[0,0],[H,0],[H,W],[0,W]])
        result = np.float32([[0,W],[H,0],[H,W],[0,2*W]])
        homo_matrix = cv2.getPerspectiveTransform(original, result)
        img = cv2.warpPerspective(img, homo_matrix, (2*W,2*H))
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR) # resizing
        img = img.transpose(2, 0, 1) # (h, w, c) -> (ch, h, w)
        return 2.0 * img - 1.0 # images of floats between -1 and 1
    
    def collate_fn(self, batch: list[dict]) -> np.ndarray:
        images = []
        captions = []
        for elem in batch:
            img = self.preprocess(cv2.cvtColor(cv2.imread(join(self.path2img, elem[0])), cv2.COLOR_BGR2RGB))
            images.append(img)
            captions.append(elem[1])
        images = torch.FloatTensor(np.array(images))
        return images, captions
        
def create_dataloader(batch_size: int) -> DataLoader:
    dset = Flickr8KDataset(r'dataset\8K Flickr dataset\Images', r'dataset\8K Flickr dataset\captions fixed.csv')
    dloader = DataLoader(dset, batch_size=batch_size, collate_fn=dset.collate_fn)
    return dloader


if __name__ == '__main__':
    dset = Flickr8KDataset(r'dataset\8K Flickr dataset\Images', r'dataset\8K Flickr dataset\captions fixed.csv')
    dloader = DataLoader(dset, batch_size=16, collate_fn=dset.collate_fn)
    for batch in dloader:
        print(type(batch))