import random
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from textwrap import wrap
SEED = 3233
np.random.seed(SEED)
random.seed(SEED)

def display_images(df: pd.DataFrame, amount: int) -> None:
    
    img_names = []
    i = 0
    while i < amount:
        img_name = df['image'].iloc[i]
        if not img_name in img_names:
            img_names.append(img_name)
            img = cv2.cvtColor(cv2.imread(f'./dataset/8K Flickr dataset/Images/{df["image"].iloc[i]}'), cv2.COLOR_BGR2RGB)
            plt.subplot(5, 4, i+1)
            plt.subplots_adjust(hspace=1, wspace=0.3)
            plt.imshow(img)
            plt.title("\n".join(wrap(df['caption'].iloc[i], 20)))
            plt.axis("off")
        i += 1
    
    plt.show()

def group_captions_images(df: pd.DataFrame) -> pd.DataFrame:

    image_names = df['image'].unique()
    names = []
    caps = []
    for name in image_names:
        captions = df['caption'][df['image'] == name]
        caption = captions.sample(1, ignore_index=True)[0]
        caps.append(caption)
        names.append(name)
    res = pd.DataFrame(zip(names, caps), columns=['image', 'caption'])
    res.to_csv(r'./dataset/8K Flickr dataset/captions fixed.csv', index=False)
    return res

if __name__ == '__main__':
    df = pd.read_csv(r'./dataset/8K Flickr dataset/captions.csv')
    #display_images(df.sample(50), 20)
    group_captions_images(df)