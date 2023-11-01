import argparse
import requests
import pandas as pd
from dreamsim import dreamsim
import torch

from lipsim.core.data.yfcc_dataset import yfcc_transform

if __name__ == '__main__':

    dreamsim_model, preprocess = dreamsim(
        pretrained=True, device="cuda", cache_dir='../checkpoints', dreamsim_type="ensemble")
    parser = argparse.ArgumentParser(description='Reading the urls from yfcc15m.numbers and downloading the images')
    parser.add_argument("--img_url_dir", type=str,
                        default='/Users/saraghazanfari/PycharmProjects/lipsim/yfcc15m.csv')
    parser.add_argument("--save_dir", type=str, default='.')

    args = parser.parse_args()
    with open(args.img_url_dir, 'r') as file:
        csvreader = pd.read_csv(file, encoding='unicode_escape')
        for idx, img_url in enumerate(csvreader):
            print(img_url)
            img_data = requests.get(img_url).content
            with open(f'{args.save_dir}/{idx}.jpg', 'wb') as handler:
                handler.write(img_data)
                embedding = dreamsim_model(yfcc_transform(img_data).cuda())
                path = f'{idx}.pkl'
                torch.save(embedding.squeeze(0).cpu(), path)
            break
