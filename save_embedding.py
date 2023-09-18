import argparse
import os
from os.path import join, exists
import torch
from PIL import Image
from core.models.dreamsim.model import dreamsim
from torchvision.transforms import transforms
import submitit

class ImagenetEmbedding:

  def __init__(self, args):
    self.imagenet_dir = args.imagenet_dir
    self.split = args.split
    self.destination_dir = args.destination_dir
    self.dreamsim_model_dir = args.dreamsim_model_dir

  def __call__(self, folder_name):

    imagenet_dir = self.imagenet_dir
    split = self.split
    destination_dir = self.destination_dir
    dreamsim_model_dir = self.dreamsim_model_dir

    print(folder_name)

    standard_transform = transforms.Compose([
      transforms.CenterCrop(224),
      transforms.ToTensor(),
    ])
    dreamsim_model, preprocess = dreamsim(
      pretrained=True, device="cuda", cache_dir=dreamsim_model_dir, dreamsim_type="ensemble")
    imagenet_dir = join(imagenet_dir, split)
    if not exists(destination_dir):
      os.mkdir(destination_dir)
    destination_dir = join(destination_dir, split)
    if not exists(destination_dir):
      os.mkdir(destination_dir)

    for root, dirs, files in os.walk(imagenet_dir):
      if root != imagenet_dir: continue
      for class_dir in dirs:
        if class_dir != folder_name: continue
        embed_dir = join(destination_dir, class_dir)
        class_dir = join(imagenet_dir, class_dir)
        if not exists(embed_dir):
          os.mkdir(embed_dir)

        for image_name in os.listdir(class_dir):
          embed_name = image_name.split('.')[0]
          img = Image.open(r"{path}".format(path=join(imagenet_dir, class_dir, image_name))).convert('RGB')
          img = standard_transform(img).unsqueeze(0)
          embed = dreamsim_model.embed(img.cuda())
          path = '{name}.pkl'.format(name=join(embed_dir, embed_name))
          print('saving image in {path}')
          torch.save(embed.squeeze(0).cpu(), path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reading ImageNet images and saving the dreamsim embeddings')
    parser.add_argument("--imagenet_dir", type=str, default='.')
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--destination_dir", type=str, default='.')
    parser.add_argument("--dreamsim_model_dir", type=str, default='.')
    # parser.add_argument("--folder_name", type=str, default='.')
    args = parser.parse_args()

    folder_list = []
    for root, dirs, files in os.walk(args.imagenet_dir):
      if root != join(args.imagenet_dir, args.split): continue
      for class_dir in dirs[4:]:
        folder_list.append(class_dir)

    save_embedding = ImagenetEmbedding(args)

    # cluster = 'slurm' if not args.local else 'local'
    executor = submitit.AutoExecutor(
      folder=f'./save_embbeding', cluster='slurm')
    executor.update_parameters(
      gpus_per_node=1,
      nodes=1,
      tasks_per_node=1,
      cpus_per_task=2,
      slurm_account='dci@v100',
      slurm_partition='gpu_p13',
      slurm_qos='qos_gpu-t3',
      slurm_signal_delay_s=0,
      timeout_min=1000,
    )
    executor.map_array(save_embedding, folder_list)



