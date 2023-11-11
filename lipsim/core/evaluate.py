import glob
import logging
import os

from os.path import join
import numpy as np
from dreamsim.model import download_weights, dreamsim
from matplotlib import pyplot as plt
from tqdm import tqdm

from lipsim.core.attack.general_attack import GeneralAttack
from lipsim.core.data.bapps_dataset import BAPPSDataset
from lipsim.core.data.night_dataset import NightDataset
from lipsim.core.eval_knn import KNNEval

from lipsim.core.models.l2_lip.model import L2LipschitzNetwork, NormalizedModel, PerceptualMetric
from lipsim.core import utils

from lipsim.core.data.readers import readers_config, N_CLASSES
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import sys


def get_2afc_score(d0s, d1s, targets):
    d0s = torch.cat(d0s, dim=0)
    d1s = torch.cat(d1s, dim=0)
    targets = torch.cat(targets, dim=0)
    scores = (d0s < d1s) * (1.0 - targets) + (d1s < d0s) * targets + (d1s == d0s) * 0.5
    twoafc_score = torch.mean(scores)
    return twoafc_score


def show_images(index_tensor, inputs, adv_inputs, last_idx):
    for index in range(index_tensor.shape[0]):
        img = inputs[index_tensor[index]]
        adv_image = adv_inputs[index_tensor[index]]
        save_single_image(img, f'original/{last_idx + index}')
        save_single_image(adv_image, f'adv/{last_idx + index}')


def save_single_image(img_ref, img_name):
    plt.imshow(img_ref.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
    plt.axis('off')
    plt.savefig(f'{img_name}.pdf', format="pdf", bbox_inches='tight', pad_inches=0)


class Evaluator:
    '''Evaluate a Pytorch Model.'''

    def __init__(self, config):
        self.config = config
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.criterion = utils.get_loss(self.config)
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.general_attack = GeneralAttack(config=config)

    def load_ckpt(self, ckpt_path=None):
        if ckpt_path is None:
            checkpoints = glob.glob(join(self.config.train_dir, 'checkpoints', 'model.ckpt-*.pth'))
            get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
            ckpt_name = sorted([ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)[-1]
            print(ckpt_name)
            ckpt_path = join(self.config.train_dir, 'checkpoints', ckpt_name)
        checkpoint = torch.load(ckpt_path)
        new_checkpoint = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'alpha' not in k:
                new_checkpoint[k] = v
        self.model.load_state_dict(new_checkpoint)
        epoch = checkpoint['epoch']
        return epoch

    def __call__(self):
        '''Run evaluation of model or eval under attack'''
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        download_weights(cache_dir='./checkpoints', dreamsim_type=self.config.teacher_model_name)
        self.dreamsim_model, _ = dreamsim(pretrained=True, dreamsim_type=self.config.teacher_model_name,
                                          cache_dir='./checkpoints', device=self.device)
        cudnn.benchmark = True

        # create a mesage builder for logging
        self.message = utils.MessageBuilder()
        # Setup logging & log the version.
        utils.setup_logging(self.config, 0)

        ngpus = torch.cuda.device_count()
        if ngpus:
            self.batch_size = self.config.batch_size * ngpus
        else:
            self.batch_size = self.config.batch_size

        # if self.config.mode != 'ssa':
        # load reader
        self.means = (0.0000, 0.0000, 0.0000)
        self.stds = (1.0000, 1.0000, 1.0000)
        self.n_classes = N_CLASSES[self.config.teacher_model_name]

        # load model
        self.model = L2LipschitzNetwork(self.config, self.n_classes)
        self.model = NormalizedModel(self.model, self.means, self.stds)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.perceptual_metric = PerceptualMetric(backbone=self.model)

        self.load_ckpt()

        if self.config.mode == 'lipsim':
            return self.model
        elif self.config.mode == 'eval':
            self.vanilla_eval()
        elif self.config.mode == 'attack':
            self.attack_eval()
        elif self.config.mode == 'ssa':
            self.distance_attack_eval()
        elif self.config.mode == 'certified':
            self.certified_eval()
        elif self.config.mode == 'lpips':
            self.lpips_eval()
        elif self.config.mode == 'knn':
            KNNEval(self.config, self.dreamsim_model).knn_classifier()

        logging.info('Done with batched inference.')

    @torch.no_grad()
    def lpips_eval(self):
        for dataset in ['traditional', 'cnn', 'superres', 'deblur', 'color',
                        'frameinterp']:
            data_loader = BAPPSDataset(data_root=self.config.data_dir, load_size=224,
                                       split='val', dataset=dataset).get_dataloader(
                batch_size=self.config.batch_size)
            twoafc_score = self.get_2afc_score_eval(data_loader)
            logging.info(f"BAPPS 2AFC score: {str(twoafc_score)}")
        return twoafc_score

    @torch.no_grad()
    def certified_eval(self):

        data_loader, dataset_size = NightDataset(config=self.config, batch_size=self.config.batch_size,
                                                 split='test_imagenet').get_dataloader()
        no_imagenet_data_loader, no_imagenet_dataset_size = NightDataset(config=self.config,
                                                                         batch_size=self.config.batch_size,
                                                                         split='test_no_imagenet').get_dataloader()
        imagenet_accuracy, imagenet_certified = self.get_certified_accuracy(data_loader)
        no_imagenet_accuracy, no_imagenet_certified = self.get_certified_accuracy(no_imagenet_data_loader)
        overall_accuracy = (imagenet_accuracy * dataset_size + no_imagenet_accuracy * no_imagenet_dataset_size) / (
                dataset_size + no_imagenet_dataset_size)
        overall_certified = (imagenet_certified * dataset_size + no_imagenet_certified * no_imagenet_dataset_size) / (
                dataset_size + no_imagenet_dataset_size)
        eps_list = np.array([36, 72, 108])
        eps_float_list = eps_list / 255
        for i, eps_float in enumerate(eps_float_list):
            self.message.add('eps', eps_float, format='.5f')
            self.message.add('imagenet accuracy', imagenet_accuracy[i], format='.5f')
            self.message.add('imagenet certified', imagenet_certified[i], format='.5f')

            self.message.add('no imagenet accuracy', no_imagenet_accuracy[i], format='.5f')
            self.message.add('no imagenet certified', no_imagenet_certified[i], format='.5f')

            self.message.add('Overall accuracy', overall_accuracy[i], format='.5f')
            self.message.add('Overall certified', overall_certified[i], format='.5f')
            logging.info(self.message.get_message())

    def get_certified_accuracy(self, data_loader):
        self.model.eval()
        running_accuracy = np.zeros(4)
        running_certified = np.zeros(4)
        running_inputs = 0
        eps_list = np.array([36, 72, 108])
        eps_float_list = eps_list / 255
        for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
            img_ref, img_left, img_right, target = img_ref.cuda(), img_left.cuda(), \
                img_right.cuda(), target.cuda()

            dist_0, dist_1, bound = self.perceptual_metric.get_cosine_score_between_images(img_ref, img_left=img_left,
                                                                                           img_right=img_right,
                                                                                           requires_normalization=True)
            outputs = torch.stack((dist_1, dist_0), dim=1)
            predicted = outputs.argmax(axis=1)
            correct = outputs.max(1)[1] == target
            fy_fi = (outputs.max(dim=1)[0].reshape(-1, 1) - outputs)
            mask = (outputs.max(dim=1)[0].reshape(-1, 1) - outputs) == 0
            fy_fi[mask] = torch.inf
            radius = (fy_fi / bound).min(dim=1)[0]

            for i, eps_float in enumerate(eps_float_list):
                certified = radius > eps_float
                running_certified[i] += torch.sum(correct & certified).item()
                running_accuracy[i] += predicted.eq(target.data).cpu().sum().numpy()
            running_inputs += img_ref.size(0)

        accuracy = running_accuracy / running_inputs
        certified = running_certified / running_inputs

        return accuracy, certified

    def distance_attack_eval(self):
        from utils.visualization_utils import visualize_att_map
        Reader = readers_config[self.config.dataset]
        self.reader = Reader(config=self.config, batch_size=self.batch_size, is_training=False)
        patch_size = int(self.config.teacher_model_name[-2:])
        dreamsim_dist_list = list()
        dataloader, _ = self.reader.get_dataloader()
        for idx, data in tqdm(enumerate(dataloader)):
            if idx * self.batch_size > 5:
                break
            inputs = data[0].to(self.device)
            input_embed = self.model(inputs).detach()
            visualize_att_map(inputs.squeeze(0), img_idx=idx, model=self.dreamsim_model.extractor_list[0].model,
                              device=self.device, patch_size=patch_size,
                              output_dir=os.path.join(self.config.teacher_model_name, 'clean'))
            if self.config.attack:
                adv_inputs = self.general_attack.generate_attack(inputs, img_0=None, img_1=None, target=input_embed,
                                                                 target_model=self.dreamsim_model.embed,
                                                                 is_dist_attack=True)

                adv_input_embed = self.model(adv_inputs).detach()

                cos_dist = 1 - self.cos_sim(input_embed.unsqueeze(0), adv_input_embed.unsqueeze(0))

                dreamsim_dist_list.append(cos_dist)

                visualize_att_map(adv_inputs.squeeze(0), img_idx=idx, model=self.dreamsim_model.extractor_list[0].model,
                                  device=self.device, patch_size=patch_size,
                                  output_dir=os.path.join(self.config.teacher_model_name, 'adv'))
        if self.config.attack:
            torch.save(dreamsim_dist_list,
                       f'{self.config.teacher_model_name}/distance_list_{self.config.attack}_{self.config.eps}')

        logging.info('finished')

    def vanilla_eval(self):
        Reader = readers_config[self.config.dataset]
        self.reader = Reader(config=self.config, batch_size=self.batch_size, is_training=False)
        loss = 0
        data_loader, _ = self.reader.get_dataloader()
        with torch.no_grad():
            for batch_n, data in enumerate(data_loader):
                inputs, _ = data
                batch_loss = self.one_step_eval(inputs)
                loss += batch_loss
                print('batch_num: {batch_n}, loss: {loss}'.format(batch_n=batch_n, loss=batch_loss))
                sys.stdout.flush()

        avg_loss = loss / self.reader.n_test_files

        self.message.add('test loss', avg_loss, format='.5f')
        logging.info(self.message.get_message())

    def one_step_eval(self, inputs):
        inputs = inputs.cuda()

        outputs = self.model(inputs)
        dino_outputs = self.dreamsim_model.embed(inputs)
        batch_loss = self.criterion(outputs, dino_outputs).item()
        return batch_loss

    def attack_eval(self):
        data_loader, dataset_size = NightDataset(config=self.config, batch_size=self.config.batch_size,
                                                 split='test_imagenet').get_dataloader()
        no_imagenet_data_loader, no_imagenet_dataset_size = NightDataset(config=self.config,
                                                                         batch_size=self.config.batch_size,
                                                                         split='test_no_imagenet').get_dataloader()
        print(len(data_loader), len(no_imagenet_data_loader))
        imagenet_score = self.get_2afc_score_eval(data_loader)
        logging.info(f"ImageNet 2AFC score: {str(imagenet_score)}")
        torch.cuda.empty_cache()
        no_imagenet_score = self.get_2afc_score_eval(no_imagenet_data_loader)
        logging.info(f"No ImageNet 2AFC score: {str(no_imagenet_score)}")
        overall_score = (imagenet_score * dataset_size + no_imagenet_score * no_imagenet_dataset_size) / (
                dataset_size + no_imagenet_dataset_size)
        logging.info(f"Overall 2AFC score: {str(overall_score)}")

    def model_wrapper(self, img_left, img_right):
        def metric_model(img):
            if len(img.shape) > 4:
                img_ref, img_0, img_1 = img[:, 0, :, :].squeeze(1), img[:, 1, :, :].squeeze(1), img[:, 2, :, :].squeeze(
                    1)
            else:
                img_ref = img
                img_0, img_1 = img_left, img_right
            dist_0, dist_1, _ = self.perceptual_metric.get_cosine_score_between_images(img_ref, img_0, img_1,
                                                                                       requires_grad=True)
            return torch.stack((dist_1, dist_0), dim=1)

        return metric_model

    def dist_wrapper(self):
        def metric_model(img):
            img_ref, img_0 = img[:, 0, :, :].squeeze(1), img[:, 1, :, :].squeeze(1)
            dist_0, _, _ = self.perceptual_metric.get_cosine_score_between_images(img_ref, img_0, img_0,
                                                                                  requires_grad=True)
            return torch.stack((1 - dist_0, dist_0), dim=1)

        return metric_model

    def dist_2_wrapper(self, img_ref):
        def metric_model(img):
            dist_0, _, _ = self.perceptual_metric.get_cosine_score_between_images(img_ref, img, img, requires_grad=True)
            return torch.stack((1 - dist_0, dist_0), dim=1)

        return metric_model

    def get_2afc_score_eval(self, test_loader):
        logging.info("Evaluating NIGHTS dataset.")
        d0s = []
        d1s = []
        targets = []
        # with torch.no_grad()
        for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(test_loader), total=len(test_loader)):
            img_ref, img_left, img_right, target = img_ref.cuda(), img_left.cuda(), \
                img_right.cuda(), target.cuda()
            dist_0, dist_1, target = self.one_step_2afc_score_eval(img_ref, img_left, img_right, target)
            d0s.append(dist_0)
            d1s.append(dist_1)
            targets.append(target)

        twoafc_score = get_2afc_score(d0s, d1s, targets)
        return twoafc_score

    def one_step_2afc_score_eval(self, img_ref, img_left, img_right, target):
        if self.config.attack:
            img_ref = self.general_attack.generate_attack(img_ref, img_left, img_right, target,
                                                          target_model=self.model_wrapper(img_left, img_right))
        dist_0, dist_1, _ = self.perceptual_metric.get_cosine_score_between_images(img_ref, img_left, img_right)
        if len(dist_0.shape) < 1:
            dist_0 = dist_0.unsqueeze(0)
            dist_1 = dist_1.unsqueeze(0)
        dist_0 = dist_0.unsqueeze(1)
        dist_1 = dist_1.unsqueeze(1)
        target = target.unsqueeze(1)
        return dist_0, dist_1, target
