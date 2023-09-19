import glob
import logging
import time
from os.path import join

from matplotlib import pyplot as plt
from tqdm import tqdm

from autoattack import AutoAttack

from core.models.l2_lip.model import L2LipschitzNetwork, NormalizedModel
from lipsim.core import utils
from core.data import NightDataset, BAPPSDataset
from lipsim.core.models import N_CLASSES

from lipsim.core.data.readers import readers_config
from dreamsim import dreamsim
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import sys
from advertorch.attacks import LinfPGDAttack, L2PGDAttack


def get_2afc_score(d0s, d1s, targets):
    d0s = torch.cat(d0s, dim=0)
    d1s = torch.cat(d1s, dim=0)
    targets = torch.cat(targets, dim=0)
    scores = (d0s < d1s) * (1.0 - targets) + (d1s < d0s) * targets + (d1s == d0s) * 0.5
    twoafc_score = torch.mean(scores)
    return twoafc_score


def show_images(img_ref, img_name):
    plt.imshow(img_ref.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
    plt.axis('off')
    plt.savefig(f'{img_name}.pdf', format="pdf", bbox_inches='tight', pad_inches=0)


class Evaluator:
    '''Evaluate a Pytorch Model.'''

    def __init__(self, config):
        self.config = config
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.dreamsim_model, _ = dreamsim(pretrained=True, dreamsim_type=config.teacher_model_name,
                                          cache_dir='./checkpoints')
        self.criterion = utils.get_loss(self.config)
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def load_ckpt(self, ckpt_path=None):
        if ckpt_path is None:
            checkpoints = glob.glob(join(self.config.train_dir, 'checkpoints', 'model.ckpt-*.pth'))
            get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
            ckpt_name = sorted([ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)[-1]
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

        if self.config.mode != 'ssa':
            # load reader
            self.means = (0.0000, 0.0000, 0.0000)
            self.stds = (1.0000, 1.0000, 1.0000)
            self.n_classes = N_CLASSES[self.config.teacher_model_name]

            # load model
            self.model = L2LipschitzNetwork(self.config, self.n_classes)
            self.model = NormalizedModel(self.model, self.means, self.stds)
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()
            self.load_ckpt()

        if self.config.mode == 'lipsim':
            return self.model
        elif self.config.mode == 'eval':
            self.vanilla_eval()
        elif self.config.mode == 'dreamsim':
            self.dreamsim_eval()
        elif self.config.mode == 'lpips':
            self.lpips_eval()
        elif self.config.mode == 'ssa':
            self.SSA_eval()

        logging.info('Done with batched inference.')

    def SSA_eval(self):
        Reader = readers_config[self.config.dataset]
        self.reader = Reader(config=self.config, batch_size=self.batch_size, is_training=False)
        dreamsim_dist_list, dino_list, open_clip_list, clip_list = list(), list(), list(), list()

        dataset = self.reader.get_dataset()
        start_time = time.time()
        for i in range(len(dataset)):
            if i % 100 != 0:
                continue
            (inputs, _) = dataset[i]
            show_images(inputs, img_name=f'original/{i}')
            inputs = inputs.cuda().unsqueeze(0)
            self.model = self.dreamsim_model.embed
            adv_inputs = self.generate_attack(inputs, img_0=None, img_1=None, target=None)
            show_images(adv_inputs, img_name=f'adv/{i}')
            input_embed = self.dreamsim_model.embed(inputs).detach()
            adv_input_embed = self.dreamsim_model.embed(adv_inputs).detach()
            dreamsim_dist_list.append((1 - self.cos_sim(input_embed, adv_input_embed)).item())
            dino_list.append((1 - self.cos_sim(input_embed[:, :768], adv_input_embed[:, :768])).item())
            open_clip_list.append(
                (1 - self.cos_sim(input_embed[:, 768: 768 + 512], adv_input_embed[:, 768:768 + 512])).item())
            clip_list.append((1 - self.cos_sim(input_embed[:, 768 + 512:], adv_input_embed[:, 768 + 512:])).item())
            end_time = int((time.time() - start_time) / 60)
            print('time: ', end_time, dreamsim_dist_list[-1], dino_list[-1], open_clip_list[-1], clip_list[-1])
            sys.stdout.flush()

        torch.save(dreamsim_dist_list, f='dreamsim_list.pt')
        torch.save(dino_list, f='dino_list.pt')
        torch.save(open_clip_list, f='open_clip_list.pt')
        torch.save(clip_list, f='clip_list.pt')
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

    def dreamsim_eval(self):
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

    def lpips_eval(self):
        for dataset in ['traditional', 'cnn', 'superres', 'deblur', 'color',
                        'frameinterp']:
            data_loader = BAPPSDataset(data_root=self.config.data_dir, load_size=224,
                                       split='val', dataset=dataset).get_dataloader(
                batch_size=self.config.batch_size)
            twoafc_score = self.get_2afc_score_eval(data_loader)
            logging.info(f"BAPPS 2AFC score: {str(twoafc_score)}")
        return twoafc_score

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

    def get_cosine_score_between_images(self, img_ref, img_left, img_right, requires_grad=False):

        embed_ref = self.model(img_ref)
        if not requires_grad:
            embed_ref = embed_ref.detach()
        embed_x0 = self.model(img_left).detach()
        embed_x1 = self.model(img_right).detach()
        dist_0 = 1 - self.cos_sim(embed_ref, embed_x0)
        dist_1 = 1 - self.cos_sim(embed_ref, embed_x1)
        return dist_0, dist_1

    def model_wrapper(self):
        def metric_model(img):
            img_ref, img_0, img_1 = img[:, 0, :, :].squeeze(1), img[:, 1, :, :].squeeze(1), img[:, 2, :, :].squeeze(1)
            dist_0, dist_1 = self.get_cosine_score_between_images(img_ref, img_0, img_1, requires_grad=True)
            return torch.stack((dist_1, dist_0), dim=1)

        return metric_model

    def generate_attack(self, img_ref, img_0, img_1, target):
        attack_method, attack_norm = self.config.attack.split('-')

        if attack_method == 'AA':
            adversary = AutoAttack(self.model_wrapper(), norm=attack_norm, eps=self.config.eps, version='standard',
                                   device='cuda')
            adversary.attacks_to_run = ['apgd-ce']
            img_ref = adversary.run_standard_evaluation(torch.stack((img_ref, img_0, img_1), dim=1), target.long(),
                                                        bs=img_ref.shape[0])
            img_ref = img_ref[:, 0, :, :].squeeze(1)
        elif attack_method == 'PGD':
            if attack_norm == 'L2':
                adversary = L2PGDAttack(self.model, loss_fn=nn.MSELoss(), eps=self.config.eps, nb_iter=200,
                                        rand_init=True,
                                        targeted=False, eps_iter=0.01, clip_min=0.0, clip_max=1.0)

            else:  # attack_type == 'Linf':
                adversary = LinfPGDAttack(self.model, loss_fn=nn.MSELoss(), eps=self.config.eps, nb_iter=50,
                                          eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1., targeted=False)

            img_ref = adversary(img_ref, self.model(img_ref))

        return img_ref

    def one_step_2afc_score_eval(self, img_ref, img_left, img_right, target):
        if self.config.attack:
            img_ref = self.generate_attack(img_ref, img_left, img_right, target)
        dist_0, dist_1 = self.get_cosine_score_between_images(img_ref, img_left, img_right)
        if len(dist_0.shape) < 1:
            dist_0 = dist_0.unsqueeze(0)
            dist_1 = dist_1.unsqueeze(0)
        dist_0 = dist_0.unsqueeze(1)
        dist_1 = dist_1.unsqueeze(1)
        target = target.unsqueeze(1)
        return dist_0, dist_1, target
