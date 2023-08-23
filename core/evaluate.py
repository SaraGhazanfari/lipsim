import glob
import logging
from os.path import join
from tqdm import tqdm
from core import utils
from core.data.readers import ImagenetDataset, readers_config, NightDataset
from core.models.model import NormalizedModel, LipschitzNetwork
from dreamsim import dreamsim
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import sys
from advertorch.attacks import LinfPGDAttack, L2PGDAttack

def get_2afc_score(d0s, d1s, targets):
    d0s = torch.cat(d0s, dim=0)
    d1s = torch.cat(d1s, dim=0)
    targets = torch.cat(targets, dim=0)
    scores = (d0s < d1s) * (1.0 - targets) + (d1s < d0s) * targets + (d1s == d0s) * 0.5
    twoafc_score = torch.mean(scores, dim=0)
    return twoafc_score


class Evaluator:
    '''Evaluate a Pytorch Model.'''

    def __init__(self, config):
        self.config = config
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.dino_model, _ = dreamsim(pretrained=True, dreamsim_type='dino_vitb16', cache_dir='./checkpoints')
        self.criterion = utils.get_loss(self.config)

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

        # load reader
        Reader = readers_config[self.config.dataset]
        self.reader = Reader(config=self.config, batch_size=self.batch_size, is_training=False)
        self.config.means = self.reader.means

        # load model
        self.model = LipschitzNetwork(self.config, self.reader.n_classes)
        self.model = NormalizedModel(self.model, self.reader.means, self.reader.stds)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        self.load_ckpt()
        if self.config.mode == 'certified':
            for eps in [36, 72, 108, 255]:
                if self.config.last_layer == 'lln':
                    self.eval_certified_lln(eps)
                else:
                    self.eval_certified(eps)
        elif self.config.mode == 'attack':
            self.eval_attack()
        elif self.config.mode == 'eval':
            self.vanilla_eval()
        elif self.config.mode == 'dreamsim':
            self.dreamsim_eval()

        logging.info('Done with batched inference.')

    @torch.no_grad()
    def eval_certified(self, eps):
        eps_float = eps / 255
        self.model.eval()
        running_accuracy = 0
        running_certified = 0
        running_inputs = 0
        lip_cst = 1.
        data_loader, _ = self.reader.load_dataset()
        for batch_n, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.model(inputs)
            predicted = outputs.argmax(axis=1)
            correct = outputs.max(1)[1] == labels
            margins = torch.sort(outputs, 1)[0]
            certified = (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * lip_cst * eps_float
            running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
            running_certified += torch.sum(correct & certified).item()
            running_inputs += inputs.size(0)
        self.model.train()
        accuracy = running_accuracy / running_inputs
        certified = running_certified / running_inputs
        self.message.add('eps', [eps, 255], format='.0f')
        self.message.add('eps', eps_float, format='.5f')
        self.message.add('accuracy', accuracy, format='.5f')
        self.message.add('certified accuracy', certified, format='.5f')
        logging.info(self.message.get_message())
        return accuracy, certified

    @torch.no_grad()
    def eval_certified_lln(self, eps):
        eps_float = eps / 255
        self.model.eval()
        running_accuracy = 0
        running_certified = 0
        running_inputs = 0
        lip_cst = 1.
        data_loader, _ = self.reader.load_dataset()
        last_weight = self.model.module.model.last_last.weight
        normalized_weight = F.normalize(last_weight, p=2, dim=1)
        for batch_n, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.model(inputs)
            predicted = outputs.argmax(axis=1)
            correct = outputs.max(1)[1] == labels
            margins, indices = torch.sort(outputs, 1)
            margins = margins[:, -1][:, None] - margins[:, 0:-1]
            for idx in range(margins.shape[0]):
                margins[idx] /= torch.norm(
                    normalized_weight[indices[idx, -1]] - normalized_weight[indices[idx, 0:-1]], dim=1, p=2)
            margins, _ = torch.sort(margins, 1)
            certified = margins[:, 0] > eps_float * lip_cst
            running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
            running_certified += torch.sum(correct & certified).item()
            running_inputs += inputs.size(0)
        accuracy = running_accuracy / running_inputs
        certified = running_certified / running_inputs
        self.message.add('eps', [eps, 255], format='.0f')
        self.message.add('eps', eps_float, format='.5f')
        self.message.add('accuracy', accuracy, format='.5f')
        self.message.add('certified accuracy', certified, format='.5f')
        logging.info(self.message.get_message())
        return accuracy, certified

    def eval_attack(self):
        '''Run evaluation under attack.'''

        attack = utils.get_attack_eval(
            self.model,
            self.config.attack,
            self.config.eps / 255,
            self.batch_size)

        running_accuracy = 0
        running_accuracy_adv = 0
        running_inputs = 0
        data_loader, _ = self.reader.load_dataset()
        for batch_n, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs_adv = attack.perturb(inputs, labels)

            outputs = self.model(inputs)
            outputs_adv = self.model(inputs_adv)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_adv = torch.max(outputs_adv.data, 1)

            running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
            running_accuracy_adv += predicted_adv.eq(labels.data).cpu().sum().numpy()
            running_inputs += inputs.size(0)

        accuracy = running_accuracy / running_inputs
        accuracy_adv = running_accuracy_adv / running_inputs
        self.message.add(f'attack: {self.config.attack} - eps', self.config.eps, format='.0f')
        self.message.add('Accuracy', accuracy, format='.5f')
        self.message.add('Accuracy attack', accuracy_adv, format='.5f')
        logging.info(self.message.get_message())

    def vanilla_eval(self):
        loss = 0
        data_loader, _ = self.reader.load_dataset()
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
        dino_outputs = self.dino_model.embed(inputs)
        batch_loss = self.criterion(outputs, dino_outputs).item()
        return batch_loss

    def dreamsim_eval(self):
        data_loader, dataset_size = NightDataset(config=self.config, batch_size=self.config.batch_size, split='test_imagenet').load_dataset()
        no_imagenet_data_loader, no_imagenet_dataset_size = NightDataset(config=self.config, 
                batch_size=self.config.batch_size, split='test_no_imagenet').load_dataset()
        imagenet_score = self.score_nights_dataset(data_loader)
        logging.info(f"ImageNet 2AFC score: {str(imagenet_score)}")
        no_imagenet_score = self.score_nights_dataset(no_imagenet_data_loader)
        logging.info(f"No ImageNet 2AFC score: {str(no_imagenet_score)}")
        overall_score = (imagenet_score * dataset_size +
                                    no_imagenet_score * no_imagenet_dataset_size) / (dataset_size + no_imagenet_dataset_size)
        logging.info(f"Overall 2AFC score: {str(overall_score)}")

    def score_nights_dataset(self, test_loader):
        logging.info("Evaluating NIGHTS dataset.")
        d0s = []
        d1s = []
        targets = []
        #with torch.no_grad()
        for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(test_loader), total=len(test_loader)):
            img_ref, img_left, img_right, target = img_ref.cuda(), img_left.cuda(), \
                img_right.cuda(), target.cuda()
            dist_0, dist_1, target = self.one_step_night_eval(img_ref, img_left, img_right, target)
            d0s.append(dist_0)
            d1s.append(dist_1)
            targets.append(target)
            twoafc_score = get_2afc_score(d0s, d1s, targets)
            logging.info('batch_num: {batch_n}, 2afc: {twoafc}'.format(batch_n=i+1, twoafc=twoafc_score))

        twoafc_score = get_2afc_score(d0s, d1s, targets)
        return twoafc_score


    def get_dreamsim_dist(self, img_ref, img_left, img_right):
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        embed_ref = self.model(img_ref).detach()
        embed_x0 = self.model(img_left).detach()
        embed_x1 = self.model(img_right).detach()
        dist_0 = 1 - cos_sim(embed_ref, embed_x0)
        dist_1 = 1 - cos_sim(embed_ref, embed_x1)
        return dist_0, dist_1

    def generate_attack(self, img_ref, p=2):
        linf_eps = 0.03
        l2_eps = 1.0
        if p==2:
            adversary = L2PGDAttack(self.model, loss_fn=nn.MSELoss(), eps=l2_eps, nb_iter=200, rand_init=True,
                            targeted=False, eps_iter=0.01, clip_min=0.0, clip_max=1.0)
        else:
            adversary = LinfPGDAttack(self.model, loss_fn=nn.MSELoss(), eps=linf_eps, nb_iter=50,
                eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False)
        img_ref_adv = adversary(img_ref, self.model(img_ref))
        
        return img_ref_adv

    def one_step_night_eval(self, img_ref, img_left, img_right, target):
        img_ref = self.generate_attack(img_ref)
        dist_0, dist_1 = self.get_dreamsim_dist(img_ref, img_left, img_right)

        if len(dist_0.shape) < 1:
            dist_0 = dist_0.unsqueeze(0)
            dist_1 = dist_1.unsqueeze(0)
        dist_0 = dist_0.unsqueeze(1)
        dist_1 = dist_1.unsqueeze(1)
        target = target.unsqueeze(1)
        return dist_0, dist_1, target

















