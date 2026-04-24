# ============ model/CIBHash.py - PaddlePaddle 版本 ============
"""
教师模型 CIBHash - PaddlePaddle 版本
用于蒸馏训练中的教师模型
支持 efficientnet_b0 / resnet 系列 backbone
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import argparse
import copy

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'os' not in dir() else '.')

from utils.data_paddle import LabeledData
from utils.evaluation_paddle import compress, distill_compress, calculate_top_map
import logging
import os


class Base_Model(nn.Layer):
    """教师模型基类 - PaddlePaddle 版本"""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.load_data()
        if self.hparams.rkd:
            self.loss_type = 'rkd'
        elif self.hparams.pkt:
            self.loss_type = 'pkt'
        elif self.hparams.sp:
            self.loss_type = 'sp'
        elif self.hparams.dr:
            self.loss_type = 'dr'
        elif self.hparams.nst:
            self.loss_type = 'nst'
        elif self.hparams.kl:
            self.loss_type = 'kl'
        elif self.hparams.crd:
            self.loss_type = 'crd'
        elif self.hparams.sskd:
            self.loss_type = 'sskd'
        elif self.hparams.crcd:
            self.loss_type = 'crcd'
        elif self.hparams.packd:
            self.loss_type = 'packd'
        else:
            print("error, please assign at least one regularization loss")
            exit()
        self.load_teacher_model()

    def load_data(self):
        self.data = LabeledData(self.hparams.dataset)

    def get_hparams_grid(self):
        raise NotImplementedError

    def define_parameters(self):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def load_teacher_model(self, ta=False):
        """加载教师模型"""
        device = 'gpu:0' if self.hparams.cuda else 'cpu'
        teacher_path = './checkpoints/' + self.hparams.data_name + '_' + self.hparams.t_model_name + '_bit:' + str(self.hparams.encode_length) + '.pt'
        self.t_model = paddle.load(teacher_path)
        self.t_model.set_state_dict(paddle.load(teacher_path))
        self.t_model.eval()

    def run_training_sessions(self):
        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, self.hparams.s_model_name + '_' + self.hparams.t_model_name + '_' + self.hparams.data_name + '_' + self.loss_type + '_' + str(self.hparams.trail) + '.log')

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        import time
        start = time.time()
        from collections import OrderedDict
        for run_num in range(1, self.hparams.num_runs + 1):
            self.run_training_session(run_num)

        from datetime import timedelta
        logging.info('Time: %s' % str(timedelta(seconds=round(time.time() - start))))
        if self.loss_type in ["crd", "packd"]:
            logging.info("This method don't save, please check the log for best result!")
            return

        model = self.load()
        logging.info('**Test**')

        val_perf, test_perf, distill_val_perf, distill_test_perf = model.run_test()
        logging.info('Val:  {:8.4f}'.format(val_perf))
        logging.info('Test: {:8.4f}'.format(test_perf))

        model = self.load_distill()
        val_perf, test_perf, distill_val_perf, distill_test_perf = model.run_test()
        logging.info('Distill_Val:  {:8.4f}'.format(distill_val_perf))
        logging.info('Distill_Test: {:8.4f}'.format(distill_test_perf))

    def run_training_session(self, run_num):
        self.train()
        if self.hparams.num_runs > 1:
            logging.info('RANDOM RUN: %d/%d' % (run_num, self.hparams.num_runs))
            for hparam, values in self.get_hparams_grid().items():
                assert hasattr(self.hparams, hparam)
                import random
                self.hparams.__dict__[hparam] = random.choice(values)

        import random
        random.seed(self.hparams.seed)
        paddle.seed(self.hparams.seed)

        self.define_parameters()
        if self.hparams.encode_length == 16:
            self.hparams.epochs = max(80, self.hparams.epochs)

        logging.info('hparams: %s' % self.flag_hparams())

        device = 'gpu:0' if self.hparams.cuda else 'cpu'
        self.to(device)

        optimizer = self.configure_optimizers()
        train_loader, val_loader, _, database_loader = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=True, get_test=False)
        best_val_perf = float('-inf')
        best_distill_val_perf = float('-inf')
        bad_epochs = 0

        for epoch in range(1, self.hparams.epochs + 1):
            forward_sum = {}
            num_steps = 0
            for batch_num, batch in enumerate(train_loader):
                optimizer.clear_grad()

                if self.hparams.crd:
                    img_i, img_j, target, index, contrast_idx = batch
                    img_i = img_i.astype('float32')
                    img_j = img_j.astype('float32')
                    forward = self.forward(img_i, img_j, index, contrast_idx)
                elif self.hparams.sskd:
                    img_i, img_j, distill_img, target = batch
                    img_i = img_i.astype('float32')
                    img_j = img_j.astype('float32')
                    forward = self.forward(img_i, img_j, distill_img=distill_img)
                elif self.hparams.packd:
                    img_i, img_j, img, target, index, contrast_idx, mixup_indexes = batch
                    img_i = img_i.astype('float32')
                    img_j = img_j.astype('float32')
                    forward = self.forward(img_i, img_j, img=img, labels=[target, index], contrast_idx=contrast_idx, mixup_indexes=mixup_indexes)
                else:
                    imgi, imgj, idxs, _ = batch
                    forward = self.forward(imgi, imgj)

                for key in forward:
                    if key in forward_sum:
                        forward_sum[key] += forward[key].numpy().item() if hasattr(forward[key], 'numpy') else forward[key]
                    else:
                        forward_sum[key] = forward[key].numpy().item() if hasattr(forward[key], 'numpy') else forward[key]
                num_steps += 1

                import math
                if math.isnan(forward_sum['loss']):
                    logging.info('Stopping epoch because loss is NaN')
                    break

                forward['loss'].backward()
                optimizer.step()

            import math
            if math.isnan(forward_sum['loss']):
                logging.info('Stopping training session because loss is NaN')
                break

            logging.info('End of epoch {:3d}'.format(epoch))
            logging.info(' '.join([' | {:s} {:8.4f}'.format(
                key, forward_sum[key] / num_steps)
                                    for key in forward_sum]))

            if epoch % self.hparams.validate_frequency == 0:
                print('evaluating...')
                val_perf, distill_val_perf = self.evaluate(database_loader, val_loader, self.data.topK, device)
                logging.info(' | val perf {:8.4f} | distill val pref {:8.4f}'.format(val_perf, distill_val_perf))

                bad_flag = False
                distll_bad_flag = False
                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    bad_epochs = 0
                    logging.info('\t\t*Best model so far, saving*')
                    logging.info('----New best {:8.4f}, saving'.format(val_perf))
                    if self.loss_type not in ["crd", "packd"]:
                        paddle.save(self.state_dict(), './checkpoints/' + self.hparams.data_name + '_' + self.hparams.s_model_name + '_' + self.hparams.t_model_name + '_' + self.loss_type + '_' + str(self.hparams.trail) + '_bit:' + str(self.hparams.encode_length) + '.pdparams')
                else:
                    bad_flag = True
                    logging.info('\t\tBad epoch %d' % bad_epochs)

                if distill_val_perf > best_distill_val_perf:
                    best_distill_val_perf = distill_val_perf
                    bad_epochs = 0
                    logging.info('\t\t*Best distill model so far, saving*')
                    logging.info('----New best distill {:8.4f}, saving'.format(distill_val_perf))
                    if self.loss_type not in ["crd", "packd"]:
                        paddle.save(self.state_dict(), './checkpoints/distill_' + self.hparams.data_name + '_' + self.hparams.s_model_name + '_' + self.hparams.t_model_name + '_' + self.loss_type + '_' + str(self.hparams.trail) + '_bit:' + str(self.hparams.encode_length) + '.pdparams')
                else:
                    distll_bad_flag = True
                    logging.info('\t\tBad epoch of distll %d' % bad_epochs)

                if bad_flag and distll_bad_flag:
                    bad_epochs = bad_epochs + 1
                if bad_epochs > self.hparams.num_bad_epochs:
                    break

    def evaluate(self, database_loader, val_loader, topK, device):
        self.eval()
        with paddle.no_grad():
            retrievalB, retrievalL, queryB, queryL = ours_compress(database_loader, val_loader, self.encode_discrete, device)
            result = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=topK)
            retrievalB, retrievalL, queryB, queryL = ours_distill_compress(database_loader, val_loader, self.encode_discrete, self.t_model.encode_discrete, device)
            distill_result = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=topK)
        self.train()
        return result, distill_result

    def load(self):
        device = 'gpu:0' if self.hparams.cuda else 'cpu'
        load_path = './checkpoints/' + self.hparams.data_name + '_' + self.hparams.s_model_name + '_' + self.hparams.t_model_name + '_' + self.loss_type + '_' + str(self.hparams.trail) + '_bit:' + str(self.hparams.encode_length) + '.pdparams'
        loaded = paddle.load(load_path)
        self.set_state_dict(loaded)
        self.to(device)
        return self

    def load_distill(self):
        device = 'gpu:0' if self.hparams.cuda else 'cpu'
        load_path = './checkpoints/distill_' + self.hparams.data_name + '_' + self.hparams.s_model_name + '_' + self.hparams.t_model_name + '_' + self.loss_type + '_' + str(self.hparams.trail) + '_bit:' + str(self.hparams.encode_length) + '.pdparams'
        loaded = paddle.load(load_path)
        self.set_state_dict(loaded)
        self.to(device)
        return self

    def run_test(self):
        device = 'gpu:0' if self.hparams.cuda else 'cpu'
        _, val_loader, test_loader, database_loader = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=False, get_test=True)

        val_perf, distill_val_perf = self.evaluate(database_loader, val_loader, self.data.topK, device)
        test_perf, distill_test_perf = self.evaluate(database_loader, test_loader, self.data.topK, device)
        return val_perf, test_perf, distill_val_perf, distill_test_perf

    def flag_hparams(self):
        flags = '%s' % (self.hparams.data_name)
        for hparam in vars(self.hparams):
            val = getattr(self.hparams, hparam)
            if str(val) == 'False':
                continue
            elif str(val) == 'True':
                flags += ' --%s' % (hparam)
            elif str(hparam) in {'data_name', 'num_runs', 'num_workers'}:
                continue
            else:
                flags += ' --%s %s' % (hparam, val)
        return flags

    @staticmethod
    def get_general_hparams_grid():
        from collections import OrderedDict
        grid = OrderedDict({
            'seed': list(range(100000)),
            'lr': [0.003, 0.001, 0.0003, 0.0001],
            'batch_size': [64, 128, 256],
        })
        return grid

    @staticmethod
    def get_general_argparser():
        parser = argparse.ArgumentParser()

        parser.add_argument('--data_name', type=str, default='cifar')
        parser.add_argument('--s_model_name', type=str, default='please_choose_a_model')
        parser.add_argument('--a_model_name', type=str, default='please_choose_a_model')
        parser.add_argument('--t_model_name', type=str, default='please_choose_a_model')
        parser.add_argument('--rkd', action='store_true')
        parser.add_argument('--pkt', action='store_true')
        parser.add_argument('--sp', action='store_true')
        parser.add_argument('--kl', action='store_true')
        parser.add_argument('--crd', action='store_true')
        parser.add_argument('--sskd', action='store_true')
        parser.add_argument('--crcd', action='store_true')
        parser.add_argument('--packd', action='store_true')

        parser.add_argument('--train', action='store_true', help='train a model?')
        parser.add_argument('--trail', default=1, type=int)
        parser.add_argument('--ta_trail', default=1, type=int)
        parser.add_argument('-d', '--dataset', default='cifar10', type=str, help='dataset [%(default)s]')
        parser.add_argument("-l", "--encode_length", type=int, default=16, help="Number of bits of the hash code [%(default)d]")
        parser.add_argument("--lr", default=1e-3, type=float, help='initial learning rate [%(default)g]')
        parser.add_argument("--batch_size", default=64, type=int, help='batch size [%(default)d]')
        parser.add_argument("-e", "--epochs", default=60, type=int, help='max number of epochs [%(default)d]')
        parser.add_argument('--cuda', action='store_true', help='use CUDA?')
        parser.add_argument('--num_runs', type=int, default=1, help='num random runs [%(default)d]')
        parser.add_argument('--num_bad_epochs', type=int, default=5, help='num indulged bad epochs [%(default)d]')
        parser.add_argument('--validate_frequency', type=int, default=10, help='validate every [%(default)d] epochs')
        parser.add_argument('--num_workers', type=int, default=2, help='num dataloader workers [%(default)d]')
        parser.add_argument('--seed', type=int, default=8888, help='random seed [%(default)d]')
        parser.add_argument('--device', type=int, default=0, help='device of the gpu')

        return parser


class CIBHash(Base_Model):
    """教师哈希模型 - PaddlePaddle 版本"""

    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.kl_criterion = DistillKL()

    def define_parameters(self):
        device = 'gpu:0' if self.hparams.cuda else 'cpu'

        if self.hparams.s_model_name == 'efficientnet_b0':
            self.efficient_net = paddle.vision.models.efficientnet_b0(pretrained=True)
            print("use efficientnet_b0 as backbone")

        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            for param in self.efficient_net.parameters():
                param.stop_gradient = True
            self.fc = nn.Sequential(
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, self.hparams.encode_length),
            )

        if self.hparams.rkd:
            self.kd_criterion = RKDLoss()
        elif self.hparams.pkt:
            self.kd_criterion = PKT()
        elif self.hparams.sp:
            self.kd_criterion = Similarity()
        elif self.hparams.crd:
            print("using crd distillation method....")
            from model.crd.criterion_paddle import CRDLoss
            self.opt = type('obj', (), {
                'embed_type': 'linear',
                's_dim': self.hparams.encode_length,
                't_dim': self.hparams.encode_length,
                'feat_dim': 128,
                'nce_k': 500,
                'nce_t': 0.05,
                'nce_m': 0.5,
                'n_data': 5000,
            })()
            self.kd_criterion = CRDLoss(self.opt)
        elif self.hparams.sskd:
            self.kd_criterion = DistillSSKD()
        elif self.hparams.crcd:
            from model.crcd.criterion_paddle import CRCDLoss
            self.opt = type('obj', (), {
                'embed_type': 'linear',
                's_dim': self.hparams.encode_length,
                't_dim': self.hparams.encode_length,
                'feat_dim': 128,
                'nce_k': 500,
                'nce_t': 0.05,
                'nce_m': 0.5,
                'n_data': 5000,
            })()
            self.criterion_kd = CRCDLoss(self.opt)
        elif self.hparams.packd:
            from model.packd.packd_paddle import PACKDConLoss
            self.opt = type('obj', (), {
                's_dim': self.hparams.encode_length,
                't_dim': self.hparams.encode_length,
                'feat_dim': 64,
                'nce_k': 500,
                'pos_k': -1,
                'nce_m': 0.5,
                'mixup_num': 0,
                'dataset': self.hparams.dataset,
                'n_data': 5000,
                'ops_eps': 0.1,
                'ops_err_thres': 0.1,
            })()
            self.kd_criterion = PACKDConLoss(self.opt)

        self.criterion = NtXentLoss(self.hparams.batch_size, self.hparams.temperature)

    def forward(self, raw_imgi, raw_imgj, index=None, contrast_idx=None, distill_img=None, img=None, labels=None, mixup_indexes=None):
        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            imgi = self.resnet(raw_imgi)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            imgi = self.efficient_net(raw_imgi)
            imgi = self.fc(imgi)
        prob_i = paddle.nn.functional.sigmoid(imgi)
        z_i = hash_layer(prob_i - 0.5)

        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            imgj = self.resnet(raw_imgj)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            imgj = self.efficient_net(raw_imgj)
            imgj = self.fc(imgj)
        prob_j = paddle.nn.functional.sigmoid(imgj)
        z_j = hash_layer(prob_j - 0.5)

        kl_loss = (self.compute_kl(prob_i, prob_j) + self.compute_kl(prob_j, prob_i)) / 2
        contra_loss = self.criterion(z_i, z_j)

        with paddle.no_grad():
            t_z_i = self.t_model.encode_discrete(raw_imgi)

        if self.hparams.rkd:
            distll_loss = self.hparams.l1_weight * self.kd_criterion(z_i, t_z_i)
        elif self.hparams.pkt:
            distll_loss = self.hparams.l1_weight * self.kd_criterion(z_i, t_z_i)
        elif self.hparams.sp:
            distll_loss = self.hparams.l1_weight * sum(self.kd_criterion(z_i, t_z_i))[0]
        elif self.hparams.crd:
            distll_loss = self.kd_criterion(z_i, t_z_i, index, contrast_idx)[0]
        elif self.hparams.sskd:
            c, h, w = distill_img.shape[-3:]
            input_x = paddle.reshape(distill_img, [-1, c, h, w])
            batch = int(input_x.shape[0] / 4)
            x = self.efficient_net(input_x)
            x = self.fc(x)
            x = paddle.nn.functional.sigmoid(x)
            x = hash_layer(x - 0.5)
            xt = self.t_model.encode_discrete(input_x)
            distll_loss = self.kd_criterion(x, xt, batch)
        elif self.hparams.crcd:
            distll_loss = self.criterion_kd(z_i, t_z_i, index, contrast_idx)
        elif self.hparams.packd:
            mixup_num = 1
            c, h, w = img.shape[-3:]
            input_x = paddle.reshape(img, [-1, c, h, w])
            batch = int(img.shape[0] // mixup_num)
            x = self.efficient_net(input_x)
            x = self.fc(x)
            x = paddle.nn.functional.sigmoid(x)
            x = hash_layer(x - 0.5)
            xt = self.t_model.encode_discrete(input_x)
            distll_loss = self.kd_criterion(x, xt, labels=labels, mask=None, contrast_idx=contrast_idx, mixup_indexes=mixup_indexes)
        elif self.hparams.kl:
            distll_loss = self.hparams.kl_distill_weight * self.kl_criterion(z_i, t_z_i)

        loss = contra_loss + self.hparams.weight * kl_loss + distll_loss
        return {'loss': loss, 'contra_loss': contra_loss, 'kl_loss': kl_loss}

    def encode_discrete(self, x):
        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            x = self.resnet(x)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            x = self.efficient_net(x)
            x = self.fc(x)
        prob = paddle.nn.functional.sigmoid(x)
        z = hash_layer(prob - 0.5)
        return z

    def compute_kl(self, prob, prob_v):
        prob_v = prob_v.detach()
        kl = prob * (paddle.log(prob + 1e-8) - paddle.log(prob_v + 1e-8)) + (1 - prob) * (paddle.log(1 - prob + 1e-8) - paddle.log(1 - prob_v + 1e-8))
        kl = paddle.mean(paddle.sum(kl, axis=1))
        return kl

    def configure_optimizers(self):
        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            return paddle.optimizer.Adam([{'params': self.resnet.fc.parameters()}], learning_rate=self.hparams.lr)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            return paddle.optimizer.Adam([{'params': self.fc.parameters()}], learning_rate=self.hparams.lr)

    def get_hparams_grid(self):
        grid = Base_Model.get_general_hparams_grid()
        grid.update({
            'temperature': [0.2, 0.3, 0.4],
            'weight': [0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
        })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument("-t", "--temperature", default=0.3, type=float, help="Temperature [%(default)d]")
        parser.add_argument('-w', "--weight", default=0.001, type=float, help='weight of I(x,z) [%(default)f]')
        parser.add_argument("--l2_weight", default=1, type=float)
        parser.add_argument("--l1_weight", default=1, type=float)
        parser.add_argument("--kl_distill_weight", default=1, type=float)
        return parser


class hash_sign(paddle.autograd.PyLayer):
    """可微分 sign 函数 - PaddlePaddle 版本"""

    @staticmethod
    def forward(ctx, input):
        return paddle.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def hash_layer(input):
    return hash_sign.apply(input)


class NtXentLoss(nn.Layer):
    """NT-Xent 对比损失 - PaddlePaddle 版本"""

    def __init__(self, batch_size, temperature):
        super().__init__()
        self.temperature = temperature
        self.similarityF = CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = paddle.ones([N, N], dtype='bool')
        mask = paddle.nn.functional.fill_diagonal_(mask, False)
        for i in range(batch_size):
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = paddle.concat([z_i, z_j], axis=0)

        sim = self.similarityF(paddle.unsqueeze(z, 1), paddle.unsqueeze(z, 0)) / self.temperature
        sim_i_j = paddle.diag(sim, batch_size)
        sim_j_i = paddle.diag(sim, -batch_size)

        mask = self.mask_correlated_samples(batch_size)
        positive_samples = paddle.concat([sim_i_j, sim_j_i], axis=0).reshape([N, 1])
        negative_samples = sim[mask].reshape([N, -1])

        labels = paddle.zeros([N], dtype='int64')
        logits = paddle.concat([positive_samples, negative_samples], axis=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class RKDLoss(nn.Layer):
    """Relational Knowledge Distillation - PaddlePaddle 版本"""

    def __init__(self, w_d=25, w_a=50):
        super().__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.reshape([f_s.shape[0], -1])
        teacher = f_t.reshape([f_t.shape[0], -1])

        with paddle.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        with paddle.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, axis=2)
            t_angle = paddle.bmm(norm_td, norm_td.transpose([0, 2, 1])).reshape([-1])

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, axis=2)
        s_angle = paddle.bmm(norm_sd, norm_sd.transpose([0, 2, 1])).reshape([-1])

        loss_a = F.smooth_l1_loss(s_angle, t_angle)
        loss = self.w_d * loss_d + self.w_a * loss_a
        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = paddle.sum(paddle.pow(e, 2), axis=1)
        prod = paddle.mm(e, e.t())
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clip(min=eps)
        if not squared:
            res = paddle.sqrt(res)
        res = res.clone()
        diag = paddle.arange(len(e))
        res[diag, diag] = 0
        return res


class PKT(nn.Layer):
    """Probabilistic Knowledge Transfer - PaddlePaddle 版本"""

    def __init__(self):
        super().__init__()

    def forward(self, f_s, f_t):
        return self.cosine_similarity_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
        output_net_norm = paddle.sqrt(paddle.sum(output_net ** 2, axis=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0

        target_net_norm = paddle.sqrt(paddle.sum(target_net ** 2, axis=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0

        model_similarity = paddle.mm(output_net, output_net.t())
        target_similarity = paddle.mm(target_net, target_net.t())

        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        model_similarity = model_similarity / paddle.sum(model_similarity, axis=1, keepdim=True)
        target_similarity = target_similarity / paddle.sum(target_similarity, axis=1, keepdim=True)

        loss = paddle.mean(target_similarity * paddle.log((target_similarity + eps) / (model_similarity + eps)))
        return loss


class Similarity(nn.Layer):
    """Similarity-Preserving Knowledge Distillation - PaddlePaddle 版本"""

    def __init__(self):
        super().__init__()

    def forward(self, g_s, g_t):
        return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.reshape([bsz, -1])
        f_t = f_t.reshape([bsz, -1])

        G_s = paddle.mm(f_s, f_s.t())
        G_s = F.normalize(G_s, p=2, axis=1)
        G_t = paddle.mm(f_t, f_t.t())
        G_t = F.normalize(G_t, p=2, axis=1)

        G_diff = G_t - G_s
        loss = paddle.sum(G_diff * G_diff) / (bsz * bsz)
        return loss


class DistillKL(nn.Layer):
    """KL 蒸馏损失 - PaddlePaddle 版本"""

    def __init__(self, T=4):
        super().__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, axis=1)
        p_t = F.softmax(y_t / self.T, axis=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss


class DistillSSKD(nn.Layer):
    """SSKD 蒸馏损失 - PaddlePaddle 版本"""

    def __init__(self):
        super().__init__()

    def forward(self, s_feat, t_feat, batch):
        nor_index = (paddle.arange(4 * batch) % 4 == 0).astype('bool')
        aug_index = (paddle.arange(4 * batch) % 4 != 0).astype('bool')

        s_nor_feat = s_feat[nor_index]
        s_aug_feat = s_feat[aug_index]
        s_nor_feat = s_nor_feat.unsqueeze(2).expand([-1, -1, 3 * batch]).transpose([0, 2, 1])
        s_aug_feat = s_aug_feat.unsqueeze(2).expand([-1, -1, 1 * batch])

        s_simi = F.cosine_similarity(s_aug_feat, s_nor_feat, axis=1)

        t_nor_feat = t_feat[nor_index]
        t_aug_feat = t_feat[aug_index]
        t_nor_feat = t_nor_feat.unsqueeze(2).expand([-1, -1, 3 * batch]).transpose([0, 2, 1])
        t_aug_feat = t_aug_feat.unsqueeze(2).expand([-1, -1, 1 * batch])

        t_simi = F.cosine_similarity(t_aug_feat, t_nor_feat, axis=1)
        t_simi = t_simi.detach()

        aug_target = paddle.arange(batch).unsqueeze(1).expand([-1, 3]).reshape([-1]).astype('int64')
        rank = paddle.argsort(t_simi, axis=1, descending=True)
        rank = paddle.argmax((rank == aug_target.unsqueeze(1)).astype('int64'), axis=1)
        index = paddle.argsort(rank)
        tmp = paddle.nonzero(rank).flatten()
        wrong_num = tmp.shape[0]
        correct_num = 3 * batch - wrong_num
        wrong_keep = int(wrong_num * 1.0)
        index = index[:correct_num + wrong_keep]
        distill_index_ss = paddle.sort(index)[0]

        log_simi = F.log_softmax(s_simi / 0.5, axis=1)
        simi_knowledge = F.softmax(t_simi / 0.5, axis=1)
        distll_loss = F.kl_div(log_simi[distill_index_ss], simi_knowledge[distill_index_ss], reduction='batchmean') * 0.5 * 0.5
        return distll_loss


class CosineSimilarity(nn.Layer):
    """余弦相似度 - PaddlePaddle 版本"""

    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        x1_norm = paddle.sqrt(paddle.sum(x1 * x1, axis=self.dim, keepdim=True) + self.eps)
        x2_norm = paddle.sqrt(paddle.sum(x2 * x2, axis=self.dim, keepdim=True) + self.eps)
        x1 = x1 / (x1_norm + self.eps)
        x2 = x2 / (x2_norm + self.eps)
        if self.dim == 1:
            return paddle.sum(x1 * x2, axis=self.dim)
        elif self.dim == 2:
            return paddle.sum(x1 * x2, axis=self.dim)
        return x1 * x2


def ours_compress(train, test, encode_discrete, device):
    """旧版压缩函数"""
    retrievalB = []
    retrievalL = []
    for batch_step, (data, _, target) in enumerate(train):
        code = encode_discrete(data)
        retrievalB.extend(code.numpy())
        retrievalL.extend(target)

    queryB = []
    queryL = []
    for batch_step, (data, _, target) in enumerate(test):
        code = encode_discrete(data)
        queryB.extend(code.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)
    queryB = np.array(queryB)
    queryL = np.stack(queryL)
    return retrievalB, retrievalL, queryB, queryL


def ours_distill_compress(train, test, s_encode_discrete, t_encode_discrete, device):
    """旧版蒸馏压缩函数"""
    retrievalB = []
    retrievalL = []
    for batch_step, (data, _, target) in enumerate(train):
        code = t_encode_discrete(data)
        retrievalB.extend(code.numpy())
        retrievalL.extend(target)

    queryB = []
    queryL = []
    for batch_step, (data, _, target) in enumerate(test):
        code = s_encode_discrete(data)
        queryB.extend(code.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)
    queryB = np.array(queryB)
    queryL = np.stack(queryL)
    return retrievalB, retrievalL, queryB, queryL
