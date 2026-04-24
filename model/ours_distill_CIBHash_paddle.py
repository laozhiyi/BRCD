# ============ model/ours_distill_CIBHash_paddle.py - PaddlePaddle 版本 ============
"""
蒸馏学生模型 CIBHash - PaddlePaddle 版本
用于知识蒸馏训练的学生模型
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import argparse
import copy
import os
import sys
sys.path.insert(0, '.')

from utils.data_paddle import LabeledData
from utils.evaluation_paddle import compress, distill_compress, calculate_perceptual_metrics
import logging
from datetime import timedelta
from collections import OrderedDict
import time


class Base_Model(nn.Layer):
    """学生模型基类 - PaddlePaddle 版本"""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.load_data()
        self.loss_type = ''
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
        teacher_path = './checkpoints/' + self.hparams.data_name + '_' + self.hparams.t_model_name + '_bit:' + str(self.hparams.encode_length) + '.pdparams'
        logging.info(f"Loading teacher model from: {teacher_path}")

        teacher_hparams = copy.deepcopy(self.hparams)
        teacher_hparams.model_name = self.hparams.t_model_name

        from model.CIBHash_paddle import CIBHash as TeacherModel
        self.t_model = TeacherModel(teacher_hparams)
        self.t_model.define_parameters()

        try:
            state_dict = paddle.load(teacher_path)
            self.t_model.set_state_dict(state_dict)
        except:
            logging.warning(f"Could not load teacher model from {teacher_path}")
        self.t_model.eval()

    def run_training_sessions(self):
        """运行训练会话"""
        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'ours_' + self.hparams.s_model_name + '_' + self.hparams.t_model_name + '_' + self.hparams.data_name + '_' + self.loss_type + '_' + str(self.hparams.trail) + '.log')

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

        start = time.time()
        paddle.seed(self.hparams.seed)
        import random
        random.seed(self.hparams.seed)
        import numpy as np
        np.random.seed(self.hparams.seed)

        for run_num in range(1, self.hparams.num_runs + 1):
            self.run_training_session(run_num)

        logging.info('Time: %s' % str(timedelta(seconds=round(time.time() - start))))

        logging.info('**Loading Best Student Model (Student vs Student) for Test**')
        model = self.load()
        logging.info('**Test (Student vs Student)**')
        val_perf, test_perf, _, _ = model.run_test()
        logging.info('Val (F1-Student):   {:8.4f}'.format(val_perf))
        logging.info('Test (F1-Student):  {:8.4f}'.format(test_perf))

        logging.info('**Loading Best Distill Model (Student vs Teacher) for Test**')
        model = self.load_distill()
        logging.info('**Test (Student vs Teacher)**')
        _, _, distill_val_perf, distill_test_perf = model.run_test()
        logging.info('Distill_Val (F1-Distill):   {:8.4f}'.format(distill_val_perf))
        logging.info('Distill_Test (F1-Distill):  {:8.4f}'.format(distill_test_perf))

    def run_training_session(self, run_num):
        """运行单次训练会话"""
        self.train()

        if self.hparams.num_runs > 1:
            logging.info('RANDOM RUN: %d/%d' % (run_num, self.hparams.num_runs))
            for hparam, values in self.get_hparams_grid().items():
                assert hasattr(self.hparams, hparam)
                import random
                self.hparams.__dict__[hparam] = random.choice(values)

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

                imgi, imgj, idxs, _ = batch
                imgi = imgi.astype('float32')
                imgj = imgj.astype('float32')

                forward = self.forward(imgi, imgj, idxs, device)

                for key in forward:
                    val = forward[key].numpy().item() if hasattr(forward[key], 'numpy') else forward[key]
                    if key in forward_sum:
                        forward_sum[key] += val
                    else:
                        forward_sum[key] = val
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
            logging.info(' '.join([' | {:s} {:8.4f}'.format(key, forward_sum[key] / num_steps) for key in forward_sum]))

            if epoch % self.hparams.validate_frequency == 0:
                print('evaluating...')
                val_perf, distill_val_perf = self.evaluate(database_loader, val_loader, self.data.topK, device)
                logging.info(' | val perf (F1-Student) {:8.4f} | distill val pref (F1-Distill) {:8.4f}'.format(val_perf, distill_val_perf))

                bad_flag = False
                distll_bad_flag = False

                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    bad_epochs = 0
                    logging.info('\t\t*Best Student model so far, saving*')
                    logging.info('----New best (F1-Student) {:8.4f}, saving'.format(val_perf))
                    paddle.save(self.state_dict(), './checkpoints/ours_' + self.hparams.data_name + '_' + self.hparams.s_model_name + '_' + self.hparams.t_model_name + '_' + self.loss_type + '_' + str(self.hparams.trail) + '_bit_' + str(self.hparams.encode_length) + '.pdparams')
                else:
                    bad_flag = True
                    logging.info('\t\tBad epoch %d' % bad_epochs)

                if distill_val_perf > best_distill_val_perf:
                    best_distill_val_perf = distill_val_perf
                    bad_epochs = 0
                    logging.info('\t\t*Best Distill model so far, saving*')
                    logging.info('----New best (F1-Distill) {:8.4f}, saving'.format(distill_val_perf))
                    paddle.save(self.state_dict(), './checkpoints/ours_distill_' + self.hparams.data_name + '_' + self.hparams.s_model_name + '_' + self.hparams.t_model_name + '_' + self.loss_type + '_' + str(self.hparams.trail) + '_bit_' + str(self.hparams.encode_length) + '.pdparams')
                else:
                    distll_bad_flag = True
                    logging.info('\t\tBad epoch of distll %d' % bad_epochs)

                if bad_flag and distll_bad_flag:
                    bad_epochs = bad_epochs + 1
                if bad_epochs > self.hparams.num_bad_epochs:
                    break

    def evaluate(self, database_loader, val_loader, topK, device):
        """评估模型"""
        self.eval()
        with paddle.no_grad():
            logging.info('Calling compress (Student vs Student)...')
            try:
                retrievalB, retrievalL, retrieval_fnames, queryB, queryL, query_fnames = compress(
                    database_loader, val_loader, self.encode_discrete, device
                )
            except Exception as e:
                logging.error(f'Error in compress function: {e}', exc_info=True)
                return 0.0, 0.0

            logging.info('Calling distill_compress (Student vs Teacher)...')
            try:
                distill_retrievalB, _, _, distill_queryB, _, _ = distill_compress(
                    database_loader, val_loader, self.encode_discrete, self.t_model.encode_discrete, device
                )
            except Exception as e:
                logging.error(f'Error in distill_compress function: {e}', exc_info=True)
                return 0.0, 0.0

            logging.info('Calling calculate_perceptual_metrics...')
            gnd_json_path = os.path.join('./data', 'SCID', 'gnd_SCID.json')

            f1_result_student = 0.0
            f1_result_distill = 0.0

            if not os.path.exists(gnd_json_path):
                logging.warning(f"gnd_SCID.json not found at {gnd_json_path}.")
            else:
                try:
                    metrics_student = calculate_perceptual_metrics(
                        qB=queryB,
                        rB=retrievalB,
                        q_fnames=query_fnames,
                        r_fnames=retrieval_fnames,
                        gnd_json_path=gnd_json_path,
                        num_bits=self.hparams.encode_length
                    )
                    f1_result_student = metrics_student.get('Best F1-Score', 0.0)
                    logging.info(f"  | Best F1-Score (Student): {f1_result_student:8.4f}")

                    metrics_distill = calculate_perceptual_metrics(
                        qB=distill_queryB,
                        rB=distill_retrievalB,
                        q_fnames=query_fnames,
                        r_fnames=retrieval_fnames,
                        gnd_json_path=gnd_json_path,
                        num_bits=self.hparams.encode_length
                    )
                    f1_result_distill = metrics_distill.get('Best F1-Score', 0.0)
                    logging.info(f"  | Best F1-Score (Distill): {f1_result_distill:8.4f}")
                except Exception as e:
                    logging.error(f'Error in calculate_perceptual_metrics: {e}')

        self.train()
        return f1_result_student, f1_result_distill

    def load(self):
        """加载学生模型"""
        load_path = './checkpoints/ours_' + self.hparams.data_name + '_' + self.hparams.s_model_name + '_' + self.hparams.t_model_name + '_' + self.loss_type + '_' + str(self.hparams.trail) + '_bit_' + str(self.hparams.encode_length) + '.pdparams'
        logging.info(f'Loading model: {load_path}')
        try:
            state_dict = paddle.load(load_path)
            self.set_state_dict(state_dict)
        except:
            logging.warning(f"Could not load model from {load_path}")
        return self

    def load_distill(self):
        """加载蒸馏模型"""
        load_path = './checkpoints/ours_distill_' + self.hparams.data_name + '_' + self.hparams.s_model_name + '_' + self.hparams.t_model_name + '_' + self.loss_type + '_' + str(self.hparams.trail) + '_bit_' + str(self.hparams.encode_length) + '.pdparams'
        logging.info(f'Loading distill model: {load_path}')
        try:
            state_dict = paddle.load(load_path)
            self.set_state_dict(state_dict)
        except:
            logging.warning(f"Could not load distill model from {load_path}")
        return self

    def run_test(self):
        """运行测试"""
        _, val_loader, test_loader, database_loader = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=False, get_test=True)

        val_perf, distill_val_perf = self.evaluate(database_loader, val_loader, self.data.topK, 'gpu:0' if self.hparams.cuda else 'cpu')
        test_perf, distill_test_perf = self.evaluate(database_loader, test_loader, self.data.topK, 'gpu:0' if self.hparams.cuda else 'cpu')
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
        grid = OrderedDict({
            'seed': list(range(100000)),
            'lr': [0.003, 0.001, 0.0003, 0.0001],
            'batch_size': [64, 128, 256],
        })
        return grid

    @staticmethod
    def get_general_argparser():
        parser = argparse.ArgumentParser()
        parser.add_argument('data_name', type=str, default='cifar')
        parser.add_argument('--s_model_name', type=str, default='please_choose_a_model')
        parser.add_argument('--a_model_name', type=str, default='please_choose_a_model')
        parser.add_argument('--t_model_name', type=str, default='please_choose_a_model')
        parser.add_argument('--train', action='store_true', help='train a model?')
        parser.add_argument('--trail', default=1, type=int)
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
    """学生哈希模型 - PaddlePaddle 版本"""

    def __init__(self, hparams):
        super().__init__(hparams=hparams)

        if self.hparams.s_model_name == 'mobilenet_v2':
            self.mobilenet_v2 = paddle.vision.models.mobilenet_v2(pretrained=True)
            print("use mobilenet_v2 as backbone")

        if self.hparams.s_model_name == 'resnet18':
            self.resnet = paddle.vision.models.resnet18(pretrained=True)
            print("use resnet18 as backbone")
            block_num = 1

        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            if hasattr(self, 'resnet'):
                block_num = {'resnet18': 1, 'resnet34': 1, 'resnet50': 4, 'resnet101': 4, 'resnet152': 4}.get(self.hparams.s_model_name, 1)
            else:
                if self.hparams.s_model_name == 'resnet34':
                    self.resnet = paddle.vision.models.resnet34(pretrained=True)
                    block_num = 1
                elif self.hparams.s_model_name == 'resnet50':
                    self.resnet = paddle.vision.models.resnet50(pretrained=True)
                    block_num = 4
                elif self.hparams.s_model_name == 'resnet101':
                    self.resnet = paddle.vision.models.resnet101(pretrained=True)
                    block_num = 4
                elif self.hparams.s_model_name == 'resnet152':
                    self.resnet = paddle.vision.models.resnet152(pretrained=True)
                    block_num = 4

        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            if hasattr(self, 'resnet'):
                for param in self.resnet.parameters():
                    param.stop_gradient = True
                if block_num == 1:
                    self.resnet.fc = nn.Linear(512, self.hparams.encode_length)
                else:
                    self.resnet.fc = nn.Linear(512 * block_num, self.hparams.encode_length)

        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            model_map = {
                'efficientnet_b0': paddle.vision.models.efficientnet_b0,
                'efficientnet_b1': paddle.vision.models.efficientnet_b1,
                'efficientnet_b2': paddle.vision.models.efficientnet_b2,
                'efficientnet_b3': paddle.vision.models.efficientnet_b3,
                'efficientnet_b4': paddle.vision.models.efficientnet_b4,
                'efficientnet_b5': paddle.vision.models.efficientnet_b5,
                'efficientnet_b6': paddle.vision.models.efficientnet_b6,
                'efficientnet_b7': paddle.vision.models.efficientnet_b7,
            }
            self.efficient_net = model_map.get(self.hparams.s_model_name, paddle.vision.models.efficientnet_b0)(pretrained=True)
            for param in self.efficient_net.parameters():
                param.stop_gradient = True
            self.fc = nn.Sequential(
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, self.hparams.encode_length),
            )

        if self.hparams.s_model_name in ('mobilenet_v2',):
            for param in self.mobilenet_v2.parameters():
                param.stop_gradient = True
            self.fc = nn.Sequential(
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, self.hparams.encode_length),
            )

        self.criterion = NtXentLoss(self.hparams.batch_size, self.hparams.temperature)
        self.criterion_distill = BRCDLoss(self.hparams.batch_size, self.hparams.temperature)

    def forward(self, raw_imgi, raw_imgj, idxs, device=None):
        """前向传播"""
        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            imgi = self.resnet(raw_imgi)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            imgi = self.efficient_net(raw_imgi)
            imgi = self.fc(imgi)
        if self.hparams.s_model_name in ('mobilenet_v2',):
            imgi = self.mobilenet_v2(raw_imgi)
            imgi = self.fc(imgi)
        prob_i = F.sigmoid(imgi)
        z_i = hash_layer(prob_i - 0.5)

        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            imgj = self.resnet(raw_imgj)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            imgj = self.efficient_net(raw_imgj)
            imgj = self.fc(imgj)
        if self.hparams.s_model_name in ('mobilenet_v2',):
            imgj = self.mobilenet_v2(raw_imgj)
            imgj = self.fc(imgj)
        prob_j = F.sigmoid(imgj)
        z_j = hash_layer(prob_j - 0.5)

        kl_loss = (self.compute_kl(prob_i, prob_j) + self.compute_kl(prob_j, prob_i)) / 2
        contra_loss = self.criterion(z_i, z_j)

        with paddle.no_grad():
            t_z_i = self.t_model.encode_discrete(raw_imgi)
            t_z_j = self.t_model.encode_discrete(raw_imgj)

        distll_loss = self.criterion_distill(z_i, t_z_i, t_z_j, self.hparams.alpha)

        loss = contra_loss + self.hparams.weight * kl_loss + distll_loss
        return {'loss': loss, 'contra_loss': contra_loss, 'kl_loss': kl_loss}

    def encode_discrete(self, x):
        """离散编码"""
        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            x = self.resnet(x)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            x = self.efficient_net(x)
            x = self.fc(x)
        if self.hparams.s_model_name in ('mobilenet_v2',):
            x = self.mobilenet_v2(x)
            x = self.fc(x)
        prob = F.sigmoid(x)
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
        if self.hparams.s_model_name in ('mobilenet_v2',):
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
        parser.add_argument("--alpha", default=0.5, type=float)
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


class BRCDLoss(nn.Layer):
    """BRCD 蒸馏损失 - PaddlePaddle 版本"""

    def __init__(self, batch_size, temperature):
        super().__init__()
        self.temperature = temperature
        self.similarityF = CosineSimilarity(dim=2)
        self.similarityA = CosineSimilarity(dim=1)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = paddle.ones([N, N], dtype='bool')
        mask = paddle.nn.functional.fill_diagonal_(mask, False)
        for i in range(batch_size):
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False
        return mask

    def forward(self, z_i, t_z_i, t_z_j, alpha, device=None):
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = paddle.concat([z_i, t_z_j], axis=0)
        sim = self.similarityF(paddle.unsqueeze(z, 1), paddle.unsqueeze(z, 0)) / self.temperature

        sim_anchor = self.similarityA(z_i, t_z_i) / self.temperature
        sim_anchor = paddle.concat([sim_anchor, sim_anchor], axis=0)

        sim_i_j = paddle.diag(sim, batch_size)
        sim_j_i = paddle.diag(sim, -batch_size)

        mask = self.mask_correlated_samples(batch_size)

        positive_from_aug = paddle.concat([sim_i_j, sim_j_i], axis=0)
        positive_samples = alpha * sim_anchor + (1 - alpha) * positive_from_aug
        positive_samples = positive_samples.reshape([N, 1])

        negative_samples = sim[mask].reshape([N, -1])

        labels = paddle.zeros([N], dtype='int64')
        logits = paddle.concat([positive_samples, negative_samples], axis=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


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
