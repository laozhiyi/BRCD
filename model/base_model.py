# import math
# import torch
# import random
# import argparse
# import torch.nn as nn
# from datetime import timedelta
# from timeit import default_timer as timer
# import os 

# from utils.data import LabeledData
# from utils.evaluation import compress, calculate_top_map, calculate_perceptual_metrics

# import logging

# class Base_Model(nn.Module):
#     # ... (init, load_data, ... 等不变) ...
#     def __init__(self, hparams):
#         super().__init__()
#         self.hparams = hparams
#         self.load_data()
    
#     def load_data(self):
#         self.data = LabeledData(self.hparams.dataset)
    
#     def get_hparams_grid(self):
#         raise NotImplementedError

#     def define_parameters(self):
#         raise NotImplementedError

#     def configure_optimizers(self):
#         raise NotImplementedError

#     def run_training_sessions(self):
        
#         # --- [日志修复] ---
#         log_dir = './logs'
#         os.makedirs(log_dir, exist_ok=True)
#         log_file = os.path.join(log_dir, self.hparams.data_name + '_' + str(self.hparams.trail) + '.log')

#         # 1. 获取根 logger 并设置级别
#         logger = logging.getLogger()
#         logger.setLevel(logging.INFO)
        
#         # 2. [关键] 清除由 main.py 或其他库可能添加的任何现有 handlers
#         if logger.hasHandlers():
#             logger.handlers.clear()

#         # 3. 创建文件 Handler
#         file_handler = logging.FileHandler(log_file, mode='w') # 'w' = 覆盖
#         file_handler.setLevel(logging.INFO)
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         file_handler.setFormatter(formatter)
        
#         # 4. 创建控制台 Handler
#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(logging.INFO)
#         console_handler.setFormatter(formatter)
        
#         # 5. 添加新的 handlers
#         logger.addHandler(file_handler)
#         logger.addHandler(console_handler)
#         # --- [日志修复结束] ---
        
#         val_perfs = []
#         best_val_perf = float('-inf')
#         start = timer()
#         for run_num in range(1, self.hparams.num_runs + 1):
#             state_dict, val_perf = self.run_training_session(run_num)
#             val_perfs.append(val_perf)
   
#         logging.info('Time: %s' % str(timedelta(seconds=round(timer() - start))))
#         self.load()
#         if self.hparams.num_runs > 1:
#             logging.info('best hparams: ' + self.flag_hparams())
        
#         val_perf, test_perf = self.run_test()
#         logging.info('Final Val (F1-Score):  {:8.4f}'.format(val_perf))
#         logging.info('Final Test (F1-Score): {:8.4f}'.format(test_perf))
    
#     # ... (run_training_session, evaluate, load 等所有其他函数... ) ...
#     # ... (保持我们之前所有的 F1-Score, IndentationError 等修复不变) ...

#     def run_training_session(self, run_num):
#         logging.info('Starting run_training_session')
#         self.train()
        
#         if self.hparams.num_runs > 1:
#             logging.info('RANDOM RUN: %d/%d' % (run_num, self.hparams.num_runs))
#             for hparam, values in self.get_hparams_grid().items():
#                 assert hasattr(self.hparams, hparam)
#                 self.hparams.__dict__[hparam] = random.choice(values)
        
#         random.seed(self.hparams.seed)
#         torch.manual_seed(self.hparams.seed)

#         self.define_parameters()
#         if self.hparams.encode_length == 16:
#             self.hparams.epochs = max(80, self.hparams.epochs)

#         logging.info('hparams: %s' % self.flag_hparams())
        
#         device = torch.device('cuda' if self.hparams.cuda else 'cpu')
#         self.to(device)

#         optimizer = self.configure_optimizers()
        
#         train_loader, val_loader, _, database_loader = self.data.get_loaders(
#             self.hparams.batch_size, self.hparams.num_workers,
#             shuffle_train=True, get_test=False)
        
#         best_val_perf = float('-inf') 
#         best_state_dict = None
#         bad_epochs = 0

#         for epoch in range(1, self.hparams.epochs + 1):
#             forward_sum = {}
#             num_steps = 0
#             for batch_num, batch in enumerate(train_loader):
#                 logging.info(f'Processing epoch {epoch}, batch {batch_num}')
#                 try:
#                     optimizer.zero_grad()

#                     imgi, imgj, idxs, _ = batch
#                     imgi = imgi.to(device) 
#                     imgj = imgj.to(device) 

#                     forward = self.forward(imgi, imgj, device)

#                     for key in forward:
#                         if key in forward_sum:
#                             forward_sum[key] += forward[key]
#                         else:
#                             forward_sum[key] = forward[key]
#                     num_steps += 1

#                     if math.isnan(forward_sum['loss']):
#                         logging.info('Stopping epoch because loss is NaN')
#                         break

#                     forward['loss'].backward()
#                     optimizer.step()
#                 except Exception as e:
#                     logging.error(f'Error in epoch {epoch}, batch {batch_num}: {str(e)}', exc_info=True) # [新增] exc_info
#                     raise e
            
#             if math.isnan(forward_sum['loss']):
#                  logging.info('Stopping training session because loss is NaN')
#                  break
            
#             logging.info('End of epoch {:3d}'.format(epoch))
#             logging.info(' '.join([' | {:s} {:8.4f}'.format(
#                 key, forward_sum[key] / num_steps)
#                                     for key in forward_sum]))

#             if epoch % self.hparams.validate_frequency == 0:
#                 print('evaluating...')
#                 logging.info('Starting evaluation for epoch {:3d}'.format(epoch))
#                 try:
#                     val_perf = self.evaluate(database_loader, val_loader, self.data.topK, device)
#                     logging.info('Evaluation completed for epoch {:3d}'.format(epoch))
#                     logging.info(' | val perf (F1-Score) {:8.4f}'.format(val_perf))
#                 except Exception as e:
#                     logging.error('Error during evaluation: {}'.format(str(e)), exc_info=True) # [新增] exc_info
#                     raise e

#                 if val_perf > best_val_perf:
#                     best_val_perf = val_perf
#                     bad_epochs = 0
#                     logging.info('\t\t*Best model so far (based on F1-Score)*')
#                     logging.info("saving the best model...")
#                     try:
#                         save_path = './checkpoints/' + self.hparams.data_name + '_' + self.hparams.model_name + '_bit:' + str(self.hparams.encode_length) + '_teacher.pt'
#                         torch.save(self, save_path)
#                         logging.info(f"Saved model to {save_path}")
#                     except Exception as e:
#                         logging.error(f'Error saving model: {str(e)}', exc_info=True) # [新增] exc_info
#                         raise e
#                 else:
#                     bad_epochs += 1
#                     logging.info('\t\tBad epoch %d' % bad_epochs)

#                 if bad_epochs > self.hparams.num_bad_epochs:
#                     break

#         logging.info('Training completed')
#         return None, best_val_perf
    
#     def evaluate(self, database_loader, val_loader, topK, device):
#         logging.info('Starting evaluate method')
#         self.eval()
#         with torch.no_grad():
#             logging.info('Calling compress function')
#             try:
#                 retrievalB, retrievalL, retrieval_fnames, queryB, queryL, query_fnames = compress(
#                     database_loader, val_loader, self.encode_discrete, device
#                 )
#                 logging.info('compress function (with filenames) completed successfully')
#             except Exception as e:
#                 logging.error('Error in compress function: {}'.format(str(e)), exc_info=True) # [新增] exc_info
#                 raise e
            
#             logging.info('Calling calculate_top_map function (Semantic Metric)...')
#             try:
#                 map_result = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=topK)
#                 logging.info(f'  | mAP (Semantic): {map_result:8.4f}')
#             except Exception as e:
#                 logging.error('Error in calculate_top_map function: {}'.format(str(e)), exc_info=True) # [新增] exc_info
#                 map_result = 0.0
                
#             logging.info('Calling calculate_perceptual_metrics (Primary Metric)...')
#             gnd_json_path = os.path.join('./data', 'SCID', 'gnd_SCID.json')
            
#             f1_result = 0.0 
#             if not os.path.exists(gnd_json_path):
#                 logging.warning(f"gnd_SCID.json not found at {gnd_json_path}. Skipping perceptual metrics.")
#                 print(f"Warning: gnd_SCID.json not found at {gnd_json_path}. Skipping F1-Score/AHD.")
#             else:
#                 try:
#                     perceptual_metrics = calculate_perceptual_metrics(
#                         qB=queryB, 
#                         rB=retrievalB, 
#                         q_fnames=query_fnames, 
#                         r_fnames=retrieval_fnames, 
#                         gnd_json_path=gnd_json_path, 
#                         num_bits=self.hparams.encode_length
#                     )
#                     f1_result = perceptual_metrics.get('Best F1-Score', 0.0)
#                     logging.info(f"  | Best F1-Score (Perceptual): {f1_result:8.4f}")
#                     logging.info(f"  | AHD-Positive: {perceptual_metrics.get('AHD-Positive (Robustness)', 0.0):8.4f}")
#                     logging.info(f"  | AHD-Negative: {perceptual_metrics.get('AHD-Negative (Discriminability)', 0.0):8.4f}")

#                 except Exception as e:
#                     logging.error('Error in calculate_perceptual_metrics function: {}'.format(str(e)), exc_info=True) # [新增] exc_info
#                     f1_result = 0.0
                
#         self.train()
#         logging.info('evaluate method completed')
        
#         return f1_result

#     def load(self):
#         device = torch.device('cuda' if self.hparams.cuda else 'cpu')
#         load_path = './checkpoints/' + self.hparams.data_name + '_' + self.hparams.model_name + '_bit:' + str(self.hparams.encode_length) + '_teacher.pt'
#         logging.info('load model:' + load_path)
        
#         self_loaded = torch.load(load_path) if self.hparams.cuda \
#                      else torch.load(load_path, map_location=torch.device('cpu'))
#         self.load_state_dict(self_loaded.state_dict())
#         self.to(device)
    
#     def run_test(self):
#         device = torch.device('cuda' if self.hparams.cuda else 'cpu')
#         _, val_loader, test_loader, database_loader = self.data.get_loaders(
#             self.hparams.batch_size, self.hparams.num_workers,
#             shuffle_train=False, get_test=True)
        
#         val_perf = self.evaluate(database_loader, val_loader, self.data.topK, device)
#         test_perf = self.evaluate(database_loader, test_loader, self.data.topK, device)
#         return val_perf, test_perf

#     def flag_hparams(self):
#         flags = '%s' % (self.hparams.data_name)
#         for hparam in vars(self.hparams):
#             val = getattr(self.hparams, hparam)
#             if str(val) == 'False':
#                 continue
#             elif str(val) == 'True':
#                 flags += ' --%s' % (hparam)
#             elif str(hparam) in {'data_name', 'num_runs',
#                                  'num_workers'}:
#                 continue
#             else:
#                 flags += ' --%s %s' % (hparam, val)
#         return flags

#     @staticmethod
#     def get_general_argparser():
#         parser = argparse.ArgumentParser()

#         parser.add_argument('data_name', type=str, default='cifar')
#         parser.add_argument('model_name', type=str, default='please_choose_a_model')
#         parser.add_argument('--train', action='store_true',
#                             help='train a model?')
#         parser.add_argument('--trail', default = 1, type=int)
#         parser.add_argument('-d', '--dataset', default = 'cifar10', type=str,
#                             help='dataset [%(default)s]')
#         parser.add_argument("-l","--encode_length", type = int, default=16,
#                             help = "Number of bits of the hash code [%(default)d]")
#         parser.add_argument("--lr", default = 1e-3, type = float,
#                             help='initial learning rate [%(default)g]')
#         parser.add_argument("--batch_size", default=64,type=int,
#                             help='batch size [%(default)d]')
#         parser.add_argument("-e","--epochs", default=60, type=int,
#                             help='max number of epochs [%(default)d]')
#         parser.add_argument('--cuda', action='store_true',
#                             help='use CUDA?')
#         parser.add_argument('--num_runs', type=int, default=1,
#                             help='num random runs (not random if 1) '
#                             '[%(default)d]')
#         parser.add_argument('--num_bad_epochs', type=int, default=5,
#                             help='num indulged bad epochs [%(default)d]')
#         parser.add_argument('--validate_frequency', type=int, default=5,
#                             help='validate every [%(default)d] epochs')
#         parser.add_argument('--num_workers', type=int, default=0,
#                             help='num dataloader workers [%(default)d]')
#         parser.add_argument('--seed', type=int, default=8888,
#                             help='random seed [%(default)d]')
#         parser.add_argument('--device', type=int, default=0, 
#                             help='device of the gpu')
        
        
#         return parser


import math
import torch
import random
import argparse
import torch.nn as nn
from datetime import timedelta
from timeit import default_timer as timer
import os 

from utils.data import LabeledData
from utils.evaluation import compress, calculate_top_map, calculate_perceptual_metrics

import logging

class Base_Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.load_data()
    
    def load_data(self):
        self.data = LabeledData(self.hparams.dataset)
    
    def get_hparams_grid(self):
        raise NotImplementedError

    def define_parameters(self):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def run_training_sessions(self):
        
        # --- [日志模块] ---
        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, self.hparams.data_name + '_' + str(self.hparams.trail) + '.log')

        # 获取根 logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # 清除既有 handlers 防止重复打印
        if logger.hasHandlers():
            logger.handlers.clear()

        # 设置 File Handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 设置 Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        # --- [日志模块结束] ---
        
        val_perfs = []
        best_val_perf = float('-inf')
        start = timer()
        for run_num in range(1, self.hparams.num_runs + 1):
            state_dict, val_perf = self.run_training_session(run_num)
            val_perfs.append(val_perf)
   
        logging.info('Time: %s' % str(timedelta(seconds=round(timer() - start))))
        self.load()
        if self.hparams.num_runs > 1:
            logging.info('best hparams: ' + self.flag_hparams())
        
        val_perf, test_perf = self.run_test()
        logging.info('Final Val (F1-Score):  {:8.4f}'.format(val_perf))
        logging.info('Final Test (F1-Score): {:8.4f}'.format(test_perf))
    
    def run_training_session(self, run_num):
        logging.info('Starting run_training_session')
        self.train()
        
        if self.hparams.num_runs > 1:
            logging.info('RANDOM RUN: %d/%d' % (run_num, self.hparams.num_runs))
            for hparam, values in self.get_hparams_grid().items():
                assert hasattr(self.hparams, hparam)
                self.hparams.__dict__[hparam] = random.choice(values)
        
        random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)

        self.define_parameters()
        if self.hparams.encode_length == 16:
            self.hparams.epochs = max(80, self.hparams.epochs)

        logging.info('hparams: %s' % self.flag_hparams())
        
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        self.to(device)

        optimizer = self.configure_optimizers()
        
        train_loader, val_loader, _, database_loader = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=True, get_test=False)
        
        best_val_perf = float('-inf') 
        best_state_dict = None
        bad_epochs = 0

        for epoch in range(1, self.hparams.epochs + 1):
            forward_sum = {}
            num_steps = 0
            for batch_num, batch in enumerate(train_loader):
                logging.info(f'Processing epoch {epoch}, batch {batch_num}')
                try:
                    optimizer.zero_grad()

                    imgi, imgj, idxs, _ = batch
                    imgi = imgi.to(device) 
                    imgj = imgj.to(device) 

                    forward = self.forward(imgi, imgj, device)

                    for key in forward:
                        if key in forward_sum:
                            forward_sum[key] += forward[key]
                        else:
                            forward_sum[key] = forward[key]
                    num_steps += 1

                    if math.isnan(forward_sum['loss']):
                        logging.info('Stopping epoch because loss is NaN')
                        break

                    forward['loss'].backward()
                    optimizer.step()
                except Exception as e:
                    logging.error(f'Error in epoch {epoch}, batch {batch_num}: {str(e)}', exc_info=True)
                    raise e
            
            if math.isnan(forward_sum['loss']):
                 logging.info('Stopping training session because loss is NaN')
                 break
            
            logging.info('End of epoch {:3d}'.format(epoch))
            logging.info(' '.join([' | {:s} {:8.4f}'.format(
                key, forward_sum[key] / num_steps)
                                    for key in forward_sum]))

            if epoch % self.hparams.validate_frequency == 0:
                print('evaluating...')
                logging.info('Starting evaluation for epoch {:3d}'.format(epoch))
                try:
                    val_perf = self.evaluate(database_loader, val_loader, self.data.topK, device)
                    logging.info('Evaluation completed for epoch {:3d}'.format(epoch))
                    logging.info(' | val perf (F1-Score) {:8.4f}'.format(val_perf))
                except Exception as e:
                    logging.error('Error during evaluation: {}'.format(str(e)), exc_info=True)
                    raise e

                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    bad_epochs = 0
                    logging.info('\t\t*Best model so far (based on F1-Score)*')
                    logging.info("saving the best model...")
                    try:
                        save_path = './checkpoints/' + self.hparams.data_name + '_' + self.hparams.model_name + '_bit:' + str(self.hparams.encode_length) + '_teacher.pt'
                        torch.save(self, save_path)
                        logging.info(f"Saved model to {save_path}")
                    except Exception as e:
                        logging.error(f'Error saving model: {str(e)}', exc_info=True)
                        raise e
                else:
                    bad_epochs += 1
                    logging.info('\t\tBad epoch %d' % bad_epochs)

                if bad_epochs > self.hparams.num_bad_epochs:
                    break

        logging.info('Training completed')
        return None, best_val_perf
    
    def evaluate(self, database_loader, val_loader, topK, device):
        logging.info('Starting evaluate method')
        self.eval()
        with torch.no_grad():
            logging.info('Calling compress function')
            try:
                # compress 返回的 *_fnames 包含了完整路径，例如 "data/MyScreenDataset/jpg/image_01.png"
                retrievalB, retrievalL, retrieval_fnames, queryB, queryL, query_fnames = compress(
                    database_loader, val_loader, self.encode_discrete, device
                )
                logging.info('compress function (with filenames) completed successfully')
            except Exception as e:
                logging.error('Error in compress function: {}'.format(str(e)), exc_info=True)
                raise e
            
            logging.info('Calling calculate_top_map function (Semantic Metric)...')
            try:
                map_result = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=topK)
                logging.info(f'  | mAP (Semantic): {map_result:8.4f}')
            except Exception as e:
                logging.error('Error in calculate_top_map function: {}'.format(str(e)), exc_info=True)
                map_result = 0.0
                
            logging.info('Calling calculate_perceptual_metrics (Primary Metric)...')
            
            # --- [核心修复 1: 动态路径] ---
            # 你的 dataset name 是 'myscreendataset'，所以这里会自动指向正确位置
            if self.hparams.data_name.lower() == 'myscreendataset':
                gnd_json_path = os.path.join('./data', 'MyScreenDataset', 'gnd.json')
            else:
                gnd_json_path = os.path.join('./data', 'SCID', 'gnd_SCID.json')
            
            f1_result = 0.0 
            if not os.path.exists(gnd_json_path):
                logging.warning(f"gnd json file not found at {gnd_json_path}. Skipping perceptual metrics.")
                print(f"Warning: gnd file not found at {gnd_json_path}.")
            else:
                try:
                    # --- [核心修复 2: 文件名清洗] ---
                    # 你的 gnd.json 里的 Key 是 "image_01" (不带后缀)
                    # 但 query_fnames 是 "path/to/image_01.png"
                    # 下面这两行代码会把路径剥离，只保留无后缀的文件名
                    q_fnames_clean = [os.path.splitext(os.path.basename(f))[0] for f in query_fnames]
                    r_fnames_clean = [os.path.splitext(os.path.basename(f))[0] for f in retrieval_fnames]
                    
                    # 将清洗后的 clean 列表传入计算函数
                    perceptual_metrics = calculate_perceptual_metrics(
                        qB=queryB, 
                        rB=retrievalB, 
                        q_fnames=q_fnames_clean,   # <--- 使用清洗后的
                        r_fnames=r_fnames_clean,   # <--- 使用清洗后的
                        gnd_json_path=gnd_json_path, 
                        num_bits=self.hparams.encode_length
                    )
                    f1_result = perceptual_metrics.get('Best F1-Score', 0.0)
                    logging.info(f"  | Best F1-Score (Perceptual): {f1_result:8.4f}")
                    logging.info(f"  | AHD-Positive: {perceptual_metrics.get('AHD-Positive (Robustness)', 0.0):8.4f}")
                    logging.info(f"  | AHD-Negative: {perceptual_metrics.get('AHD-Negative (Discriminability)', 0.0):8.4f}")

                except Exception as e:
                    logging.error('Error in calculate_perceptual_metrics function: {}'.format(str(e)), exc_info=True)
                    f1_result = 0.0
                
        self.train()
        logging.info('evaluate method completed')
        
        return f1_result

    def load(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        load_path = './checkpoints/' + self.hparams.data_name + '_' + self.hparams.model_name + '_bit:' + str(self.hparams.encode_length) + '_teacher.pt'
        logging.info('load model:' + load_path)
        
        self_loaded = torch.load(load_path) if self.hparams.cuda \
                      else torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(self_loaded.state_dict())
        self.to(device)
    
    def run_test(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        _, val_loader, test_loader, database_loader = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=False, get_test=True)
        
        val_perf = self.evaluate(database_loader, val_loader, self.data.topK, device)
        test_perf = self.evaluate(database_loader, test_loader, self.data.topK, device)
        return val_perf, test_perf

    def flag_hparams(self):
        flags = '%s' % (self.hparams.data_name)
        for hparam in vars(self.hparams):
            val = getattr(self.hparams, hparam)
            if str(val) == 'False':
                continue
            elif str(val) == 'True':
                flags += ' --%s' % (hparam)
            elif str(hparam) in {'data_name', 'num_runs',
                                 'num_workers'}:
                continue
            else:
                flags += ' --%s %s' % (hparam, val)
        return flags

    @staticmethod
    def get_general_argparser():
        parser = argparse.ArgumentParser()

        parser.add_argument('data_name', type=str, default='cifar')
        parser.add_argument('model_name', type=str, default='please_choose_a_model')
        parser.add_argument('--train', action='store_true',
                            help='train a model?')
        parser.add_argument('--trail', default = 1, type=int)
        parser.add_argument('-d', '--dataset', default = 'cifar10', type=str,
                            help='dataset [%(default)s]')
        parser.add_argument("-l","--encode_length", type = int, default=16,
                            help = "Number of bits of the hash code [%(default)d]")
        parser.add_argument("--lr", default = 1e-3, type = float,
                            help='initial learning rate [%(default)g]')
        parser.add_argument("--batch_size", default=64,type=int,
                            help='batch size [%(default)d]')
        parser.add_argument("-e","--epochs", default=60, type=int,
                            help='max number of epochs [%(default)d]')
        parser.add_argument('--cuda', action='store_true',
                            help='use CUDA?')
        parser.add_argument('--num_runs', type=int, default=1,
                            help='num random runs (not random if 1) '
                            '[%(default)d]')
        parser.add_argument('--num_bad_epochs', type=int, default=5,
                            help='num indulged bad epochs [%(default)d]')
        parser.add_argument('--validate_frequency', type=int, default=5,
                            help='validate every [%(default)d] epochs')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='num dataloader workers [%(default)d]')
        parser.add_argument('--seed', type=int, default=8888,
                            help='random seed [%(default)d]')
        parser.add_argument('--device', type=int, default=0, 
                            help='device of the gpu')
        
        return parser