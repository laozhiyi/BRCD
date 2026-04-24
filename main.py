# PaddlePaddle 兼容层
from paddle_compat import torch
# import argparse
# import torch
# # import torch.nn as nn
# import random
# import logging
# # from torch.autograd import Variable
# # import pickle
# from model.CIBHash import CIBHash
# import psutil

# count = psutil.cpu_count()
# print(f"the number of logit cpu is{count}")
# p = psutil.Process()
# p.cpu_affinity(list(random.sample(range(1, count), 6)))
# torch.multiprocessing.set_sharing_strategy('file_system')

# if __name__ == '__main__':
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler('training.log'),
#             logging.StreamHandler()
#         ]
#     )
#     logging.info('Starting training script')
    
#     try:
#         argparser = CIBHash.get_model_specific_argparser()
#         hparams = argparser.parse_args()
#         logging.info(f'Arguments parsed: {hparams}')
        
#         torch.cuda.set_device(hparams.device)
#         device = torch.device(f"cuda:{hparams.device}" if torch.cuda.is_available() else "cpu")
#         logging.info(f'Using device: {device}')
        
#         model = CIBHash(hparams)
#         logging.info('Model created successfully')
        
#         logging.info('Starting training sessions')
#         model.run_training_sessions()
#         logging.info('Training sessions completed successfully')
#     except Exception as e:
#         logging.error(f'Error in main execution: {str(e)}')
#         raise e

# # 这两个文件的主要区别在于使用的模型类来源不同，ours_distill_main.py
# # 使用的是蒸馏版本的 CIBHash 实现，而 main.py 使用的是原始版本。其他代码逻辑完全相同。


# import argparse
import torch
# import torch.nn as nn
import random
import logging # <-- [保留] import
# from torch.autograd import Variable
# import pickle
from model.CIBHash import CIBHash
import psutil

count = psutil.cpu_count()
print(f"the number of logit cpu is{count}")
p = psutil.Process()
p.cpu_affinity(list(random.sample(range(1, count), 6)))
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    
    # [修复] 移除了 main.py 中的 logging.basicConfig
    # 只需获取根记录器即可。
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # 确保即使 base_model 失败，日志也能打印
    
    logging.info('Starting training script (main.py)')
    
    try:
        argparser = CIBHash.get_model_specific_argparser()
        hparams = argparser.parse_args()
        logging.info(f'Arguments parsed: {hparams}')
        
        torch.cuda.set_device(hparams.device)
        device = torch.device(f"cuda:{hparams.device}" if torch.cuda.is_available() else "cpu")
        logging.info(f'Using device: {device}')
        
        model = CIBHash(hparams)
        logging.info('Model created successfully')
        
        logging.info('Starting training sessions')
        model.run_training_sessions()
        logging.info('Training sessions completed successfully')
    except Exception as e:
        logging.error(f'Error in main execution: {str(e)}', exc_info=True) # [新增] exc_info
        raise e