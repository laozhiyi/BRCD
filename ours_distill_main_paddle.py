# ============ ours_distill_main_paddle.py - PaddlePaddle 蒸馏训练入口 ============
"""
PaddlePaddle 蒸馏训练主入口
使用方法: python ours_distill_main_paddle.py --dataset scid --s_model_name mobilenet_v2 --t_model_name vit_b_16 ...
"""

import argparse
import paddle
import random
import logging
import psutil
import os
import sys
sys.path.insert(0, '.')

from model.ours_distill_CIBHash_paddle import CIBHash as StudentCIBHash


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info('Starting distill training script (PaddlePaddle version)')

    try:
        argparser = StudentCIBHash.get_model_specific_argparser()
        hparams = argparser.parse_args()
        logging.info(f'Arguments parsed: {hparams}')

        paddle.device.set_device('gpu' if hparams.cuda else 'cpu')

        model = StudentCIBHash(hparams)
        logging.info('Student model created successfully')

        logging.info('Starting training sessions')
        model.run_training_sessions()
        logging.info('Training sessions completed successfully')
    except Exception as e:
        logging.error(f'Error in main execution: {str(e)}', exc_info=True)
        raise e
