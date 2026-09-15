import pycuda.driver as cuda
import numpy as np
import cv2
from datetime import datetime
import os, sys, logging, json
from config.detection_config import MODEL_CONFIG, SAVE_CONFIG, LOG_CONFIG
from myutils.logger import setup_logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))))
logger = setup_logger('detection', LOG_CONFIG['base_dir'], LOG_CONFIG['sub_dirs']['detection'])

cuda_initialized = False
device = None
context = None

def initialize_cuda():
    global cuda_initialized, device, context
    if not cuda_initialized:
        cuda.init()
        device = cuda.Device(0)
        context = device.make_context()
        cuda_initialized = True

def release_cuda():
    global cuda_initialized, context, device
    if cuda_initialized and context:
        context.detach()
        context = None
        device = None
        cuda_initialized = False

initialize_cuda()
print('CUDA initialized:', cuda_initialized)
