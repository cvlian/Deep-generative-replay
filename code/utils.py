import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from scipy.io import loadmat
import matplotlib.pyplot as plt

abspath = os.getcwd()
datapath = os.path.join(abspath, "dataset")
parampath = os.path.join(abspath, "param")

def get_session(memory_limit=8192):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
        except RuntimeError as e:
            print(e)

def normalize(x):
    return (x-127.5)/127.5
    
def recover(x):
    return ((x+127.5)*127.5).astype(np.uint8)

def load_data(mat_file):
    """
    Helper function for loading a MAT-File
    """
    
    global datapath
    
    src = os.path.join(datapath, mat_file)
    data = loadmat(src)
    
    return data['X'], data['y']

def visualize_acc(histories, maxiter=50):
    """
    Plot accuracy
    """
    
    num_task = len(histories)
    
    fig, axs = plt.subplots(1, num_task, figsize=(12, 4))
    plt.rc('font',family='DejaVu Sans', size=14)
    
    axs[0].set_ylabel("Accuracy", fontsize=18)
    
    for i in range(num_task) :
        for metric, result in histories[i].items():
            
            if metric[:8] != 'accuracy':
                continue
                
            task_name = metric[9:].replace('data', '')
                
            axs[i].plot(list(range(0, maxiter+1)), 
                           [histories[i-1][metric][maxiter] if i > 0 and metric in histories[i-1] else 0.0]+result[:maxiter], 
                           label=task_name, linewidth=3, clip_on=False)
            
            axs[i].set_xlim([0, maxiter])
            axs[i].set_ylim([0.0, 1.0])
            axs[i].set_xticklabels(list(range(0, maxiter+1, maxiter//5)), fontsize=14)
            axs[i].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
            axs[i].set_xticks(range(0, maxiter+1, maxiter//10), minor=True)
            axs[i].set_yticks([i/10 for i in range(0, 11)], minor=True)
            axs[i].set_xlabel("Iterations", fontsize=18) 
            axs[i].legend(markerscale=1, fontsize=14, loc='lower right')
            axs[i].grid(which='both', color='#BDBDBD', linestyle='--', linewidth=1)
    
    fig.tight_layout()
    plt.show()