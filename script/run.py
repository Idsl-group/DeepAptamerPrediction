import tensorflow as tf
import keras
import re
import keras_tuner
from keras.regularizers import *
from tabnanny import verbose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import ipykernel
import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.metrics import average_precision_score as  aps
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D,Conv2D, Dense, MaxPooling1D,MaxPooling2D, Flatten, LSTM, Input # type: ignore
import tensorflow.keras.backend as K # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore
from sklearn.metrics import confusion_matrix
import itertools
#import pydot
import getopt
import sys
import numpy as np
from sklearn import metrics
import pylab as plt
import math
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding,Bidirectional,Dropout # type: ignore
from tensorflow.keras.layers import concatenate # type: ignore
from tensorflow.keras.models import Model # type: ignore
from focal_loss import BinaryFocalLoss
import sys
import os
from deepapta import *

# Setup Directory Files
TF_ENABLE_ONEDNN_OPTS=0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'script'))
datadir = os.path.join(BASE_DIR, "../data/sequence/")
weightdir = os.path.join(BASE_DIR, "../model/")

# CTFG Model Creation
CTGF_DeepAptamer = deepapta(0.1, os.path.join(datadir, "CT_Shapes/CT-20_data.pkl"), os.path.join(datadir, "CT-20.txt"), '150000', '300000', os.path.join(weightdir, 'CTGF/9k_1w/'))

# CTFG Model Data Setup
CTGF_DeepAptamer.data_process()

# # CTFG Model Sample Setup
pos,neg=9000,-10000
CTGF_DeepAptamer.data_sample(pos, neg)


# CTGF_DeepAptamer.tuner()

# diagnose_training_data(
#     CTGF_DeepAptamer.X_train,
#     CTGF_DeepAptamer.X_train2.reshape(-1, 126, 1),
#     CTGF_DeepAptamer.y_train
# )

# temp_model = create_cnnbilstm_attention(gamm=2,pos_alpha=0.25,dropout=0.1,layer_num=100)
# diagnose_model_behavior(
#     temp_model,
#     CTGF_DeepAptamer.X_train,
#     CTGF_DeepAptamer.X_train2.reshape(-1, 126, 1),
#     CTGF_DeepAptamer.y_train
# )


# CTFG Model Setup Model & Run
CTGF_DeepAptamer.model(128,100,64)

# CTFG Model Get Metrics
CTGF_DeepAptamer.all_metrics('CTGF')