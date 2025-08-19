import tensorflow as tf
import keras
import re
import keras_tuner as kt
from keras.regularizers import *
from tabnanny import verbose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import ipykernel
#import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.metrics import average_precision_score as  aps
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input, Conv1D,Conv2D, Dense, MaxPooling1D,MaxPooling2D, Flatten, LSTM, Input, Attention, MultiHeadAttention, Concatenate 
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import itertools
#import pydot
import getopt
import sys
import numpy as np
from sklearn import metrics
import pylab as plt
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Bidirectional,Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from focal_loss import BinaryFocalLoss
import datetime
import pickle as pkl

# CONSTANTS

TF_ENABLE_ONEDNN_OPTS=0
EPOCHS = 100 #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 300 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.    
INPUT_DIM = 4 # a vocabulary of 4 words in case of fnn sequence (ATCG)
OUTPUT_DIM = 1 # Embedding output
RNN_HIDDEN_DIM = 62
DROPOUT_RATIO = 0.1 # proportion of neurones not used for training
MAXLEN = 150 # cuts text after number of these characters in pad_sequences

# CNN

def create_cnn():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=12, 
                 input_shape=(35, 4)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
    model.summary()
    return model

# BiLSTM

def create_bilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_hidden_dim, return_sequences=True),input_shape=(35, 4)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(rnn_hidden_dim)))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# CNN-BiLSTM (DeepAptamer)

# def create_cnnbilstm_attention(gamm,pos_alpha, dropout,layer_num):
#     first_inp=Input(shape=(35,4), name='md_1')
#     model1 = Sequential()(first_inp)
#     model1=Conv1D(filters=12, kernel_size=1, #padding='same',
#                  input_shape=(35, 4))(model1)         
#     model1=MaxPooling1D(pool_size=1)(model1)
#     model1 = Dense(32, activation='relu')(model1)
#     model1=Dense(4, activation='softmax')(model1)
    
    
#     second_inp=Input(shape=(126,1), name='md_2')
#     model2 = Sequential()(second_inp)
#     model2=Conv1D(filters=1, kernel_size=100,
#                  input_shape=(126,1))(model2)
#     model2=MaxPooling1D(pool_size=20)(model2)
#     model2 = Dense(4, activation='relu')(model2)
    

#     model3=concatenate([model1,model2],axis=1)
#     model3 = Bidirectional(LSTM(units=100, return_sequences=True))(model3)
#     model3 = Dropout(dropout)(model3)
#     model3 = Bidirectional(LSTM(units=layer_num))(model3)
#     model3 = Dropout(dropout)(model3)
#     model3 = Attention()([model3,model3])
#     model3 = Dense(2, activation='softmax')(model3)

#     model=Model(inputs=[first_inp, second_inp], outputs=model3)
#     model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
#     model.summary()
#     return model

# def create_cnnbilstm_attention(gamm=2, pos_alpha=0.25, dropout=0.1, layer_num=100):

#     # DNA (one-hot) sequences process
#     first_inp = Input(shape=(35,4), name='md_1')
#     model1 = Conv1D(filters=12, kernel_size=1, 
#                     kernel_regularizer=l2(0.01))(first_inp)
#     model1 = MaxPooling1D(pool_size=1)(model1)
#     model1 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(model1)
#     model1 = Dense(4, activation='softmax', kernel_regularizer=l2(0.01))(model1)

#     # DNA Shape params process
#     second_inp = Input(shape=(126,1), name='md_2')
#     model2 = Conv1D(filters=1, kernel_size=100,
#                     kernel_regularizer=l2(0.01))(second_inp)
#     model2 = MaxPooling1D(pool_size=20)(model2)
#     model2 = Dense(4, activation='relu', kernel_regularizer=l2(0.01))(model2)

#     # Combine
#     model3 = concatenate([model1, model2], axis=1)
#     model3 = Bidirectional(LSTM(units=100, return_sequences=True, kernel_regularizer=l2(0.01)))(model3)
#     model3 = Dropout(dropout)(model3)
#     model3 = Bidirectional(LSTM(units=layer_num, return_sequences=True, kernel_regularizer=l2(0.01)))(model3)
#     model3 = Dropout(dropout)(model3)
#     model3 = Attention()([model3, model3])
#     model3 = GlobalAveragePooling1D()(model3)
#     model3 = Dense(2, activation='softmax', kernel_regularizer=l2(0.01))(model3)

#     model = Model(inputs=[first_inp, second_inp], outputs=model3)

#     # Use focal loss and AdaDelta
#     optimizer = tf.keras.optimizers.Adadelta(
#         learning_rate=0.001,
#         rho=0.3,           # Decay rate (0.2-0.5 range)
#         epsilon=1e-6       # Delta values
#     )

#     model.compile(
#         optimizer=optimizer,
#         loss=BinaryFocalLoss(gamma=gamm, pos_weight=pos_alpha),
#         metrics=['accuracy']
#     )
#     return model

def create_cnnbilstm_attention(gamm=2, pos_alpha=0.25, dropout=0.3, layer_num=64):
    """
    CNN-BiLSTM with Attention (DeepAptamer-style) architecture.
    Matches the diagram you provided.
    """

    # --- Sequence Input (one-hot DNA encoding) ---
    seq_input = Input(shape=(35, 4), name="md_1")
    x1 = Sequential()(seq_input)
    x1 = Conv1D(12, kernel_size=3, activation="relu", padding="same", kernel_regularizer=l2(2.1513326719705766e-06))(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Dense(32, activation="relu", kernel_regularizer=l2(2.1513326719705766e-06))(x1)
    x1 = Dense(4, activation="relu", kernel_regularizer=l2(2.1513326719705766e-06))(x1)

    # --- Shape Input (secondary structure features) ---
    shape_input = Input(shape=(138, 1), name="md_2")
    x2 = Sequential()(shape_input)
    x2 = Conv1D(1, kernel_size=3, activation="relu", padding="same", kernel_regularizer=l2(2.1513326719705766e-06))(x2)
    x2 = MaxPooling1D(pool_size=2)(x2)
    x2 = Dense(4, activation="relu", kernel_regularizer=l2(2.1513326719705766e-06))(x2)

    # --- Merge features ---
    print("x1 shape:", x1.shape)  # expect (None, T1, 4)
    print("x2 shape:", x2.shape)  # expect (None, T2, 4)
    merged = Concatenate(axis=1)([x1, x2])   # (batch, timesteps, features)

    # --- BiLSTM layers (both return sequences=True) ---
    x = Bidirectional(LSTM(layer_num, return_sequences=True))(merged)
    x = Dropout(dropout)(x)
    x = Bidirectional(LSTM(layer_num, return_sequences=True))(x)
    x = Dropout(dropout)(x)

    # --- Attention Layer (self-attention) ---
    attn = tf.keras.layers.Attention()([x, x])
    attn = tf.keras.layers.GlobalAveragePooling1D(name='gap_time')(attn)  

    # --- Output classifier ---
    out = Dense(2, activation="softmax")(attn)

    # --- Model ---
    model = Model(inputs=[seq_input, shape_input], outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.91, beta_2=0.985, epsilon=3.1334771163070644e-07),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.0),
        metrics=["accuracy"]
    )

    return model

import tensorflow.keras.layers as layers
def build_deepaptamer(hp):
    """
    CNN-BiLSTM with Attention (DeepAptamer-style) architecture.
    Tunable hyperparameters: layer_num, dropout, lr, optimizer, L2 reg, Adam betas, Adam epsilon.
    """

    # --- Tunable hyperparameters ---
    layer_num = hp.Int("layer_num", min_value=64, max_value=256, step=64)
    dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
    lr = hp.Float("lr", 1e-5, 1e-3, sampling="log")
    opt_choice = hp.Choice("optimizer", ["adam", "adadelta"])

    # regularization
    l2_reg = hp.Float("l2_reg", 1e-6, 1e-2, sampling="log")

    # Adam params
    beta_1 = hp.Float("beta_1", 0.85, 0.95, step=0.02)
    beta_2 = hp.Float("beta_2", 0.98, 0.999, step=0.005)
    epsilon = hp.Float("epsilon", 1e-8, 1e-6, sampling="log")

    # --- Inputs ---
    seq_input = Input(shape=(35, 4), name="md_1")
    shape_input = Input(shape=(138, 1), name="md_2")

    # --- Sequence branch ---
    x1 = Sequential()(seq_input)
    x1 = layers.Conv1D(12, kernel_size=3, activation="relu", padding="same",
                       kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x1)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)
    x1 = layers.Dense(32, activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x1)
    x1 = layers.Dense(4, activation="relu")(x1)

    # --- Shape branch ---
    x2 = Sequential()(shape_input)
    x2 = layers.Conv1D(1, kernel_size=3, activation="relu", padding="same",
                       kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x2)
    x2 = layers.MaxPooling1D(pool_size=2)(x2)
    x2 = layers.Dense(4, activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x2)

    # --- Merge ---
    merged = layers.Concatenate(axis=1)([x1, x2])

    # --- BiLSTM stack ---
    x = layers.Bidirectional(layers.LSTM(layer_num, return_sequences=True))(merged)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(layer_num, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    # --- Attention + pooling ---
    attn = tf.keras.layers.Attention()([x, x])
    attn = tf.keras.layers.GlobalAveragePooling1D(name='gap_time')(attn)

    # --- Classifier ---
    out = layers.Dense(2, activation="softmax")(attn)

    model = Model(inputs=[seq_input, shape_input], outputs=out)

    # --- Optimizer ---
    if opt_choice == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )
    else:
        optimizer = tf.keras.optimizers.Adadelta(
            learning_rate=1.0, rho=0.95, epsilon=epsilon
        )

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def run_batchsize_bayes(
    X_seq, X_shape, y,
    workdir="tuner_results",
    batch_sizes=[128, 256, 300, 512],
    max_trials=20, epochs=50, val_fraction=0.1
):
    """
    Runs Bayesian Optimization tuner for multiple batch sizes,
    returns best (model, hyperparams, batch_size).
    """

    # --- Train/val split
    Xs_tr, Xs_val, Xsh_tr, Xsh_val, y_tr, y_val = train_test_split(
        X_seq, X_shape, y, test_size=val_fraction, random_state=42, shuffle=True
    )

    # --- Class weights
    y_integers = np.argmax(y_tr, axis=1)
    cw = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    class_weights = {int(cls): float(w) for cls, w in zip(np.unique(y_integers), cw)}
    print("Using class weights:", class_weights)

    best_overall = None
    best_acc = -1

    # --- Search over batch sizes ---
    for bs in batch_sizes:
        print(f"\n🔍 Running BayesianOptimization tuner with batch size = {bs}\n")

        tuner = kt.BayesianOptimization(
            build_deepaptamer,
            objective="val_accuracy",
            max_trials=max_trials,
            executions_per_trial=2,
            directory=workdir,
            project_name=f"deepaptamer_bs{bs}"
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        tuner.search(
            x=[Xs_tr, Xsh_tr],
            y=y_tr,
            validation_data=([Xs_val, Xsh_val], y_val),
            epochs=epochs,
            batch_size=bs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        best_hp = tuner.get_best_hyperparameters(1)[0]
        best_model = tuner.get_best_models(1)[0]

        val_acc = best_model.evaluate([Xs_val, Xsh_val], y_val, verbose=0)[1]
        print(f"✅ Batch size {bs}: val_acc={val_acc:.4f}, best_hp={best_hp.values}")

        tuner.results_summary(num_trials=10)

        if val_acc > best_acc:
            best_acc = val_acc
            best_overall = (best_model, best_hp, bs)

    print("\n=== Best Configuration ===")
    print(f"Batch size: {best_overall[2]}, Val Accuracy: {best_acc:.4f}")
    print("Best Hyperparameters:", best_overall[1].values)

    return best_overall

# Helper

def model_run(epoch,serialize_dir,X_train, y_train,model,batch_size):
    param = model.summary()
    print(param)

    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=np.argmax(y_train, axis=1))
    class_weights = {0: weights[0], 1: weights[1]}

    history = model.fit(X_train, y_train
                    , batch_size = batch_size
                    , epochs=epoch
                    , verbose=1
                    , validation_split=0.1
                    , class_weight=class_weights
                    )

    # Save as json
    model_json = model.to_json()
    with open(serialize_dir+"model.json", "w") as json_file:
        json_file.write(model_json)

    # Save as h5 file
    model.save_weights(serialize_dir+"model.h5")
    print("Saved model to disk")
    return history


def create_plots(model,trained_model,outpath):
    plt.plot(trained_model.history['accuracy'])
    plt.plot(trained_model.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(outpath+'accuracy.png')
    plt.show()
    plt.clf()

    plt.plot(trained_model.history['loss'])
    plt.plot(trained_model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(outpath+'loss.png')
    plt.show()
    plt.clf()

    plot_model(model,to_file=outpath+'model.png')   


def evaluate(model,X_test,y_test,resultdir,dnn_type):
    y_score = model.predict(X_test)
    plt.rcParams['font.size'] = 14
    Font={'size':43, 'family':'Arial'}
    cm = confusion_matrix(np.argmax(y_test, axis=1), 
                      np.argmax(y_score, axis=1))
    print('Confusion matrix:\n',cm)
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(dnn_type,Font)
    plt.colorbar()

    plt.ylabel('True label',fontdict=Font)
    plt.xlabel('Predicted label',Font)
    plt.xticks([0, 1]); plt.yticks([0, 1])
    plt.grid('off')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 fontdict=Font,
                 horizontalalignment='center',
                 color='white' if cm[i, j] > 0.5 else 'black')
    
    plt.savefig(resultdir+'confusion_matrix.png')
    plt.show()
    plt.clf()
    
    Font={'size':18, 'family':'Arial'}
    fpr, tpr, thersholds = roc_curve(y_test.T[1], y_score.T[1], pos_label=1)
    roc_auc = auc(fpr,tpr)
    plt.figure(figsize=(6,6))
    #plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr,label = dnn_type+' = %0.3f' % roc_auc, color='RoyalBlue')
    #plt.xlabel('False positive rate')
    #plt.ylabel('True positive rate')
    plt.title('ROC curve',Font)
    #plt.legend(loc='best')
    #plt.show()   
    #plt.savefig(resultdir+'ROC.png')
    #plt.clf() 
    plt.legend(loc = 'lower right', prop=Font)
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.tick_params(labelsize=18)
    plt.savefig(resultdir+'ROC.png')
    plt.show()
    return y_test, y_score,fpr,tpr,roc_auc


def diagnose_training_data(X_seq, X_shape, y):
    print("\n--- DATA SHAPE INFO ---")
    print(f"Sequence input shape: {X_seq.shape}")   # Expect (N, 35, 4)
    print(f"Shape input shape: {X_shape.shape}")    # Expect (N, 126, 1)
    print(f"Labels shape: {y.shape}")               # Expect (N, 2) if one-hot, or (N, 1) if binary

    # Check label distribution
    if y.ndim == 2 and y.shape[1] == 2:
        class_counts = y.sum(axis=0)
        print(f"Class counts: {class_counts}  (ratio: {class_counts[0]/class_counts.sum():.2f} / {class_counts[1]/class_counts.sum():.2f})")
    else:
        unique, counts = np.unique(y, return_counts=True)
        print(f"Class counts: {dict(zip(unique, counts))}")

    # Peek at first few samples
    print("\nFirst few labels:")
    print(y[:5])
    
    # Sanity check: random sample label vs data
    idx = np.random.randint(0, len(y))
    print(f"\nRandom sample index {idx}: Label = {y[idx]}")
    print(f"Sequence data sample (first row):\n{X_seq[idx][:5]}")
    print(f"Shape data sample (first row):\n{X_shape[idx][:5]}")

    # Detect all-zero / constant feature issue
    print("\nSequence feature variance:", np.var(X_seq))
    print("Shape feature variance:", np.var(X_shape))

def diagnose_model_behavior(model, X_seq, X_shape, y):
    # Predict before training
    preds = model.predict([X_seq[:200], X_shape[:200]], verbose=0)
    print("\n--- INITIAL MODEL OUTPUT SAMPLE ---")
    print(preds[:5])

    # Distribution of first output neuron before training
    print("Prediction mean per class (before training):")
    if preds.shape[1] == 2:
        print("Class 0 mean:", preds[:,0].mean())
        print("Class 1 mean:", preds[:,1].mean())
    else:
        print("Sigmoid output mean:", preds.mean())


def ks(target,resultdir,y_predicted1, y_true1, dnn_type1, y_predicted2, y_true2, dnn_type2, y_predicted3, y_true3, dnn_type3):
  Font={'size':18, 'family':'Arial'}
  
  label1=y_true1
  label2=y_true2
  label3=y_true3
  fpr1,tpr1,thres1 = roc_curve(label1.T[1], y_predicted1.T[1],pos_label=1)
  fpr2,tpr2,thres2 = roc_curve(label2.T[1], y_predicted2.T[1],pos_label=1)
  fpr3,tpr3,thres3 = roc_curve(label3.T[1], y_predicted3.T[1],pos_label=1)
  roc_auc1 = metrics.auc(fpr1, tpr1)
  roc_auc2 = metrics.auc(fpr2, tpr2)
  roc_auc3 = metrics.auc(fpr3, tpr3)
  
  plt.figure(figsize=(6,6))
  plt.plot(fpr1, tpr1, 'b', label = dnn_type1+' = %0.3f' % roc_auc1, color=plt.cm.Paired(0),lw=2)
  plt.plot(fpr2, tpr2, 'b', label = dnn_type2+' = %0.3f' % roc_auc2, color=plt.cm.Paired(1),lw=2)
  plt.plot(fpr3, tpr3, 'b', label = dnn_type3+' = %0.3f' % roc_auc3, color=plt.cm.Paired(2),lw=2)
  plt.legend(loc = 'lower right', prop=Font)
  plt.plot([0, 1], [0, 1],'k--')
  plt.xlim([0, 1.05])
  plt.ylim([0, 1.05])
  plt.ylabel('True Positive Rate', Font,weight='bold')
  plt.xlabel('False Positive Rate', Font,weight='bold')
  plt.tick_params(labelsize=15)
  plt.title(target,Font,weight='bold')
  plt.savefig(resultdir+'roc_auc.png')
  plt.show()

  return roc_auc1,roc_auc2,roc_auc3


def ks_pr(target,resultdir,y_predicted1, y_true1, dnn_type1, y_predicted2, y_true2, dnn_type2, y_predicted3, y_true3, dnn_type3):
  Font={'size':18, 'family':'Arial'}
  
  label1=y_true1
  label2=y_true2
  label3=y_true3
  precision1,recall1,thres1 = pr_curve(label1.T[1], y_predicted1.T[1])#,pos_label=1)
  precision2,recall2,thres2 = pr_curve(label2.T[1], y_predicted2.T[1])#,pos_label=1)
  precision3,recall3,thres3 = pr_curve(label3.T[1], y_predicted3.T[1])#,pos_label=1)
  pr_auc1 = aps(label1.T[1], y_predicted1.T[1])#,pos_label=1)
  pr_auc2 = aps(label2.T[1], y_predicted2.T[1])#,pos_label=1)
  pr_auc3 = aps(label3.T[1], y_predicted3.T[1])#,pos_label=1)
  
  plt.figure(figsize=(6,6))
  plt.plot(recall1, precision1, 'b', label = dnn_type1+' = %0.3f' % pr_auc1, color=plt.cm.Paired(0),lw=2)
  plt.plot(recall2, precision2, 'b', label = dnn_type2+' = %0.3f' % pr_auc2, color=plt.cm.Paired(1),lw=2)
  plt.plot(recall3, precision3, 'b', label = dnn_type3+' = %0.3f' % pr_auc3, color=plt.cm.Paired(2),lw=2)
  plt.legend(loc = 'lower right', prop=Font)
  plt.plot([0, 1], [0, 1],'k--')
  plt.xlim([0, 1.05])
  plt.ylim([0, 1.05])
  plt.ylabel('Precision', Font,weight='bold')
  plt.xlabel('Recall', Font,weight='bold')
  plt.tick_params(labelsize=15)
  plt.title(target,Font,weight='bold')
  plt.savefig(resultdir+'pr_auc.png')
  plt.show()

  return [pr_auc1,pr_auc2,pr_auc3], [precision1,precision2,precision3],[recall1,recall2,recall3]


class deepapta:
    def __init__(self, test_ratio,inputfile,seqfile,pos_num,neg_num,outputpath):
        self.test_ratio,self.inputfile,self.seqfile,self.pos_num,self.neg_num,self.outputpath = test_ratio,inputfile,seqfile,pos_num,neg_num,outputpath
        now = datetime.datetime.now()
        self.time = f"{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
    
    def data_process(self):
        onehot_fea = []
        shape_fea = []
        label_pos=float(self.pos_num)
        label_neg=float(self.neg_num)
        
        with open(self.inputfile, "rb") as f:
            data = pkl.load(f)
            
            onehot_fea = data['onehot_sequences']
            shape_fea = data['shapes']
            
        onehot_features = np.array(onehot_fea[:int(label_pos)]+onehot_fea[-1*int(label_neg):])
        shape_features = np.array(shape_fea[:int(label_pos)]+shape_fea[-1*int(label_neg):])
        print("One-hot DNA Sequences Shape : " + str(onehot_features.shape))
        print("Shape of DNA Sequences Shape : " + str(shape_features.shape))

        label_pos = int(label_pos)#*len(se))
        label_neg = int(label_neg)#*len(se))
        labels = []
        labels = ['1']*int(label_pos)+ ['0']*int(label_neg)
        one_hot_encoder = OneHotEncoder(categories='auto')
        labels = np.array(labels).reshape(-1, 1)
        input_labels = one_hot_encoder.fit_transform(labels).toarray()
        print('Labels:\n',labels.T)
        print('Label Shape : ' + str(labels.shape))
        print('One-hot encoded labels:\n',input_labels.T)
        print('One-hot encoded labels Shape : ' + str(input_labels.shape))
        self.onehot_feature,self.input_label,self.shape_feature=onehot_features,input_labels,shape_features
        print("DATA PROCESSING COMPLETE")
    
    
    def data_sample(self,pos,neg):
        self.onehot_features=np.concatenate((self.onehot_feature[:pos],self.onehot_feature[neg:]))
        self.input_labels=np.concatenate((self.input_label[:pos],self.input_label[neg:]))
        self.shape_features=np.concatenate((self.shape_feature[:pos],self.shape_feature[neg:]))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.onehot_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        self.X_train2, self.X_test2, self.y_train, self.y_test = train_test_split(self.shape_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        print(self.y_train.shape)
        print('DATA SAMPLING COMPLETE')
    
    
    def model(self,BATCH_SIZE,epochs,layer_num):
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.onehot_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        # self.X_train2, self.X_test2, _, _ = train_test_split(self.shape_features, self.input_labels, test_size=self.test_ratio, random_state=42)

        X_train_idx, X_test_idx, y_train, y_test = train_test_split(
            np.arange(len(self.input_labels)), self.input_labels, 
            test_size=self.test_ratio, random_state=42
        )

        # Use same indices for both feature sets
        self.X_train  = self.onehot_features[X_train_idx]
        self.X_test   = self.onehot_features[X_test_idx]
        self.X_train2 = self.shape_features[X_train_idx]
        self.X_test2  = self.shape_features[X_test_idx]
        self.y_train, self.y_test = y_train, y_test

        self.resultdir = self.outputpath

        # CNN BiLSTM
        
        self.model_cnnbilstm= create_cnnbilstm_attention(gamm=2,pos_alpha=0.25,dropout=0.3,layer_num=layer_num)

        self.history_cnnbilstm = model_run(epochs,self.resultdir+'cnn_bilstm/', [self.X_train,self.X_train2], self.y_train, self.model_cnnbilstm,BATCH_SIZE)
        
        create_plots(self.model_cnnbilstm,self.history_cnnbilstm,self.resultdir+'cnn_bilstm/')
        
        self.y_label_cnnlstm, self.y_score_cnnlstm, self.fpr_cnnlstm, self.tpr_cnnlstm, self.roc_auc_cnnlstm = evaluate(self.model_cnnbilstm,[self.X_test,self.X_test2],self.y_test,self.resultdir+'cnn_bilstm/','DeepAptamer')
        print("MODELS CREATED AND EVALUATED")

    def tuner(self):
        
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(
            np.arange(len(self.input_labels)), self.input_labels, 
            test_size=self.test_ratio, random_state=42
        )

        # Use same indices for both feature sets
        self.X_train  = self.onehot_features[X_train_idx]
        self.X_test   = self.onehot_features[X_test_idx]
        self.X_train2 = self.shape_features[X_train_idx]
        self.X_test2  = self.shape_features[X_test_idx]
        self.y_train, self.y_test = y_train, y_test

        run_batchsize_bayes(self.X_train, self.X_train2, self.y_train)
    
    def model_load(self,layer_num,modeldir):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.onehot_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        self.X_train2, self.X_test2, self.y_train, self.y_test = train_test_split(self.shape_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        #BATCH_SIZE = 32
        self.resultdir = self.outputpath
        #resultdir = "/Volumes/Data Backup/YX/Aptamer/script/Classification/LSTM/"
        
        self.model_cnn = create_cnn()
        self.model_cnn.load_weights(self.resultdir+modeldir+'cnn/model.h5')
        self.y_label_cnn, self.y_score_cnn, self.fpr_cnn, self.tpr_cnn, self.roc_auc_cnn = evaluate(self.model_cnn,self.X_test,self.y_test,self.resultdir+'cnn/','CNN')
        
        
        self.model_bilstm = create_bilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, dropout = DROPOUT_RATIO)
        self.model_bilstm.load_weights(self.resultdir+modeldir+'bilstm/model.h5')
        self.y_label_lstm, self.y_score_lstm, self.fpr_lstm, self.tpr_lstm, self.roc_auc_lstm = evaluate(self.model_bilstm
                                                                 ,self.X_test
                                                                #,self.X_test2.reshape(self.X_test2.shape[0],self.X_test2.shape[1],1)
                                                              ,self.y_test,self.resultdir+'bilstm/','BiLSTM')    
        
      
        #model_cnnbilstm = create_cnnbilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
        #model_cnnbilstm.load_weights(self.resultdir+'cnn_bilstm/model.h5')
      
        self.model_cnnbilstm= create_cnnbilstm_attention(gamm=2,pos_alpha=0.25,dropout=0.1,layer_num=layer_num)
        self.model_cnnbilstm.load_weights(self.resultdir+modeldir+'cnn_bilstm/model.h5')
        #create_plots(self.model_cnnbilstm,self.history_cnnbilstm,self.resultdir+'cnn_bilstm/')
        self.y_label_cnnlstm, self.y_score_cnnlstm, self.fpr_cnnlstm, self.tpr_cnnlstm, self.roc_auc_cnnlstm = evaluate(self.model_cnnbilstm,[self.X_test,self.X_test2],self.y_test,self.resultdir+'cnn_bilstm/','DeepAptamer')

        #history_cnnbilstm = model_run(100,self.resultdir+'cnn_bilstm/', [self.X_train,self.X_train2], self.y_train, model_cnnbilstm,BATCH_SIZE)
        #y_label_cnnlstm, y_score_cnnlstm, self.fpr_cnnlstm, self.tpr_cnnlstm, roc_auc_cnnlstm = evaluate(model_cnnbilstm,[self.X_test,self.X_test2],self.y_test,self.resultdir+'cnn_bilstm/','DeepAptamer')
        
        #self.y_score_cnn,self.y_label_cnn,self.y_score_lstm,self.y_label_lstm,self.y_score_cnnlstm,self.y_label_cnnlstm=y_score_cnn,y_label_cnn,y_score_lstm,y_label_lstm,y_score_cnnlstm,y_label_cnnlstm
        #self.model_cnn,self.model_bilstm,self.model_cnnbilstm=model_cnn,model_bilstm,model_cnnbilstm
    
    def all_metrics(self,target):

        self.roc_auc1,self.roc_auc2,self.roc_auc3=ks(target,self.resultdir,self.y_score_cnn,self.y_label_cnn,"CNN",self.y_score_lstm,self.y_label_lstm,"BiLSTM",self.y_score_cnnlstm,self.y_label_cnnlstm,"DeepAptamer")  
        self.pr_auc,self.precision,self.recall=ks_pr(target,self.resultdir,self.y_score_cnn,self.y_label_cnn,"CNN",self.y_score_lstm,self.y_label_lstm,"BiLSTM",self.y_score_cnnlstm,self.y_label_cnnlstm,"DeepAptamer")  
        self.f1_score1=f1_score(self.y_label_cnn.T[1], self.y_score_cnn.T[1].round(),pos_label=1)
        self.f1_score2=f1_score(self.y_label_lstm.T[1], self.y_score_lstm.T[1].round(),pos_label=1)
        self.f1_score3=f1_score(self.y_label_cnnlstm.T[1], self.y_score_cnnlstm.T[1].round(),pos_label=1)
        self.mmc1=mcc(self.y_label_cnn.T[1], self.y_score_cnn.T[1].round())
        self.mmc2=mcc(self.y_label_lstm.T[1], self.y_score_lstm.T[1].round())
        self.mmc3=mcc(self.y_label_cnnlstm.T[1], self.y_score_cnnlstm.T[1].round())
        self.precision=[metrics.precision_score(self.y_label_cnn.T[1], self.y_score_cnn.T[1].round())\
                        ,metrics.precision_score(self.y_label_lstm.T[1], self.y_score_lstm.T[1].round())
                        ,metrics.precision_score(self.y_label_cnnlstm.T[1], self.y_score_cnnlstm.T[1].round())
                        ]
        self.recall=[metrics.recall_score(self.y_label_cnn.T[1], self.y_score_cnn.T[1].round())
                     ,metrics.recall_score(self.y_label_lstm.T[1], self.y_score_lstm.T[1].round())
                     ,metrics.recall_score(self.y_label_cnnlstm.T[1], self.y_score_cnnlstm.T[1].round())
                    ]
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import rcParams
        import matplotlib as mpl
        sns.set_theme(style="white",font='Arial',font_scale=1.4)
        #custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        #sns.set_theme(style="ticks", rc=custom_params)
        mpl.rcParams["font.family"] = 'Arial'
        mpl.rcParams["mathtext.fontset"] = 'cm' 
        mpl.rcParams["axes.linewidth"] = 2
        font = {'family':'Arial','size':45}
        mpl.rc('font',**font)
        #mpl.rc('legend',**{'fontsize':45})
        mpl.rcParams['savefig.bbox'] = 'tight'
        font = {'family' : 'Arial','weight' : 'bold'}  
        plt.rc('font', **font) 
        Font={'size':18, 'family':'Arial'}
        self.metrics_df=pd.DataFrame({"Performance":[self.roc_auc1,self.roc_auc2,self.roc_auc3]
                                 +self.pr_auc
                                 +self.precision
                                 +self.recall
                                 +[self.f1_score1,self.f1_score2,self.f1_score3]
                                 +[self.mmc1,self.mmc2,self.mmc3],
                                   "Metrics":["AUROC"]*3
                                            +["AUPRC"]*3
                                            +["Precision"]*3
                                            +["Recall"]*3
                                            +["F1_Score"]*3
                                            +["MCC"]*3
                                ,"Models":["CNN","BiLSTM","DeepAptamer"]*6
                                            })
        
        plt.figure(figsize=(11,8))
        fig = sns.barplot(x ="Metrics", y = 'Performance', data = self.metrics_df, hue = "Models",palette=sns.color_palette("Paired"))
        fig.legend(loc='center right', bbox_to_anchor=(.8,1.05), ncol=3,fontsize=16)
        #fig.legend(ncol=4)
        fig.set_xticklabels( ["AUROC" ,
                                             "AUPRC" ,
                                             "Precision" ,
                                             "Recall" ,
                                             "F1_Score" ,
                                             "MCC" ], fontsize=14,weight='bold')
        fig.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0], fontsize=14,weight='bold')
        plt.ylabel('Performance', Font,weight='bold')
        plt.xlabel(target, Font,weight='bold')
        plt.grid(False)
        
        plt.savefig(self.resultdir+'all_metircs.png')