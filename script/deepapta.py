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
#import ipykernel
#import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.metrics import average_precision_score as  aps
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D,Conv2D, Dense, MaxPooling1D,MaxPooling2D, Flatten, LSTM, Input, Attention, MultiHeadAttention
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
from focal_loss import BinaryFocalLoss
import datetime
import pickle as pkl

# CONSTANTS

TF_ENABLE_ONEDNN_OPTS=0
EPOCHS = 100 #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 500 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.    
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

def create_cnnbilstm_attention(gamm,pos_alpha, dropout,layer_num):
    first_inp=Input(shape=(35,4), name='md_1')
    model1 = Sequential()(first_inp)
    model1=Conv1D(filters=12, kernel_size=1, #padding='same',
                 input_shape=(35, 4))(model1)         
    model1=MaxPooling1D(pool_size=1)(model1)
    model1 = Dense(32, activation='relu')(model1)
    model1=Dense(4, activation='softmax')(model1)
    
    
    second_inp=Input(shape=(126,1), name='md_2')
    model2 = Sequential()(second_inp)
    model2=Conv1D(filters=1, kernel_size=100,
                 input_shape=(126,1))(model2)
    model2=MaxPooling1D(pool_size=20)(model2)
    model2 = Dense(4, activation='relu')(model2)
    

    model3=concatenate([model1,model2],axis=1)
    model3 = Bidirectional(LSTM(units=100, return_sequences=True))(model3)
    model3 = Dropout(dropout)(model3)
    model3 = Bidirectional(LSTM(units=layer_num))(model3)
    model3 = Dropout(dropout)(model3)
    model3 = Attention()([model3,model3])
    model3 = Dense(2, activation='sigmoid')(model3)

    model=Model(inputs=[first_inp, second_inp], outputs=model3)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def create_combine(gamm,pos_alpha, dropout,layer_num):
    first_inp=Input(shape=(35,4), name='md_1')
    model1 = Sequential()(first_inp)
    model1=Conv1D(filters=12, kernel_size=1,
                 input_shape=(35, 4))(model1)         
    model1=MaxPooling1D(pool_size=1)(model1)
    model1 = Dense(32, activation='relu')(model1)
    model1=Dense(4, activation='softmax')(model1)
    
    
    second_inp=Input(shape=(126,1), name='md_2')
    model2 = Sequential()(second_inp)
    model2=Conv1D(filters=1, kernel_size=100,
                 input_shape=(126,1))(model2)
    model2=MaxPooling1D(pool_size=20)(model2)
    model2 = Dense(4, activation='relu')(model2)
    

    model3=concatenate([model1,model2],axis=1)
    model3 = Bidirectional(LSTM(units=100, return_sequences=True))(model3)
    model3 = Dropout(dropout)(model3)
    model3 = Bidirectional(LSTM(units=layer_num))(model3)
    model3 = Dropout(dropout)(model3)
    model3 = Dense(2, activation='sigmoid')(model3)

    
    model=Model(inputs=[first_inp, second_inp], outputs=model3)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# Helper

def model_run(epoch,serialize_dir,X_train, y_train,model,batch_size):
    param = model.summary()
    print(param)
    history = model.fit(X_train, y_train
                    , batch_size = batch_size
                    , epochs=epoch
                    , verbose=1
                    , validation_split=0.1
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

        label_pos = int(label_pos)#*len(se))
        label_neg = int(label_neg)#*len(se))
        labels = []
        labels = ['1']*int(label_pos)+ ['0']*int(label_neg)
        one_hot_encoder = OneHotEncoder(categories='auto')
        labels = np.array(labels).reshape(-1, 1)
        input_labels = one_hot_encoder.fit_transform(labels).toarray()
        print('Labels:\n',labels.T)
        print('One-hot encoded labels:\n',input_labels.T)
        self.onehot_feature,self.input_label,self.shape_feature=onehot_features,input_labels,shape_features
        print("DATA PROCESSING COMPLETE")
    
    
    def data_sample(self,pos,neg):
        self.onehot_features=np.concatenate((self.onehot_feature[:pos],self.onehot_feature[neg:]))
        self.input_labels=np.concatenate((self.input_label[:pos],self.input_label[neg:]))
        self.shape_features=np.concatenate((self.shape_feature[:pos],self.shape_feature[neg:]))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.onehot_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        self.X_train2, self.X_test2, self.y_train, self.y_test = train_test_split(self.shape_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        print('DATA SAMPLING COMPLETE')
    
    
    def model(self,BATCH_SIZE,epochs,layer_num):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.onehot_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        self.X_train2, self.X_test2, self.y_train, self.y_test = train_test_split(self.shape_features, self.input_labels, test_size=self.test_ratio, random_state=42)
        self.resultdir = self.outputpath
        
        self.model_cnn = create_cnn()
        #model_cnn.load_weights(self.resultdir+'cnn/model.h5')
        self.history_cnn = model_run(epochs,self.resultdir+'cnn/', self.X_train, self.y_train, self.model_cnn,BATCH_SIZE)
        create_plots(self.model_cnn,self.history_cnn,self.resultdir+'cnn/')
        self.y_label_cnn, self.y_score_cnn, self.fpr_cnn, self.tpr_cnn, self.roc_auc_cnn = evaluate(self.model_cnn,self.X_test,self.y_test,self.resultdir+'cnn/','CNN')
        
        
        self.model_bilstm = create_bilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, dropout = DROPOUT_RATIO)
        #model_bilstm.load_weights(self.resultdir+'bilstm/model.h5')
        self.history_bilstm = model_run(epochs,self.resultdir+'bilstm/'
                                        , self.X_train
                                        #, self.X_train2.reshape(self.X_train2.shape[0],self.X_train2.shape[1],1)
                                        , self.y_train, self.model_bilstm,BATCH_SIZE)
        create_plots(self.model_bilstm,self.history_bilstm,self.resultdir+'bilstm/')
        self.y_label_lstm, self.y_score_lstm, self.fpr_lstm, self.tpr_lstm, self.roc_auc_lstm = evaluate(self.model_bilstm
                                                                 ,self.X_test
                                                                #,self.X_test2.reshape(self.X_test2.shape[0],self.X_test2.shape[1],1)
                                                              ,self.y_test,self.resultdir+'bilstm/','BiLSTM')    
        
        
        #model_cnnbilstm = create_cnnbilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
        #model_cnnbilstm.load_weights(self.resultdir+'cnn_bilstm/model.h5')
        self.model_cnnbilstm= create_cnnbilstm_attention(gamm=2,pos_alpha=0.25,dropout=0.1,layer_num=layer_num)
        self.history_cnnbilstm = model_run(epochs,self.resultdir+'cnn_bilstm/', [self.X_train,self.X_train2], self.y_train, self.model_cnnbilstm,BATCH_SIZE)
        create_plots(self.model_cnnbilstm,self.history_cnnbilstm,self.resultdir+'cnn_bilstm/')
        self.y_label_cnnlstm, self.y_score_cnnlstm, self.fpr_cnnlstm, self.tpr_cnnlstm, self.roc_auc_cnnlstm = evaluate(self.model_cnnbilstm,[self.X_test,self.X_test2],self.y_test,self.resultdir+'cnn_bilstm/','DeepAptamer')
        print("MODELS CREATED AND EVALUATED")
    
    def model_combine(self,BATCH_SIZE,epochs,layer_num):
        #model_cnnbilstm = create_cnnbilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
        #model_cnnbilstm.load_weights(self.resultdir+'cnn_bilstm/model.h5')
        self.model_cnnbilstm= create_combine(gamm=2,pos_alpha=0.25,dropout=0.1,layer_num=layer_num)
        self.history_cnnbilstm = model_run(epochs,self.resultdir+'cnn_bilstm/', [self.X_train,self.X_train2], self.y_train, self.model_cnnbilstm,BATCH_SIZE)
        create_plots(self.model_cnnbilstm,self.history_cnnbilstm,self.resultdir+'cnn_bilstm/')
        self.y_label_cnnlstm, self.y_score_cnnlstm, self.fpr_cnnlstm, self.tpr_cnnlstm, self.roc_auc_cnnlstm = evaluate(self.model_cnnbilstm,[self.X_test,self.X_test2],self.y_test,self.resultdir+'cnn_bilstm/','DeepAptamer')
    
    def model_cnnbilstm_attention(self,BATCH_SIZE,epochs,layer_num):
        self.resultdir = self.outputpath
        #model_cnnbilstm = create_cnnbilstm(rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO)
        #model_cnnbilstm.load_weights(self.resultdir+'cnn_bilstm/model.h5')
        self.model_cnnbilstm= create_cnnbilstm_attention(gamm=2,pos_alpha=0.25,dropout=0.1,layer_num=layer_num)
        self.history_cnnbilstm = model_run(epochs,self.resultdir+'cnn_bilstm/', [self.X_train,self.X_train2], self.y_train, self.model_cnnbilstm,BATCH_SIZE)
        create_plots(self.model_cnnbilstm,self.history_cnnbilstm,self.resultdir+'cnn_bilstm/')
        self.y_label_cnnlstm, self.y_score_cnnlstm, self.fpr_cnnlstm, self.tpr_cnnlstm, self.roc_auc_cnnlstm = evaluate(self.model_cnnbilstm,[self.X_test,self.X_test2],self.y_test,self.resultdir+'cnn_bilstm/','DeepAptamer')
    
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