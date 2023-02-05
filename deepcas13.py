################################################################
# Script Name : DeepCas13.py
# Description : Functions for DeepCas13 model
# Author      : Xiaolong Cheng from Dr. Wei Li Lab
# Affiliation : Children's National Hospital
# Email       : xcheng@childrensnational.org
################################################################


import pandas as pd
import numpy as np
import RNA
import os
import argparse
from argparse import RawTextHelpFormatter

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv2D, SpatialDropout2D
from tensorflow.keras.layers import Input, LeakyReLU, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, Reshape
from tensorflow.keras.layers import Dropout, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import concatenate, TimeDistributed
from tensorflow.keras.layers import Activation, Embedding, GRU, LSTM, Bidirectional, SpatialDropout1D, SimpleRNN
import tensorflow.keras.backend as K
import tensorflow as tf

import logging
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

## Parse Parameters
parser = argparse.ArgumentParser(description="Predict CRISPR-Cas13d sgRNA on-target efficiency by DeepCas13", 
                                 formatter_class=RawTextHelpFormatter)

group_predict = parser.add_argument_group(title='Predict sgRNA efficiency')
group_predict.add_argument("--seq", type=str, help="The sequence input. The input file can be .csv, .txt or .fasta format. Please note that: \n\t1) if the input is in .csv or .txt format, there should be no header in the file. The input can contain one column or two columns (ID + seq); \n\t2) if --type is set as target, there should be one single sequence in the input file")
group_predict.add_argument("--model", type=str, help="Specify the path to the pretrained model")
group_predict.add_argument("--output", type=str, default="DeepCas13_predicted_sgRNA.csv", help="The output file")
group_predict.add_argument("--type", type=str, choices=["sgrna", "target"], default="sgrna", 
                   help="The acceptable prediction type: \n\t1) sgrna (default): predict the on-target efficiency of sgRNAs; \n\t2) target: design sgRNAs for the input target sequence")
group_predict.add_argument("--length", type=int, default=22, help="The sgRNA length. Default length is 22nt and this parameter only works when --type is set as target")

group_train = parser.add_argument_group(title='Train DeepCas13 model')
group_train.add_argument("--train", action='store_true', help="Set training mode")
group_train.add_argument("--data", type=str, help="The training data. The input file can be .csv or .txt format. There should be two columns in the file(no header names): \n\t1) the first column is the sgRNA sequence; \n\t2) the second columns is the sgRNA LFC")
group_train.add_argument("--savepath", type=str, help="Specify the path to save the model")

parser.add_argument("--basename", type=str, default="DeepCas13_Model", help="The basename of model files. Default basename is DeepCas13_Model")

##
dct_ohc_seq = {'A': [1, 0, 0, 0],
               'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'T': [0, 0, 0, 1],
               'N': [0, 0, 0, 0]}

dct_ohc_fold = {'(': [1, 0, 0],
                ')': [0, 1, 0],
                '.': [0, 0, 1],
                'N': [0, 0, 0]}

def get_fold(seq):
    for i in range(33-len(seq)):
        seq = seq + 'N'
    fc = RNA.fold_compound(seq)
    # compute MFE and MFE structure
    (mfe_struct, mfe) = fc.mfe()
    return mfe_struct

def seq_one_hot_code(seq):
    seq = seq.upper()
    lst_seq = list(seq)
    lst_seq.extend(['N' for i in range(33-len(seq))])
    return [dct_ohc_seq[i] for i in lst_seq]

def fold_one_hot_code(seq):
    lst_seq = list(seq)
    lst_seq.extend(['N' for i in range(33-len(seq))])
    return [dct_ohc_fold[i] for i in lst_seq]

def read_seq(file_path):
    logger.info('read sgRNA sequence file: ' + file_path)
    if file_path.endswith('.csv'):
        df_seq = pd.read_csv(file_path, header=None, index_col=None)
        if len(df_seq.columns) == 1:
            df_seq.columns = ['seq']
            df_seq['sgrna'] = ['sgrna_'+i for i in df_seq.seq.to_list()]
            df_seq = df_seq[['sgrna', 'seq']]
        elif len(df_seq.columns) == 2:
            df_seq.columns = ['sgrna', 'seq']
    elif file_path.endswith('.txt'):
        df_seq = pd.read_csv(file_path, names=['seq'], index_col=None, sep='\t')
        if len(df_seq.columns) == 1:
            df_seq.columns = ['seq']
            df_seq['sgrna'] = ['sgrna_'+i for i in df_seq.seq.to_list()]
            df_seq = df_seq[['sgrna', 'seq']]
        elif len(df_seq.columns) == 2:
            df_seq.columns = ['sgrna', 'seq']
    elif file_path.endswith('.fa') or file_path.endswith('.fasta'):
        with open(file_path) as f:
            lines = f.read().splitlines()
        df_seq = pd.DataFrame(columns = ['sgrna', 'seq'])
        df_seq['sgrna'] = [i[1:] for i in lines if i.startswith('>')]
        df_seq['seq'] = [i for i in lines if not i.startswith('>')]
    logger.info('find ' + str(len(df_seq)) + ' sgRNAs')
    return df_seq

def read_target(file_path, length):
    logger.info('read target sequence file: ' + file_path)
    if file_path.endswith('.csv'):
        df_seq = pd.read_csv(file_path, header=None, index_col=None)
        if len(df_seq.columns) == 1:
            df_seq.columns = ['seq']
            df_seq['sgrna'] = ['sgrna_'+i for i in df_seq.seq.to_list()]
            df_seq = df_seq[['sgrna', 'seq']]
        elif len(df_seq.columns) == 2:
            df_seq.columns = ['sgrna', 'seq']
    elif file_path.endswith('.txt'):
        df_seq = pd.read_csv(file_path, names=['seq'], index_col=None, sep='\t')
        if len(df_seq.columns) == 1:
            df_seq.columns = ['seq']
            df_seq['sgrna'] = ['sgrna_'+i for i in df_seq.seq.to_list()]
            df_seq = df_seq[['sgrna', 'seq']]
        elif len(df_seq.columns) == 2:
            df_seq.columns = ['sgrna', 'seq']
    elif file_path.endswith('.fa') or file_path.endswith('.fasta'):
        with open(file_path) as f:
            lines = f.read().splitlines()
        df_seq = pd.DataFrame(columns = ['sgrna', 'seq'])
        df_seq['sgrna'] = [i[1:] for i in lines if i.startswith('>')]
        df_seq['seq'] = [i for i in lines if not i.startswith('>')]
    ##
    seq_target = df_seq['seq'].to_list()[0]
    logger.info('target sequence contains: ' + str(len(seq_target)) + ' nt')
    if len(seq_target) <= length:
        return False
    lst_sgrna = []
    lst_seq = []
    for k in range(len(seq_target) - length):
        lst_sgrna.append('sgRNA_'+str(k)+'_'+str(k+length))
        lst_seq.append(seq_target[k:k+length])
    logger.info('there are ' + str(len(lst_sgrna)) + ' possible sgRNAs')
    ##
    dct_ACGT = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N', 'U':'A'}
    df_target = pd.DataFrame(columns = ['sgrna', 'seq'])
    df_target['sgrna'] = lst_sgrna
    df_target['seq'] = [''.join([dct_ACGT[i] for i in q][::-1]) for q in lst_seq]
    return df_target

def get_DeepScore(df_seq, model, basename):
    lst_input = df_seq.seq.to_list()
    lst_seq = lst_input.copy()
    for i in range(len(lst_seq)):
        seq = lst_seq[i]
        seq = seq.upper()
        seq = seq.replace('U', 'T')
        lst_seq[i] = seq
    # import models
    logger.info('load pretrained model from path: ' + model)
    lst_model = []
    for k in range(5):
        lst_model.append(keras.models.load_model(os.path.join(model, basename+str(k))))
    # preprocessing
    logger.info('predict sgRNA efficiency')
    X_test_seq = [seq_one_hot_code(seq) for seq in lst_seq]
    lst_fold = [get_fold(seq) for seq in lst_seq]
    X_test_fold = [fold_one_hot_code(fold) for fold in lst_fold]
    X_test_arr_seq = np.array(X_test_seq)
    X_test_arr_fold = np.array(X_test_fold)
    X_test_seq_CNN = np.reshape(X_test_arr_seq, (len(X_test_arr_seq), 1, 33, 4, 1)) 
    X_test_fold_CNN = np.reshape(X_test_arr_fold, (len(X_test_arr_fold), 1, 33, 3, 1))
    df_score = pd.DataFrame(columns=['y_pred', 'M0', 'M1', 'M2', 'M3', 'M4'])
    for k in range(5):
        y_pred = lst_model[k].predict([X_test_seq_CNN, X_test_fold_CNN])
        df_score['M'+str(k)] = [i[0] for i in y_pred]
    df_score['deepscore'] = df_score[['M0', 'M1', 'M2', 'M3', 'M4']].mean(axis=1)
    df_result = df_seq.copy()
    df_result['deepscore'] = df_score['deepscore'].to_list()
    return df_result

## Train model
def read_training_data(file_path):
    # read file
    logger.info('read training data file: ' + file_path)
    if file_path.endswith('.csv'):
        df_data = pd.read_csv(file_path, names=['seq', 'LFC'], index_col=None)
    elif file_path.endswith('.txt'):
        df_data = pd.read_csv(file_path, names=['seq', 'LFC'], index_col=None, sep='\t')
    # create y_value
    x1 = -0.3
    y1 = 0.7
    x2 = 0
    y2 = 0.3
    param_n = 1
    param_a = (np.log(1.0/(1-y2)-1) - np.log(1.0/(1-y1)-1))/(param_n*x1 - param_n*x2)
    param_b = -1*np.log(1.0/(1-y1)-1)/param_a - param_n*x1
    df_data['y_value'] = [1 - 1/(1+np.exp(-1*param_a*(param_n*i+param_b))) for i in df_data['LFC'].to_list()]
    return df_data

def train_deepcas13_model(df_train, savepath, basename):
    logger.info('train DeepCas13 model')
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=32)
    N = 0
    for train, test in kf.split(df_train):
        X_train = df_train.iloc[list(train),:]
        y_train = df_train.iloc[list(train),2]
        X_train_seq = [seq_one_hot_code(i) for i in X_train.seq.to_list()]
        X_train_fold = [fold_one_hot_code(get_fold(i)) for i in X_train.seq.to_list()]
        ###
        X_train_arr_seq = np.array(X_train_seq)
        X_train_arr_fold = np.array(X_train_fold)
        ###
        X_train_seq_CNN = np.reshape(X_train_arr_seq, (len(X_train_arr_seq), 1, 33, 4, 1)) 
        X_train_fold_CNN = np.reshape(X_train_arr_fold, (len(X_train_arr_fold), 1, 33, 3, 1))
        ## Seq
        seq_input = Input(shape=(1, 33, 4, 1))
        seq_conv1 = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(seq_input)
        seq_norm1 = TimeDistributed(BatchNormalization())(seq_conv1)
        seq_conv2 = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(seq_norm1)
        seq_norm2 = TimeDistributed(BatchNormalization())(seq_conv2)
        seq_drop1 = TimeDistributed(Dropout(0.5))(seq_norm2)
        seq_pool1 = TimeDistributed(MaxPooling2D((2, 2)))(seq_drop1)
        seq_flat1 = TimeDistributed(Flatten())(seq_pool1)
        seq_lstm1 = LSTM(100)(seq_flat1)
        seq_drop2 = Dropout(0.3)(seq_lstm1)
        seq_output = Dense(64, activation='relu')(seq_drop2)
        ## Fold
        fold_input = Input(shape=(1, 33, 3, 1))
        fold_conv1 = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(fold_input)
        fold_norm1 = TimeDistributed(BatchNormalization())(fold_conv1)
        fold_conv2 = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(fold_norm1)
        fold_norm2 = TimeDistributed(BatchNormalization())(fold_conv2)
        fold_drop1 = TimeDistributed(Dropout(0.5))(fold_norm2)
        fold_pool1 = TimeDistributed(MaxPooling2D((2, 2)))(fold_drop1)
        fold_flat1 = TimeDistributed(Flatten())(fold_pool1)
        fold_lstm1 = LSTM(100)(fold_flat1)
        fold_drop2 = Dropout(0.3)(fold_lstm1)
        fold_output = Dense(64, activation='relu')(fold_drop2)
        ## Merge
        merged = concatenate([seq_output, fold_output], axis=1, name='merged')
        NN_drop1 = Dropout(0.3)(merged)
        NN_dense2 = Dense(64)(NN_drop1)
        NN_output = Dense(1, activation='sigmoid')(NN_dense2)
        ###
        NN_model = Model([seq_input, fold_input], NN_output)
        NN_model.compile(optimizer='Adam', loss='mse')
        ###
        NN_model.fit([X_train_seq_CNN, X_train_fold_CNN],  y_train, epochs=30, batch_size=128, shuffle=True, verbose=0)
        logger.info('save trained model to path: ' + savepath + ' (Part ' + str(N+1) + ')')
        NN_model.save(os.path.join(savepath, basename+str(N)))
        N = N + 1

if __name__ == "__main__":
    # create logger
    logger = logging.getLogger('DeepCas13')
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter for console handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to console handler
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    ##
    args = parser.parse_args()
    if args.train:
        logger.info('Set training mode')
        if not os.path.isdir(args.savepath):
            logger.info('no such savepath: '+ args.savepath)
            os.mkdir(args.savepath)
            logger.info('create folder: '+ args.savepath)
        train_deepcas13_model(read_training_data(args.data), args.savepath, args.basename)
    else:
        if args.type == 'sgrna':
            logger.info('Predict sgRNA on-target efficiency')
            df_score = get_DeepScore(read_seq(args.seq), args.model, args.basename)
            logger.info('save DeepScore to file: ' + args.output)
            if args.output.endswith('.csv'):
                df_score.to_csv(args.output, header=True, index=False)
            elif args.output.endswith('.txt') or args.output.endswith('.tsv'):
                df_score.to_csv(args.output, header=True, index=False, sep='\t')
        elif args.type == 'target':
            logger.info('Design sgRNAs for target sequence')
            df_score = get_DeepScore(read_target(args.seq, args.length), args.model, args.basename)
            logger.info('save sgRNAs and DeepScore to file: ' + args.output)
            if args.output.endswith('.csv'):
                df_score.to_csv(args.output, header=True, index=False)
            elif args.output.endswith('.txt') or args.output.endswith('.tsv'):
                df_score.to_csv(args.output, header=True, index=False, sep='\t')
    logger.info('Done !')

