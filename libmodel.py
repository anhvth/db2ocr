import tensorflow.python.keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import BatchNormalization,Permute,TimeDistributed,Bidirectional,Lambda,GRU
from tensorflow.python.keras.layers import Flatten,BatchNormalization,Permute,TimeDistributed,Dense,Bidirectional,GRU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.optimizers import SGD, Adam
import numpy as np
import os,h5py,sys
from keras.utils import np_utils
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from tensorflow.python import keras
import itertools
from scipy import misc
# from keras import backend as K
K.set_learning_phase(1)
from pyson.utils import *
# ctc loss implemented by tensorflow
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
# A model base on cnn->lstm-> ctcloss
def cnn_feature_extractor(inputs):
    m = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',name='conv1')(inputs)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool1')(m)
    m = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same',name='conv2')(m)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool2')(m)
    m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv3')(m)
    m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv4')(m)

    m = ZeroPadding2D(padding=(0,1))(m)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool3')(m)

    m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv5')(m)
    m = BatchNormalization(axis=1)(m)
    m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv6')(m)
    m = BatchNormalization(axis=1)(m)
    m = ZeroPadding2D(padding=(0,1))(m)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool4')(m)
    m = Conv2D(512,kernel_size=(2,2),activation='relu',padding='valid',name='conv7')(m)
    return m
def cnn_lstm_ctc_model(height,nclass, tensors=None, width=None):
    if tensors is None:
        tensors = {
         'the_input':None, 'the_labels': None, 'input_length':None, 'label_length':None,'target_tensor':None
        }
        
    rnnunit  = 256
    inputs = Input(shape=(height,width,1),name='the_input', tensor=tensors['the_input'])
    #1. convnet layers
    m = cnn_feature_extractor(inputs)
    cnn_model = Model(inputs=[inputs], outputs=[m])
    #2. bi-lstm layers
    m = Permute((2,1,3),name='permute')(m)
    m = TimeDistributed(Flatten(),name='timedistrib')(m)

    m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm1')(m)
    m = Dense(rnnunit,name='blstm1_out',activation='linear')(m)
    m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm2')(m)
    basemodel = Model(inputs=inputs, outputs=m)    
    
    last_features = Bidirectional(GRU(rnnunit,return_sequences=False),name='last_features')(m)
    is_datetime = Dense(1, activation='sigmoid')(last_features)
    model_is_datetime = Model([inputs], [is_datetime])
    if not nclass == 3811:
        m = BatchNormalization()(m)
    y_pred = Dense(nclass,name='blstm2_out',activation='softmax')(m)
    
    #3. CTC loss compute 
    labels = Input(name='the_labels', shape=[None,], dtype='float32', tensor=tensors['the_labels'])
    input_length = Input(name='input_length', shape=[1], dtype='int64', tensor=tensors['input_length'])
    label_length = Input(name='label_length', shape=[1], dtype='int64', tensor=tensors['label_length'])
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])
    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.9)
    fn_compile = lambda: model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, metrics=['acc'],optimizer=adam, target_tensors=[tensors['target_tensor']])
    test_func = K.function([inputs], [y_pred])
    return {'model':model, \
            'basemodel':basemodel, \
            'test_func':test_func, \
            'fn_compile':fn_compile, \
            'cnn_model':cnn_model}

def cnn_cnn1d_ctc_model(height,nclass, tensors=None, width=None):
    if tensors is None:
        tensors = {
            'the_input':None, 'the_labels': None, 'input_length':None, 'label_length':None,'target_tensor':None
        }
        
    rnnunit  = 256
    inputs = Input(shape=(height,width,1),name='the_input', tensor=tensors['the_input'])
    #1. convnet layers
    m = cnn_feature_extractor(inputs)
    cnn_model = Model(inputs=[inputs], outputs=[m])
    #2. bi-lstm layers
    m = Permute((2,1,3),name='permute')(m)
    m = TimeDistributed(Flatten(),name='timedistrib')(m)
    m = Conv1D(512, 3, padding='valid')(m)
    m = MaxPool1D()(m)
    m = Conv1D(512, 3, padding='valid')(m)
    basemodel = Model(inputs=inputs, outputs=m)    

    last_features = Bidirectional(GRU(rnnunit,return_sequences=False),name='last_features')(m)
    is_datetime = Dense(1, activation='sigmoid')(last_features)
    model_is_datetime = Model([inputs], [is_datetime])
    if not nclass == 3811:
        m = BatchNormalization()(m)
    y_pred = Dense(nclass,name='blstm2_out',activation='softmax')(m)

    #3. CTC loss compute 
    labels = Input(name='the_labels', shape=[None,], dtype='float32', tensor=tensors['the_labels'])
    input_length = Input(name='input_length', shape=[1], dtype='int64', tensor=tensors['input_length'])
    label_length = Input(name='label_length', shape=[1], dtype='int64', tensor=tensors['label_length'])
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])
    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.9)
    fn_compile = lambda: model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, metrics=['acc'],optimizer=adam, target_tensors=[tensors['target_tensor']])
    test_func = K.function([inputs], [y_pred])
    return {'model':model, \
            'basemodel':basemodel, \
            'test_func':test_func, \
            'fn_compile':fn_compile, \
            'cnn_model':cnn_model}
def get_latest_checkpoint(checkpoint):
    sorted_paths_by_time = sorted(get_paths(checkpoint,'h5'), key=lambda x: os.path.getmtime(x))
    return sorted_paths_by_time[-1]




def ctc_beam_op(y_pred, input_length, greedy=True):
    y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-7)
    input_length = tf.to_int32(input_length)
    if greedy:
        # print("greedy decoder")
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length)
    else:
        # print("beam decoder")
        (decoded, log_prob) = tf.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=10,
            top_paths=2,
            merge_repeated=False)
    decoded_dense = [
        tf.sparse_to_dense(
            st.indices, st.dense_shape, st.values, default_value=-1)
        for st in decoded
    ]
    return (decoded_dense, log_prob)

def ctc_beam_decode(y_pred,input_length,greedy=True):
    y_pred_ph = tf.placeholder(tf.float32,shape=[None,None,None])
    input_length_ph = tf.placeholder(tf.int32,shape=[None])
    decoded_dense, _ = ctc_beam_op(y_pred_ph,input_length_ph,greedy)
    with tf.Session() as sess:
        dc =sess.run(decoded_dense,feed_dict={y_pred_ph: y_pred, input_length_ph: input_length})
    return dc

def cnn_lstm_ctc_pred_model(height,nclass, width=None):
    tensors = {
        'the_input':None, 'the_labels': None, 'input_length':None, 'label_length':None,'target_tensor':None
    }
        
    rnnunit  = 256
    inputs = Input(shape=(height,width,1),name='the_input', tensor=tensors['the_input'])
    m = cnn_feature_extractor(inputs)

    model_is_datetime = keras.models.Model([inputs], 
        [Activation('sigmoid')
            (Dense(1)(GlobalAveragePooling2D()(m)))])

    #2. bi-lstm layers
    m = Permute((2,1,3),name='permute')(m)
    m = TimeDistributed(Flatten(),name='timedistrib')(m)

    m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm1')(m)
    m = Dense(rnnunit,name='blstm1_out',activation='linear')(m)
    m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm2')(m)
    
    if not nclass == 3811:
        m = BatchNormalization()(m)
    y_pred = Dense(nclass,name='blstm2_out',activation='softmax')(m)
    
    #3. CTC loss compute 
    labels = Input(name='the_labels', shape=[None,], dtype='float32', tensor=tensors['the_labels'])
    input_length = Input(name='input_length', shape=[1], dtype='int64', tensor=tensors['input_length'])
    label_length = Input(name='label_length', shape=[1], dtype='int64', tensor=tensors['label_length'])
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.9)
    fn_compile = lambda: model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, metrics=['acc'],optimizer=adam, target_tensors=[tensors['target_tensor']])
    test_func = K.function([inputs], [y_pred, model_is_datetime.outputs[0]])
    return {'model':model, \
            'test_func':test_func, \
            'fn_compile':fn_compile, \
            'model_is_datetime':model_is_datetime}
