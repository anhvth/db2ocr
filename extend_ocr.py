from tensorflow.python.keras.layers import *
from tensorflow.python import keras
from libmodel import cnn_lstm_ctc_model, get_latest_checkpoint
from pyson.utils import * 

def get_generator(paths_db_true, paths_db_false, batch_size=1):
    paths_true, _ = load_multiple_db(paths_db_true)
    paths_false, _ = load_multiple_db(paths_db_false)
    while True:
        X, y = [], []
        for _ in range(batch_size):
            label = np.random.choice(2)
            if label == 0:
                path = np.random.choice(paths_true)
            else:
                path = np.random.choice(paths_false)
            img = read_img(path, is_gray=True, output_is_float=True)
            f = 48/img.shape[0]
            img = resize_by_factor(img, f)
            img = np.expand_dims(img, axis=-1)
            X = np.expand_dims(img, axis=0)
           
            y.append(label)
        yield X, y
        
def get_model_pred_datetime(inputs, base_model_output):
    '''
    '''
    #inputs = Input(tensor=features)

    #NICK
    base_model_output = GlobalAveragePooling2D()(base_model_output)
    x = Dense(1)(base_model_output)
    is_datetime = Activation('sigmoid')(x)

    #ANSON
    # x = LSTM(1, return_sequences=False)(base_model_output)
    # is_datetime = Activation('sigmoid')(x)
    
    return keras.models.Model([inputs], [is_datetime])


# GET MODEL
model_config = cnn_lstm_ctc_model(48, 3811)
path_model_weight = get_latest_checkpoint('frozen_model/ocr_3811/')
model = model_config['model']
for l in model.layers:l.trainable=False
model.load_weights(path_model_weight)

# GALE FETURE
# base_model_output= model_config['basemodel'].outputs[0] # --> 2d tensor. [Timestep, features]

# MY FEATURE
# base_model_output= model_config['cnn_model'].outputs[0] # --> 2d tensor. [Timestep, features]
# base_model_output  = Permute((2,1,3),name='permute')(base_model_output) 
# base_model_output = TimeDistributed(Flatten())(base_model_output)

# NICK FEATURE

base_model_output= model_config['cnn_model'].outputs[0] # --> 2d tensor. [Timestep, features]
#--------------
model_pred_datetime = get_model_pred_datetime(model.inputs[0], base_model_output) 

# GET DATA
DATASET_PATH = '../../../DATASET/'
db_paths_true = get_paths('{}/ngaythang/*'.format(DATASET_PATH),'json')
db_paths_false = get_paths('{}/synthetic_poc/*'.format(DATASET_PATH),'json')

db_paths_eval_true = get_paths('{}/real_data/ngaythang'.format(DATASET_PATH),'json')
db_paths_eval_false = get_paths('{}/real_data/ff_data_rnd_1'.format(DATASET_PATH),'json')

generator_eval = get_generator(db_paths_true, db_paths_false, 1)


generator = get_generator(db_paths_true, db_paths_false, 1)
# TRAIN MODEL
model_pred_datetime.summary()
model_pred_datetime.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_pred_datetime.fit_generator(generator, epochs=10, steps_per_epoch=1000)
model_pred_datetime.evaluate_generator(generator_eval, steps=1000)

