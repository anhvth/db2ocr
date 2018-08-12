from tensorflow.python import keras
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.applications import *
from pyson.utils import *
from tf_dataset import *
K = keras.backend
from datetime import datetime
from libmodel import cnn_lstm_ctc_model
import argparse
import libmodel
from tqdm import tqdm
parse = argparse.ArgumentParser()

parse.add_argument('--max_length',  '-m', type=int, default=40,
                   help='the maximum number of prediction tobe made')
parse.add_argument('--text2label', '-t2l', default=None,
                   help='path to json file maping text and label')
parse.add_argument('--output_dir', required=True,
                   help='directory to save model')
parse.add_argument('--checkpoint', default=None,
                   help='directory to load model')
parse.add_argument('--height', type=int, default=48, help='Default height')
parse.add_argument('--epochs', type=int, default=100,
                   help='num of training epoch')
parse.add_argument('--model_choice', type=int, default=1,
                   choices=[1, 2, 3], help='1: cnn_lstm_ctc, 2: vgg_lstm_ctc, 3:vgg_cnn_ctc')
parse.add_argument('--freeze', type=bool, default=False,
                   help='freeze the feature extractor and retrain RNN-FC part')
parse.add_argument('--keep_checkpoint', type=int, default=None,
                   help='number of latest checkpoint tobe kept')
parse.add_argument('--batch_size', type=int, default=8,
                   help='Num of sample per training step')

args = parse.parse_args()

jp = lambda *args: os.path.join(*args)



def get_latest_checkpoint():
    sorted_paths_by_time = sorted(
        get_paths(args.checkpoint, 'h5'), key=lambda x: os.path.getmtime(x))
    return sorted_paths_by_time[-1]


DATASET_PATH = '../../../DATASET'
db_paths = get_paths('{}/synthetic_poc/*'.format(DATASET_PATH), 'json')

real_db_paths = get_paths(jp(DATASET_PATH, 'real_data/*/'), 'json')
paths, labels = load_multiple_db(db_paths)
paths_real, labels_real = load_multiple_db(real_db_paths)

imgs = [read_img(path) for path in paths[:5]]
vocabs = set()
for sentence in labels+labels_real:
    for word in sentence:
        vocabs.add(word)
print('vocab:', len(vocabs))


models = {1: libmodel.cnn_lstm_ctc_model}
downsample_factors = {1: 4,
                      2: 33,
                      3: 33}

# Look at checkpoint -> output_dir to see if any text2label.json, 
# IF not then create a text2label dictionary
if args.text2label is not None:
    print('INFO: Load text2label:', args.text2label)
    text2label = read_json(args.text2label)
elif args.checkpoint is not None and os.path.exists(os.path.join(args.checkpoint, 'text2label.json')):
    print('INFO: Load text2label:', args.checkpoint)
    text2label = read_json(os.path.join(args.checkpoint, 'text2label.json'))
elif os.path.exists(os.path.join(args.output_dir, 'text2label.json')):
    print('INFO: Load text2label:', args.output_dir)
    text2label = read_json(os.path.join(args.output_dir, 'text2label.json'))
else:
    print('INFO: create new text2label')
    text2label = {text: label for label, text in enumerate(vocabs)}

text2label[''] = -1

os.makedirs(args.output_dir, exist_ok=True)
with open(jp(args.output_dir, 'text2label.json'), 'w') as f:
    json.dump(text2label, f)

# Get Training generator
generator = get_generator(args.batch_size, real_db_paths, text2label,
                          real_ds_path=real_db_paths, ABS_MAX_LENGTH=args.max_length,
                           downsample_factor=downsample_factors[args.model_choice])
# get model configure dictionary
model_config = models[args.model_choice](args.height, len(text2label))
if args.model_choice == 1:
    if len(text2label) == 3811:
        model_config['model'].load_weights(
            '/mnt/DATA/anson/Dropbox/rnd_shared_workspace/ocr_model/chunhat.h5')
        print("INFO: LOADED model chunhat.h5")

    else:
        model_config['basemodel'].load_weights(
            '/mnt/DATA/anson/Dropbox/rnd_shared_workspace/ocr_model/chunhat_basemodel.h5')
        print("INFO: LOADED base model chunhat_basemodel.h5")

# Load pretrained model
if args.checkpoint is not None:
    latest_checkpoint = get_latest_checkpoint()
    print('LOADING MODEL FROM:', latest_checkpoint)
    model_config['model'].load_weights(latest_checkpoint)
    print('LOADED MODEL FROM:', latest_checkpoint)

if args.freeze:
    print("Freeze the base model")
    for layer in model_config['basemodel'].layers:
        layer.trainable = False

# Train model

model_config['fn_compile']()
model_config['model'].summary()

for e in range(args.epochs):
    model_config['model'].fit_generator(
        generator, epochs=2, steps_per_epoch=1000)
    ckpt_path = os.path.join(
        args.output_dir, '{}.h5'.format(str(datetime.now())))
    model_config['model'].save_weights(ckpt_path)
    if args.keep_checkpoint is not None:
        sorted_paths_by_time = sorted(
            get_paths(args.output_dir, 'h5'), key=lambda x: os.path.getmtime(x))
        if len(sorted_paths_by_time) > args.keep_checkpoint:
            for ckpt in sorted_paths_by_time[:-args.keep_checkpoint]:
                os.remove(ckpt)
    print('SAVED MODEL AT: ', ckpt_path)
    e += 1
