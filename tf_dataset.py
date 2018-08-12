import tensorflow as tf
import os
from pyson.utils import *

NORMALIZE_TEXT = read_json('./json/text_normalize.json')
MAX_STRING_LENGTH = 40

def normalize_text(text_line):
    for w in text_line:
        if w in NORMALIZE_TEXT:
            text_line = text_line.replace(w,NORMALIZE_TEXT[w])
    return text_line

def get_tensors(generator, sess):
    def fn_parse_gen():
        while True:
            inputs, outputs = next(generator)
            yield inputs["the_input"], inputs["the_labels"], inputs[
                "input_length"
            ], inputs["label_length"], outputs["ctc"]

    dataset = (
        tf.data.Dataset()
        .from_generator(
            fn_parse_gen,
            output_types=tuple([tf.float32] * 5),
            output_shapes=(
                tf.TensorShape([None, 48, None, 1]),
                tf.TensorShape([None, MAX_STRING_LENGTH]),
                tf.TensorShape([None, 1]),
                tf.TensorShape([None, 1]),
                tf.TensorShape([None,]),
            ),
        )
    )
    dataset = dataset.prefetch(128)
    iter = dataset.make_initializable_iterator()
    a, b, c, d, e = iter.get_next()
    sess.run(iter.initializer)
    return {
        "the_input": a,
        "the_labels": b,
        "input_length": c,
        "label_length": d,
        "target_tensor": e,
        "iter": iter,
    }



def read_process(path, img_h=48):
    assert os.path.exists(path), "{} doesn't exist".format(path)
    img = read_img(path, is_gray=True, output_is_float=True)
    if img.shape[0] != img_h:
        f =  img_h/img.shape[0]
        img = resize_by_factor(img, f)
    return img
def get_generator(batch_size, 
                  db_path, 
                  text2label, 
                  img_h = 48, 
                  max_sample=-1, 
                  ABS_MAX_LENGTH=40, 
                  mode='train', 
                  shuffle=True, 
                  real_ds_path=None, 
                  downsample_factor=4,
                  real_ratio=4):
    def load_data(db_path):
        if type(db_path) is list:
            img_paths, db_source_strs = load_db(db_path[0], ABS_MAX_LENGTH)
            for path in db_path[1:]:
                _ = load_db(path)
                img_paths += _[0]
                db_source_strs += _[1]
        else:
            img_paths, db_source_strs = load_db(db_path, ABS_MAX_LENGTH)
        return img_paths, db_source_strs
    
    db_paths, db_source_strs = load_data(db_path)
    db_paths = db_paths[:max_sample]
    db_source_strs = db_source_strs[:max_sample]
    print('Len of samples: ', len(db_paths))
    if real_ds_path is not None:
        db_paths_real, db_source_strs_real = load_data(real_ds_path)
        

    
    def convert_to_batch(imgs):
        shapes = np.array([img.shape[:2] for img in imgs])
        h_max, w_max = shapes[:,0].max(), shapes[:,1].max()
        w_max = max(256*4, w_max)
        pad = np.ones(shape=[len(imgs), h_max, w_max, 1])
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            pad[i,:h,:w,0] = img
        return pad
    
    def pick_from_db(db_paths, db_source_strs):
        idx = np.random.choice(len(db_paths))
        path = db_paths[idx]
        s = str(db_source_strs[idx])
        if len(s) < MAX_STRING_LENGTH:
            img = read_process(path)
        else:
            return pick_from_db(db_paths, db_source_strs)
        return img, s, path
    def source2label(source):
        label = np.ones([ABS_MAX_LENGTH])*-1
        source = normalize_text(source)
        assert len(source) < len(label), '{}<{}'.format(len(source), len(label))
        for i, s in enumerate(source):
            if s in text2label.keys():
                label[i] = text2label[s]
            else:
                label[i] = text2label[' ']
        return label, len(source)
    if mode == 'train':
        while True:
            input_path = []
            X_data = []
            source_str = []
            
            while len(X_data) < batch_size:
                chance = np.random.uniform() < real_ratio
                if real_ds_path is not None and chance:
                    img, s, path = pick_from_db(db_paths_real, db_source_strs_real)
                else:
                    img, s, path = pick_from_db(db_paths, db_source_strs)
                X_data.append(img)
                source_str.append(str(s))
                input_path.append(path)
                
            labels = np.ones([batch_size, ABS_MAX_LENGTH]) * -1
            input_length = np.zeros([batch_size, 1])
            label_length = np.zeros([batch_size, 1])
                
            X_data = convert_to_batch(X_data)
            batch_img_w = X_data.shape[2]
            i = 0 
            for i in range(len(X_data)):
                lbl_i, lbl_len_i = source2label(source_str[i])

                labels[i, :] = lbl_i
                label_length[i] = lbl_len_i
                input_length[i] = int(batch_img_w // downsample_factor) - 1
            print('input_length:', input_length[-1])

            inputs = {
                "the_input": X_data,
                "the_labels": labels,
                "input_length": input_length,
                "label_length": label_length,
                "source_str": source_str,  # used for evaluation only
                "input_path":input_path
            }

            outputs = {
                "ctc": np.zeros([batch_size])
            }
            
            yield inputs, outputs
    elif mode=='test':
        for k in range(0, len(db_paths), batch_size):
            X_data = []
            input_path = []
            source_str = []
            
            for idx in range(k, min(k+batch_size, len(db_paths)), 1):
                s = str(db_source_strs[idx])
                path = db_paths[idx]
                if len(s) < MAX_STRING_LENGTH:
                    img = read_process(path)
                    X_data.append(img)
                    source_str.append(s)
                    input_path.append(path)
                    
            labels = np.ones([len(X_data), ABS_MAX_LENGTH]) * -1
            input_length = np.zeros([len(X_data), 1])
            label_length = np.zeros([len(X_data), 1])
            
            X_data = convert_to_batch(X_data)
            batch_img_w = X_data.shape[2]
            for i in range(len(X_data)):
                lbl_i, lbl_len_i = source2label(source_str[i])
                labels[i, :] = lbl_i
                label_length[i] = lbl_len_i
                input_length[i] = int(batch_img_w // 4) - 1

            inputs = {
                "the_input": X_data,
                "the_labels": labels,
                "input_length": input_length,
                "label_length": label_length,
                "source_str": source_str,  # used for evaluation only
                "input_path":input_path
            }

            outputs = {
                "ctc": np.zeros([len(X_data)])
            }
            
            yield inputs, outputs



