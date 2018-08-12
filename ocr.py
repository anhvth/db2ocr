import libmodel
import os
import sys
import numpy as np
import timeit
from libmodel import cnn_lstm_ctc_model, ctc_beam_decode, cnn_lstm_ctc_pred_model
from pyson.utils import *
class OCR(object):
    def __init__(self,weights_path,label_text,nclass):
        self.weights_path = weights_path
        self.label_text = label_text
        self.nclass = nclass
        self._build_graph()

    def _build_graph(self):
        model, basemodel, test_func, fn_compile = cnn_lstm_ctc_model(48, self.n_class, tensors=None)
        fn_compile()
        self.model = basemodel
        model.load_weights(self.weights_path)
        print("Model Loaded !!!")
        self.model.summary()

    def labels_to_text(self,labels):
        ret = []
        for c in labels:
            if c == len(self.label_text) or c==-1:  # CTC Blank
                ret.append("")
            else:
                ret.append(self.label_text[c])
        return "".join(ret)

    def predict(self,X,greedy=False):
        
        out = self.model.predict(X)
        dec = ctc_beam_decode(out,[out.shape[1]]*out.shape[0],greedy=greedy)[0]
        ret = []
        for j in range(len(dec)):
            out_best = dec[j]

            outstr = self.labels_to_text(out_best)

            ret.append(outstr)
        return ret

from tf_dataset import read_process
def read_1_image(path):
    x = read_process(path)
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, -1)
    return x
def pred(path):
    _, is_datetime = m_ocr['test_func']([read_1_image(path)])
    return is_datetime
if __name__=='__main__':
    weight_path_ocr = 'frozen_model/ocr_3811/chunhat.h5'
    weight_path_is_datetime = 'frozen_model/is_ngaythang.h5'
    weight_path_datetime = 'frozen_model/date_time/2018-08-07 00:36:36.970666.h5'
    DATASET = '../../../DATASET'

    db_paths_false = get_paths('{}/real_data/ff_data_rnd_1'.format(DATASET), 'json')
    db_paths_true = get_paths('{}/real_data/ngaythang/'.format(DATASET), 'json')

    paths_false, _ = load_multiple_db(db_paths_false)
    paths_true, _ = load_multiple_db(db_paths_true)


    m_ocr=cnn_lstm_ctc_pred_model(48, 3811)
    m_ocr['model'].load_weights(weight_path_ocr)
    m_ocr['model_is_datetime'].load_weights(weight_path_is_datetime)


    # weights_path = './checkpoint/weights_daichi.h5'
    # training_path = os.path.join("../src/utils/corpus", "jpn.training_text.bigram_freqs")
    # monogram_path = os.path.join("../src/utils/corpus", "jpn.training_text.unigram_freqs")
    # poc_text = os.path.join("../src/utils/corpus", "poc")
    # latin_number_text = os.path.join("../src/utils/corpus", "latin_number")
    # word_gen = WordList(training_path, monogram_path, poc_text, latin_number_text)
    # labels_text = word_gen.get_label_to_text()
    # my_ocr = OCR(weights_path, labels_text, word_gen.get_nclass() + 1)
    # #generate sample image for testing purposes
    # img_h = 48
    # img_w = img_h * 40
    # X = []
    # for i in range(2):
    #     sample_line, _,_ = word_gen.get_line(10, True)
    #     print(sample_line)
    #     img = render_line(sample_line, img_w, img_h)
    #     X.append(img)
    # X = np.array(X)
    # c,h,w = X[0].shape
    # X = X.reshape(-1,h,w,c)
    # start = timeit.default_timer()
    # print(my_ocr.predict(X,greedy=True))
    # stop = timeit.default_timer()
    # print("running time: {}".format(stop - start))