from ocr import OCR
import os
import sys
sys.path.insert(0, '../src/')
from utils.word_list import WordList
from utils.TextImageGenerator import render_line
from nltk.metrics.distance import edit_distance
from PIL import Image
import numpy as np
import time
import xlsxwriter
import pickle
import matplotlib.pyplot as plt
import itertools
import io
from keras import backend as K
from pyson.utils import *
K.set_learning_phase(1)
import cv2

class Eval(object):
    def __init__(self,model):
        self.model = model
    def evaluate(self,data,label,batch_size=10):
        error_rate = 0.0
        num_sample = data.shape[0]
        num_batch = int(num_sample/batch_size)
        for batch in range(num_batch):
            x_batch = data[batch*batch_size:(batch*batch_size + batch_size)]
            y_label = label[batch*batch_size:(batch*batch_size + batch_size)]
            pred = self.model.predict(x_batch)
            # print(y_label)
            # print(pred)
            error_rate += self.error_rate(y_label,pred,batch_size)
        return error_rate/num_batch

    def evaluate_with_tess(self,data,label):
        num_samples = data.shape[0]
        error_rate = 0.0
        for i in range(num_samples):
            pil_img = Image.fromarray(data[i,:,:,0] * 255)
            # pil_img.convert('RGB').save("{}.png".format(i))
            tess_label = pytesseract.image_to_string(pil_img, lang='jpn_ori',config='-psm 6')
            error_rate += self.error_rate(label[i], tess_label, 1)
            # if i % 10 == 0:
            # print(label[i])
            # print(tess_label)
        return error_rate/num_samples
    def error_rate(self,label,pred,batch_size):
        label = np.array(label).reshape(-1,)
        pred = np.array(pred).reshape(-1,)
        mean_ed = 0.0
        for i in range(batch_size):
            # print(float(edit_distance(pred[i].replace(" ", ""), label[i].replace(" ", ""))))
            # print(len(label[i]))
            mean_ed += float(edit_distance(pred[i].replace(" ", ""), label[i].replace(" ", "")))/len(label[i])
            # print(mean_ed)
        mean_ed /= batch_size
        return mean_ed


    def export_report(self,data,label,tesseract=False,google=False):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        workbook = xlsxwriter.Workbook(os.path.join('report','ocr_report_{}.xlsx'.format(timestr)))
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Image')
        worksheet.write('B1', 'RnD OCR')
        worksheet.write('D1', 'LABEL')
        worksheet.write('E1', 'RnD OCR ERROR RATE')
        worksheet.write('G1', 'TOTAL RnD OCR ACCURACY RATE (%)')
        worksheet.write('H1', 'TOTAL Google API ACCURACY RATE (%)')
        total_rnd_err = 0
        total_google_err = 0
        for i in range(len(data)):
            h,w = data[i].shape
            img_4d = data[i].reshape(-1,h,w,1)
            if np.max(img_4d) > 1:
                img_4d = img_4d/255
            pred = self.model.predict(img_4d,greedy=False)[0]
            rnd_model_err =  self.error_rate(label[i],pred,1)
            total_rnd_err += rnd_model_err
            worksheet.write('B{}'.format(i + 2), pred)
            worksheet.write('E{0:.4f}'.format(i + 2), rnd_model_err)
            plt.imsave(os.path.join("eval_img","{}.png").format(i),data[i][:,:],cmap='gray')
            if google:
                google_pred = google_vision(os.path.join("eval_img","{}.png").format(i))
                google_err = self.error_rate(label[i], google_pred, 1)
                worksheet.write('C{}'.format(i + 2), google_pred)
                worksheet.write('F{0:.4f}'.format(i + 2), google_err)
                total_google_err += google_err
            worksheet.write('D{}'.format(i + 2), label[i])
            worksheet.insert_image('A{}'.format(i+2),os.path.join("eval_img","{}.png".format(i)),{'x_scale': 0.4, 'y_scale': 0.4})
        total_rnd_err /= len(data)
        total_google_err /= len(data)
        total_rnd_correct = (1.0 - total_rnd_err)*100
        total_google_correct = (1.0 - total_google_err)*100
        worksheet.write('G2', total_rnd_correct)
        worksheet.write('H2', total_google_correct)
        workbook.close()

    def export_report_jp(self,data,label,tesseract=False,google=False):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        workbook = xlsxwriter.Workbook(os.path.join('report','ocr_report_{}.xlsx'.format(timestr)))
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Image')
        worksheet.write('B1', 'RnD OCR')
        worksheet.write('D1', 'LABEL')
        worksheet.write('E1', 'RnD OCR ERROR RATE')
        worksheet.write('G1', 'TOTAL RnD OCR ACCURACY BY CHAR')
        worksheet.write('H1', 'TOTAL Google API ACCURACY BY CHAR')
        worksheet.write('I1', 'TOTAL RnD OCR ACCURACY BY FIELD')
        worksheet.write('J1', 'TOTAL Google API ACCURACY BY FIELD')
        total_rnd_acc = 0
        total_google_acc = 0
        total_length = 0
        total_rnd_field_acc = 0
        total_google_field_acc = 0
        total_field = 0
        total_rnd_fscore = 0
        total_google_fscore = 0
        if google:
            worksheet.write('C1', 'Google Vision API')
            worksheet.write('F1', 'Google Vision API ERROR RATE')
        for i in range(len(data)):
            h,w = data[i].shape
            img_4d = data[i].reshape(-1,h,w,1)
            if np.max(img_4d) > 1:
                img_4d = img_4d/255
            pred = self.model.predict(img_4d,greedy=True)[0]
            rnd_model_err =  self.error_rate(label[i],pred,1)
            total_rnd_acc += (1.0 - rnd_model_err)*len(label[i])
            if (rnd_model_err - 0) == 0.0:
                total_rnd_field_acc += 1
            total_field += 1
            total_length += len(label[i])
            worksheet.write('B{}'.format(i + 2), pred)
            worksheet.write('E{0:.4f}'.format(i + 2), rnd_model_err)
            plt.imsave(os.path.join("eval_img","{}.png").format(i),data[i][:,:],cmap='gray')
            if google:
                google_pred = google_vision(os.path.join("eval_img","{}.png").format(i))
                google_err = self.error_rate(label[i], google_pred, 1)
                worksheet.write('C{}'.format(i + 2), google_pred)
                worksheet.write('F{0:.4f}'.format(i + 2), google_err)
                if(google_err - 0) == 0.0:
                    total_google_field_acc += 1
                total_google_acc += (1.0 - google_err)*len(label[i])
            worksheet.write('D{}'.format(i + 2), label[i])
            worksheet.insert_image('A{}'.format(i+2),os.path.join("eval_img","{}.png".format(i)),{'x_scale': 0.4, 'y_scale': 0.4})
        total_rnd_acc /= total_length
        total_google_acc /= total_length
        total_rnd_field_acc /= total_field
        total_google_field_acc /= total_field
        worksheet.write('G2', total_rnd_acc)
        worksheet.write('H2', total_google_acc)
        worksheet.write('I2', total_rnd_field_acc)
        worksheet.write('J2', total_google_field_acc)
        workbook.close()

def load_model(model_dir, nclass):
    model_path = get_latest_model(model_dir)
    json_path = os.path.join(model_dir, 'label.json') 
    label2text = read_json(json_path)
    return model_path, label2text


if __name__=='__main__':
    weights_path = '/mnt/DATA/anson/Dropbox/rnd_shared_workspace/ocr_model/chunhat.h5'#sorted(get_paths('anson_ckpt/retrain_lstm_fc/', 'h5'))[-1]
    
    label2text = {int(k):v for k, v in read_json('json/label_text.json').items()}
    word2int = {v:k for k, v in label2text.items()}
    my_ocr = OCR(weights_path, label2text, 3811)
    eval_ocr = Eval(my_ocr)
    
    eval_data = []
    paths, labels = load_db('../notebook/db/special_font/label.json')
    
    for path, label in zip(paths, labels):
        img = cv2.imread(path, 0)
        f = 48/img.shape[0]
        img = resize_by_factor(img, f)
        eval_data.append((img, str(label)))
    X = []
    label = []
    for data in eval_data[:10]:
        x_data, y_data = data
        X.append(x_data)
        label.append(word_gen._normalize_text_line(y_data))
    eval_ocr.export_report_jp(X,label,google=False)
