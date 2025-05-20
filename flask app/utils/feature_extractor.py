import cv2
import numpy as np
import keras
from keras import backend as K


alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

# preprocessing the image
def preprocess(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    (h, w) = img.shape
    
    final_img = np.ones([64, 256])*255 # blank white image
    
    # crop
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
    
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)


def decode_prediction(preds):
    decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                   greedy=True)[0][0])
    decoded_pred = decoded[0]
    return decoded_pred


def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret