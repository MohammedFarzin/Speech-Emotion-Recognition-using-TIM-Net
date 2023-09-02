import numpy as np
import os
import tensorflow as tf
import argparse
from model import TIMNET_Model
import librosa
import joblib


parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--model_path', type=str, default='./Models/')
parser.add_argument('--result_path', type=str, default='./Results/')
parser.add_argument('--test_path', type=str, default='./Test_Models/RAVDE_46')
parser.add_argument('--data', type=str, default='RAVDE')
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--beta1', type=float, default=0.93)
parser.add_argument('--beta2', type=float, default=0.98)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--random_seed', type=int, default=46)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--filter_size', type=int, default=39)
parser.add_argument('--dilation_size', type=int, default=8)
parser.add_argument('--kernel_size', type=int, default=2)
parser.add_argument('--stack_size', type=int, default=1)
parser.add_argument('--split-fold', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0')

args, unknown = parser.parse_known_args()

if args.data == 'IEMOCAP' and args.dilation_size != 10:
  args.dilation_size = 10

# allowing dynamic gpu memory growth
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print(os.environ['CUDA_VISIBLE_DEVICES'])
print(tf.__version__)
gpus = tf.config.list_physical_devices(device_type='GPU')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# CLASS_LABELS_finetune = ('angry', 'fear', 'happy', 'neutral', 'sad')
# CASIA_CLASS_LABELS = ('angry', 'fear', 'happy', 'neutral', 'sad', 'surprise')
# EMODB_CLASS_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad")
SAVEE_CLASS_LABELS = ("angry","disgust", "fear", "happy", "neutral", "sad", "surprise")
RAVDE_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise")
# IEMOCAP_CLASS_LABELS = ("angry", "happy", "neutral", "sad")
# EMOVO_CLASS_LABELS = ("angry", "disgust", "fear", "happy","neutral","sad","surprise")
CLASS_LABELS_dict = {
    # "CASIA" : CASIA_CLASS_LABELS,
    # "EMODB" : EMODB_CLASS_LABELS,
    "SAVEE" : SAVEE_CLASS_LABELS,
    "RAVDE" : RAVDE_CLASS_LABELS,
    # "IEMOCAP" : IEMOCAP_CLASS_LABELS,
    # 'EMOVO' : EMOVO_CLASS_LABELS
}

data = np.load('./MFCC/'+args.data+'.npy', allow_pickle=True).item()
x_source = data["x"]
y_source = data["y"]
CLASS_LABELS = CLASS_LABELS_dict[args.data]
print(x_source.shape)
model = TIMNET_Model(args=args, input_shape=x_source.shape[1:], class_label=CLASS_LABELS)
# if args.mode == 'train':
#   model.train(x_source, y_source)
# elif args.mode == 'test':
#   x_feats, y_labels = model.test(x_source, y_source, path=args.test_path)

import librosa
import numpy as np
def get_feature(file_path: str, mfcc_len: int = 39, mean_signal_length: int = 110000):
    signal, fs = librosa.load(file_path)
    s_len = len(signal)
    print(fs)
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values = 0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=39)
    mfcc = mfcc.T
    feature = mfcc
    return feature





file_path = r"D:\Machine learning\Project\Speech Recognition System\SER_dj\ser_dj\timnet\Dataset\RAVDE\Actor_02\03-01-01-01-01-01-02.wav"
audio = get_feature(file_path)
weight_path = "./Test_Models/RAVDE_46/10-fold_weights_best_4.hdf5"
audio = np.expand_dims(audio, axis=0)
model.create_model()
model.model.load_weights(weight_path)
y_pred = model.model.predict(audio)
y_pred = np.argmax(y_pred)
print(y_pred)


# audio_file = "C:/Users/FARZIN/Downloads/Movie quote from BLADE RUNNER 2049 - Sometimes, to love someone, you gotta be a stranger.wav"
# audio_feature = get_feature(audio_file)

# model.create_model()
# weight_path = args.model_path + args.data + "_46/10-fold_weights_best_4.hdf5"
# model.model.load_weights(weight_path)
# emotion_prob = model.model.predict(audio_feature) 
# emotion_label = CLASS_LABELS_dict[args.data][np.argmax(emotion_prob)] 
# print(f"The predicted emotion for {audio_file} is {emotion_label}.")