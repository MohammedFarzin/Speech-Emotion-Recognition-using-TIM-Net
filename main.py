import numpy as np
import os
import tensorflow as tf
import argparse
from model import TIMNET_Model

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--model_path', type=str, default='./Models/')
parser.add_argument('--result_path', type=str, default='./Results/')
parser.add_argument('--test_path', type=str, default='./Test_Models/EMODB_46')
parser.add_argument('--data', type=str, default='EMODB')
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--beta1', type=float, default=0.93)
parser.add_argument('--beta2', type=float, default=0.98)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=5)
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
print(args)

if args.data == 'IEMOCAP' and args.dilation_size != 10:
  args.dilation_size = 10

# allowing dynamic gpu memory growth
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpus = tf.config.list_physical_devices(device_type='GPU')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
print(session)
print(f'GPU:{gpus}')

CLASS_LABELS_finetune = ('angry', 'fear', 'happy', 'neutral', 'sad')
CASIA_CLASS_LABELS = ('angry', 'fear', 'happy', 'neutral', 'sad', 'surprise')
EMODB_CLASS_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad")
SAVEE_CLASS_LABELS = ("angry","disgust", "fear", "happy", "neutral", "sad", "surprise")
RAVDE_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise")
IEMOCAP_CLASS_LABELS = ("angry", "happy", "neutral", "sad")
EMOVO_CLASS_LABELS = ("angry", "disgust", "fear", "happy","neutral","sad","surprise")
CLASS_LABELS_dict = {
    "CASIA" : CASIA_CLASS_LABELS,
    "EMODB" : EMODB_CLASS_LABELS,
    "SAVEE" : SAVEE_CLASS_LABELS,
    "RAVDE" : RAVDE_CLASS_LABELS,
    "IEMOCAP" : IEMOCAP_CLASS_LABELS,
    'EMOVO' : EMOVO_CLASS_LABELS
}

data = np.load('./MFCC/'+args.data+'.npy', allow_pickle=True).item()
x_source = data["x"]
y_source = data["y"]
CLASS_LABELS = CLASS_LABELS_dict[args.data]

model = TIMNET_Model(args=args, input_shape=x_source.shape[1:], class_label=CLASS_LABELS)
if args.mode == 'train':
  model.train(x_source, y_source)
elif args.mode == 'test':
  x_feats, y_labels = model.test(x_source, y_source, path=args.test_path)





