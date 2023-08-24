from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
import pyaudio
import wave
import librosa
import numpy as np
from .models import BankDetails, Feedback
from timnet.model import TIMNET_Model
import os
import tensorflow as tf
import argparse
import librosa
import joblib

# Create your views here.

def home(request):
    # bank_details = BankDetails.objects.all()
    bank_feedback = Feedback.objects.all()
    context = {
        # 'bank_details' : bank_details,
        'bank_feedback' : bank_feedback        
    }
    return render(request, 'index.html', context)


def feedback(request, bank_slug):
    
    bank = get_object_or_404(BankDetails, slug=bank_slug)
    feedbacks = Feedback.objects.filter(bank_id = bank.id)
    print(feedbacks)
    context = {
        'bank' : bank,
        'feedbacks' : feedbacks
    }
    return render(request, 'feedback.html', context)


def emotion_generation(request):
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

    # allowing dynamic gpu memory growth
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    print(tf.__version__)
    gpus = tf.config.list_physical_devices(device_type='GPU')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    RAVDE_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise")
    CLASS_LABELS_dict = {
        "RAVDE" : RAVDE_CLASS_LABELS
    }

    data = np.load(r"D:\Machine learning\Project\Speech Recognition System\SER_dj\ser_dj\timnet\MFCC\RAVDE.npy", allow_pickle=True).item()
    x_source = data["x"]
    y_source = data["y"]
    CLASS_LABELS = CLASS_LABELS_dict[args.data]
    print(x_source.shape)
    model = TIMNET_Model(args=args, input_shape=x_source.shape[1:], class_label=CLASS_LABELS)





    file_path = r"D:\Machine learning\Project\Speech Recognition System\SER_dj\ser_dj\output.wav"
    audio = get_feature(file_path)
    weight_path = "D:\Machine learning\Project\Speech Recognition System\SER_dj\ser_dj\timnet\Test_Models\RAVDE_46\10-fold_weights_best_4.hdf5"
    audio = np.expand_dims(audio, axis=0)
    model.create_model()
    model.model.load_weights(weight_path)
    y_pred = model.model.predict(audio)
    y_pred = np.argmax(y_pred)
    return y_pred







def voice_record(request, feedback_id, CHUNK = 1024, FORMAT = pyaudio.paInt16, CHANNELS = 1, RATE = 22050):
    url = request.META.get('HTTP_REFERER')
    print(url)
    if request.method == 'POST':
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("start recording....")

        frames = []
        seconds = 5
        for i in range(0, int(RATE / CHUNK * seconds)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        print('Start stopped.')

        stream.stop_stream()
        stream.close()
        p.terminate() 

        wf = wave.open('output.wav', 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        model_emotion = emotion_generation(request)
        emotion = None
        if model_emotion == 0:
            emotion = 'angry'
        elif model_emotion == 1:
            emotion = 'calm'
        elif model_emotion == 2:
            emotion = 'disgust'
        elif model_emotion == 3:
            emotion = 'fear'
        elif model_emotion == 4:
            emotion = 'happy'
        elif model_emotion == 5:
            emotion = 'neutral'
        elif model_emotion == 6:
            emotion = 'sad'
        elif model_emotion == 7:
            emotion = 'surprise'
        else:
            emotion = None
        return HttpResponse("Recording done")


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

# def single_bank_details(request, bank_slug):
#     single_bank = BankDetails.objects.get(slug = bank_slug)

#     context = {
#         'single_bank_details' : single_bank
#     }

#     return render


