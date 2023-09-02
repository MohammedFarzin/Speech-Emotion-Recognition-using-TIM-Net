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
