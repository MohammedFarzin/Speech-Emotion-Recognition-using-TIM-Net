from django.shortcuts import render
from django.http import HttpResponse
import pyaudio
import wave

# Create your views here.

def welcome(request):
    return render(request, 'index.html')


def feedback(request):
    if request.method == 'POST':
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 22050

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



    
    return render(request, 'feedback.html')

def voice_record(request, CHUNK = 1024, FORMAT = pyaudio.paInt16, CHANNELS = 1, RATE = 22050):
    print(request.method)
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
    return HttpResponse("Recording done")