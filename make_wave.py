import scipy.io.wavfile
import os

os.system('ffmpeg -y -i data/happy-whistling-ukulele.mp3 -acodec pcm_s16le -ar 44100 data/input.wav')
wav = scipy.io.wavfile.read('data/input.wav')[1]

with open('data/wave.txt', 'w') as f:
    for a in wav:
        print(a[0], a[1], file=f)
