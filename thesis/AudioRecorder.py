import pyaudio
import wave
import threading
import settings #file with global variables
from SpeechEmotionRecognitionModel import SpeechEmotionRecognitionModel

audioPrediction = SpeechEmotionRecognitionModel(path='Emotion_Voice_Detection_Model.h5', file='temp_audio.wav')

class AudioRecorder():
    "Audio class based on pyAudio and Wave"
    def __init__(self, filename="temp_audio.wav", rate=16000, fpb=1, channels=2, record_seconds=5): # rate=16000 è fondamentale per farlo funzionare sulla mia webcam.
        self.open = True
        self.rate = rate
        self.frames_per_buffer = fpb
        self.channels = channels
        self.format = pyaudio.paInt16
        self.audio_filename = filename
        self.record_seconds = record_seconds
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []

    def record(self):
        "Audio starts being recorded"
        self.stream.start_stream()
        for i in range(0, int(self.rate / self.frames_per_buffer * self.record_seconds)):
            data = self.stream.read(self.frames_per_buffer)
            self.audio_frames.append(data)
        self.stop()

    def stop(self):
        "Finishes the audio recording therefore the thread too"
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        waveFile = wave.open(self.audio_filename, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.format))
        waveFile.setframerate(self.rate)
        waveFile.writeframes(b''.join(self.audio_frames))
        waveFile.close()

        audioPrediction.makepredictions()
        settings.audio_emotion = audioPrediction.makepredictions()


    def start(self):
        "Launches the audio recording function using a thread"
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()
        return audio_thread
