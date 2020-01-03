from __future__ import print_function, division
import os
import settings
from decisionTreeThreeVariables import decisionTreeThreeVariables
from decisionTreeAllVariables import decisionTreeAllVariables
from VideoRecorder import VideoRecorder
from AudioRecorder import AudioRecorder
from random import randint
from gtts import gTTS
from playsound import playsound



N_EPISODES = 4
LEN_THIRD_VARIABLE = 3 # number of possible values for the third feature (Object State) are 3
LEN_FOURTH_VARIABLE = 5 # number of possible values for the fourth feature (Environmental Noise) are 5


def file_manager(filename="test"):
    "Required and wanted processing of final files"
    local_path = os.getcwd()
    if os.path.exists(str(local_path) + "/temp_audio.wav"):
        os.remove(str(local_path) + "/temp_audio.wav")
    if os.path.exists(str(local_path) + "/temp_video.avi"):
        os.remove(str(local_path) + "/temp_video.avi")


def preprocess():
    third_feature = randint(0, LEN_THIRD_VARIABLE-1)
    fourth_feature = randint(0, LEN_FOURTH_VARIABLE-1)
    while ((third_feature == 1 and fourth_feature == 4) or (third_feature == 2 and fourth_feature == 4)):
        third_feature = randint(0, LEN_THIRD_VARIABLE-1)
        fourth_feature = randint(0, LEN_FOURTH_VARIABLE-1)
    return third_feature, fourth_feature

def videoPreProcess(video_output):
    if (video_output == "Neutral"):
        video_output = 0
    if (video_output == "Surprise" or video_output == "Happy"):
        video_output = 1
    if (video_output == "Angry" or video_output == "Disgust" or video_output == "Fear" or video_output == "Sad"):
        video_output = 2
    return video_output

def audioPreProcess(audio_output):
    if (audio_output == "calm" or audio_output == "neutral"):
        audio_output = 0
    if (audio_output == "surprised" or audio_output == "happy"):
        audio_output = 1
    if (audio_output == "angry" or audio_output == "fearful" or audio_output == "disgust" or audio_output == "sad"):
        audio_output = 2
    return audio_output


if __name__ == '__main__':

    settings.init() # set global variables from file settings.py
    i = 0
    #solver = decisionTreeThreeVariables()
    solver = decisionTreeAllVariables()
    solver.num_run = 1
    solver.buildTree(True)

    states4 = [(2, 2, 1, 2), (2, 0, 0, 1), (0, 0, 2, 0), (1, 0, 1, 3), (1, 1, 1, 1)]
    states3 = [(2, 1, 2), (2, 0, 1), (0, 2, 0), (1, 1, 3), (1, 1, 1)]

    while (i < len(states3)):
        video_thread = VideoRecorder()
        audio_thread = AudioRecorder()
        #a = audio_thread.start()
        b = video_thread.start()
        #a.join()
        b.join()
        video_output = settings.video_emotion
        print(video_output)
        #audio_output = sings.audio_emotion
        video_output = videoPreProcess(video_output)
        #audio_output = audioPreProcess(audio_output)
        state = states4[i]#(video_output, audio_output, third_feature, fourth_feature)
        solver.env.state = state
        #solver.env.getState()
        action = solver.choose_action(state, 0)
        action = solver.env.getAction(action)

        # Text-Speech
        myobj = gTTS(text=action, lang='en', slow=False)
        # Saving the converted audio in a mp3 file named action
        myobj.save("action.mp3")
        # Playing the converted file
        playsound('action.mp3')

        print("Observation n. " + str(i))
        i += 1
    file_manager()