# Project Title

Machine Learning for Social Assistive Robots

## Description

<b>Case study</b>: physiotherapy for children during recovery from an injury.

<b>Problem</b>: lack of focus and disengagement of children in the physiotherapy.

<b>Project goal</b>: develop an agent that can assist therapists during children’s therapies.

The project is structured in two modules:
- Supervised Learning (extracting features from sensors)
- Reinforcement Learning (making decisions)

<p align="center">
  <img width="690" height="315" src="https://i.ibb.co/3khCZj7/Cattura.jpg">
</p>

## Components

- Environment Simulator based on OpenAI Gym - [gym_foo](gym_foo)
- Supervised Learning module: [AudioRecorder.py](thesis/AudioRecorder.py) and [VideoRecorder.py](thesis/VideoRecorder.py) respectively record audio and video clips subsequently classified using the pre-trained CNNs [FER - Facial Expression Recognition](https://github.com/mayurmadnani/fer) and [Emotion-Classification-Ravdess](https://github.com/marcogdepinto/Emotion-Classification-Ravdess).
- Reinforcement Learning module: [decisionTreeAllVariables.py](decisionTreeAllVariables.py) and [decisionTreeThreeVariables.py](decisionTreeThreeVariables.py) implement the learning agent considering respectively all features (Face Expression Recognition, Speech Emotion Recognition, Object State, Environmental Sound) characterizing the state and the three most important.

## Requirements
- tensorflow == 1.15.0
- keras == 2.2.5
- cv2 == 4.1.2
- pyaudio == 0.2.11
- wavy == 1.0.1
- numpy == 1.17.4
- librosa == 0.7.1

## Installation
You can run this application just installing the custom environment:
```
$ pip install -e .
```
and running the [main.py](thesis/main.py):
'''
$ python main.py
'''







