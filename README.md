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
  






