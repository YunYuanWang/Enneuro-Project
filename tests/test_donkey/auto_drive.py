import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'code'))

import os
import cv2
import numpy as np
import gymnasium as gym
import gym_donkeycar

from model import ResNet18AutoDrive
from eneuro.utils.serializer import Serializer
from eneuro.base import Config
from eneuro.base.core import as_array


env = gym.make("donkey-generated-roads-v0")

obv = env.reset()

model = ResNet18AutoDrive()
script_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(script_dir, "results", "model.json")

serializer = Serializer()
serializer.load(model, model_folder)

action = np.array([0, 0.2])

frame, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated

for t in range(2500):
    img = frame.astype(np.float32) / 255.0
    img = img.transpose([2, 0, 1])
    img = np.expand_dims(img, axis=0)

    with Config.using_config("train", False):
        prelabel = model(img)
        steering_angle = as_array(prelabel[0, 0])

    factor = 1.5
    action = np.array([steering_angle * factor, 0.2])
    frame, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

obv = env.reset()
