import pygame
import math
import sys
import random
import cv2
import numpy as np
import os
import time

from PathPlanning.trajectory import quick_visual_check

dir = "data"


fpath_path = os.path.join(dir, "path{}{}.npy")
fpath_resampled = os.path.join(dir, "resample{}{}.npy")
fpath_traj = os.path.join(dir, "traj{}{}.npy")

for planner in ["A", "R"]:
    for phase in [1, 2]:

        path = np.load(fpath_path.format(planner, phase))
        resampled = np.load(fpath_resampled.format(planner, phase))
        traj = np.load(fpath_traj.format(planner, phase))

        quick_visual_check(path, resampled, traj)