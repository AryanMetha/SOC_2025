'''
LunarLander-v3 with keyboard control
'''

__credits__ = ["Andrea PIERRÉ"]

import math
from typing import TYPE_CHECKING

import pygame

import numpy as np

import gymnasium as gym

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,

               enable_wind=True, wind_power=10.0, turbulence_power=0,render_mode='human')

env.reset()

pygame.init()
screen = pygame.display.set_mode((400, 300))

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        action = 1
    elif keys[pygame.K_DOWN]:
        action = 2
    elif keys[pygame.K_RIGHT]:
        action = 3
    else:
        action = 0

    state,reward,terminated,truncated,info = env.step(action)
    env.render()
    print(info)
    if terminated or truncated:
        running = False
        state = env.reset()
        if reward > 0:
            print("You win!")
        else:
            print("You lose!")
    print(f"Action: {action}, Reward: {reward}")  
    pygame.time.delay(100)
