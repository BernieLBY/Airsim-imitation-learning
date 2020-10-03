import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import tempfile
from pynput import keyboard

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0
client.reset()
max_speed=7
steer_right=0.4
steer_left=-0.4

def on_press(key):
    if (str(key) == 'Key.up'):
        car_state = client.getCarState()
        if (car_state.speed < max_speed):
            car_controls.steering = 0
            car_controls.throttle = 1
        else:
            car_controls.steering = 0
            car_controls.throttle = 0
        # print('car speed: {0}'.format(car_state.speed))
        client.setCarControls(car_controls)
    elif (str(key) == 'Key.right'):
        car_state = client.getCarState()
        if (car_state.speed < max_speed):
            car_controls.throttle = 1
            car_controls.steering = steer_right
        else:
            car_controls.throttle = 0
            car_controls.steering = steer_right
        client.setCarControls(car_controls)
    elif (str(key) == 'Key.left'):
        car_state = client.getCarState()
        if (car_state.speed < max_speed):
            car_controls.throttle = 1
            car_controls.steering = steer_left
        else:
            car_controls.throttle = 0
            car_controls.steering = steer_left
        # print('car speed: {0}'.format(car_state.speed))
        client.setCarControls(car_controls)
    else:
        car_state = client.getCarState()
        if (car_state.speed < max_speed):
            car_controls.throttle = 1
            car_controls.steering = 0
        else:
            car_controls.throttle = 0
            car_controls.steering = 0
        # print('car speed: {0}'.format(car_state.speed))
        client.setCarControls(car_controls)


    #car_controls.steering = 1


# Collect events until released
with keyboard.Listener(
        on_press=on_press) as listener:
    listener.join()

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(on_press=on_press)
listener.start()
