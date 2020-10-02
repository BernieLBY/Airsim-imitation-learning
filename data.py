import os
import pandas as pd
import setup_path
import airsim
# DIR_NAME = 'D:\Windows\AirSimTutorial\imitation_learning\raw_data\2018-11-23-16-06-30_segmentation_data'
# current_df = pd.read_csv('./airsim_rec.txt', sep='\t')
#
# # print(current_df['ImageFile'])
# MAX_SPEED = 70.0
# for i in range(1, current_df.shape[0] - 1):
#     speed = current_df.iloc[i - 1][['Speed']] * 3.6 #/MAX_SPEED
#     print(speed)

client = airsim.CarClient()
client.confirmConnection()
# client.enableApiControl(False)
while True:
    car_state = client.getCarState()
    if car_state.speed > 15:
        print("=================================")
        print(car_state.speed)
        print(car_state.speed * 3.6)
        print("=================================")
    # speed = car_state['speed']
    # print(speed)