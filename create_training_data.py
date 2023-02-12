import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array

    [W,A,S,D] boolean values.
    '''
    output = [0,0,0,0]
    
    if 'W' in keys:
        output[0] = 1
    if 'A' in keys:
        output[1] = 1
    if 'S' in keys:
        output[2] = 1
    if 'D' in keys:
        output[3] = 1
    
    return output


file_name = 'training_data.npz'
imgData,keyData = np.ndarray(shape=(1,480,270)),np.ndarray(shape=(1,4))
if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    npzFile = np.load(file_name,allow_pickle=True)
    print(npzFile)
    imgData = npzFile["arr_0"]
    keyData = npzFile["arr_1"]
else:
    print('File does not exist, starting fresh!')



def main():

    for i in list(range(4))[::-1]:
        print("Begin in",i+1)
        time.sleep(1)

    paused = False
    while(True):

        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0,40,800,640))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            screen = cv2.resize(screen, (480,270))
            print(screen)
            
            # resize to something a bit more acceptable for a CNN
            keys = key_check()
            output = keys_to_output(keys)
            print(output)

            np.append(imgData,screen)
            np.append(keyData,output)
            print(imgData.size)
            if imgData.size % 1000 == 0:
                print(len(imgData))
                print(imgData.size)
                np.savez(file_name,imgData,keyData)

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main()