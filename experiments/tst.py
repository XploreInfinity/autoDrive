import cv2,time
from grabscreen import grab_screen
import numpy as np
time.sleep(2)
screen = grab_screen(region=(0,40,1920,1120))
screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
screen = cv2.resize(screen, (480,270))
cv2.imshow("window",screen)
cv2.waitKey(0)
lolc= [screen]
output= [[0,0,0,1],[0,1,0,1]]
#np.savez('lcat.npy',lolc,output)
lolc = np.load('lcat.npy.npz')
cv2.imshow("amy",lolc["arr_1"][0])
cv2.waitKey(0)
