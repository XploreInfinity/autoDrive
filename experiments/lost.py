import cv2,time
from grabscreen import grab_screen
while True:
        time.sleep(1) 
        screen = grab_screen(region=(0,40,1920,1120))
        screen = cv2.resize(screen, (480,270))
        # run a color convert:
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        cv2.imshow('window',cv2.resize(screen,(640,360)))