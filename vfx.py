import keyboard
import numpy as np
import cv2
import av

# track a green shape
def track():
    video = cv2.VideoCapture(0)

    while True:
        #frame is in BGR, shape: (720, 1280, 3)
        ret, frame = video.read()
        print(frame.shape)
        if not ret:
            print("Frame not read correctly")
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Threshold for green
        lower_bound = np.array([25, 35, 140])
        upper_bound = np.array([50, 255, 255])
        threshold_frame = cv2.inRange(hsv, lower_bound, upper_bound)

        # detect shape
        contours, hiearchy  = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in range(len(contours)):
            if c == 0:
                continue
            if (cv2.contourArea(contours[c]) > 100):
                cv2.drawContours(frame, [contours[c]], 0, (0, 0, 255), 5)
                # find the center of the shape
                M = cv2.moments(contours[c])
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.circle(frame, (cx,cy), 5, (255, 0, 0), 3)

        # display frame
        cv2.imshow("computer camera", frame)
        if cv2.waitKey(1) == ord('q'):
            break
'''
currX = 0
currY = 0
def keyboardPress(keyboardEvent):
    
    global currX
    global currY
    #begin keyboard loop
    while 1:
        #currKey = keyboardEvent.name
        currKey = ""
        if currKey == "left":
            if currX > 0:
                currX -= 1
        elif currKey == "right":
            if currX < 400:
                currX += 1
        elif currKey == "up":
            if currY > 0:
                currY -= 1
        elif currKey == "down":
            if currY < 400:
                currY += 1
        print(currX,currY,currKey)
'''

def get_effect_video(path):
    container = av.open(path)
    skip_count = 0
    for frame in container.decode(video=0):
        if skip_count == 2:
            skip_count = 0
            continue
        elif skip_count == 0:
            array = frame.to_ndarray(format='rgba')
            print(array.shape)
            cv2.imshow("fire", array)
            if cv2.waitKey(1) == ord('q'):
                break
        skip_count +=1
    return array

if __name__ == "__main__": 
    #keyboard.hook(keyboardPress)
    input_file = "fire_effect.mov"
    effect_arr = get_effect_video(input_file)
    #track()