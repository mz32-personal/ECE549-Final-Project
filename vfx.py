import keyboard
import numpy as np
import cv2
import av
import skimage

# track a green shape and add fireball effect
def track(effect):
    video = cv2.VideoCapture(0)
    vfx_num = 0
    vfx_length = effect.shape[0]
    vfx_shape = effect[0].shape
    #x, y (col, row)
    center = [0,0] 
    # if the center hasn't been detected for a certain amount of time stop displaying flame
    no_center_count = 0
    no_center_limit = 5
    while True:
        #frame is in BGR, shape: (720, 1280, 3)
        ret, frame = video.read()
        print(frame.shape)
        if not ret:
            print("Frame not read correctly")
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Threshold for green
        lower_bound = np.array([30, 50, 120])
        upper_bound = np.array([40, 200, 255])
        threshold_frame = cv2.inRange(hsv, lower_bound, upper_bound)

        # detect shape
        contours, hiearchy  = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in range(len(contours)):
            if c == 0:
                continue
            if (cv2.contourArea(contours[c]) > 50):
                #cv2.drawContours(frame, [contours[c]], 0, (0, 0, 255), 5)
                # find the center of the shape
                M = cv2.moments(contours[c])
                center[0] = int(M['m10']/M['m00'])
                center[1] = int(M['m01']/M['m00'])
                no_center_count = 0
        #cv2.circle(frame, (center[0],center[1]), 5, (255, 0, 0), 3)

        if center[0] > 0 and no_center_count < no_center_limit:
            # add in vfx using alpha channel
            right_b = min(vfx_shape[1]//2, frame.shape[1]-center[0])
            left_b = min(vfx_shape[1] - vfx_shape[1]//2, center[0])
            top_b = min(vfx_shape[0], center[1])
            bottom_b = 0
            #h = min(vfx_shape[0], frame.shape[0]-center[1])
            #w = min(vfx_shape[1], frame.shape[1]-center[0])
            for i in range(3):
                frame[center[1] - top_b: center[1], center[0]-left_b:center[0]+right_b, i] = frame[center[1] - top_b: center[1], center[0]-left_b:center[0]+right_b, i]* (1 - effect[vfx_num][vfx_shape[0] - top_b:, vfx_shape[1] - vfx_shape[1]//2 - left_b:vfx_shape[1] - vfx_shape[1]//2 + right_b, 3])
               
                frame[center[1] - top_b: center[1], center[0]-left_b:center[0]+right_b, i] = frame[center[1] - top_b: center[1], center[0]-left_b:center[0]+right_b, i]+ effect[vfx_num][vfx_shape[0] - top_b:, vfx_shape[1] - vfx_shape[1]//2 - left_b:vfx_shape[1] - vfx_shape[1]//2 + right_b, i]*255*effect[vfx_num][vfx_shape[0] - top_b:, vfx_shape[1] - vfx_shape[1]//2 - left_b:vfx_shape[1] - vfx_shape[1]//2 + right_b, 3]
                
                #frame[center[1]: center[1] + h, center[0]:center[0] + w, i] = frame[center[1]: center[1] + h, center[0]:center[0] + w, i]* (1 - effect[vfx_num][:h, :w, 3])
                #frame[center[1]: center[1] + h, center[0]:center[0] + w, i] = frame[center[1]: center[1] + h, center[0]:center[0] + w, i] + effect[vfx_num][:h, :w, i]*255*effect[vfx_num][:h, :w, 3]
            '''print(type(effect[0][0][0][0]))
            print(type(frame[0][0][0]))
            print(frame[center[1]: center[1] + h, center[0]:center[0] + w].shape)
            print(effect[:h, :w, :3].shape)'''
        
        
        # increment to next frame in vfx
        vfx_num = (vfx_num+1)%vfx_length
        # no center count continuously increments and is only reset if a center is found
        no_center_count +=1
        # display frame
        cv2.imshow("computer camera", frame)
        if cv2.waitKey(1) == ord('q'):
            break

# convert .mov to np array
def get_effect_video(path):
    container = av.open(path)
    skip_count = 0
    vfx = []
    for frame in container.decode(video=0):
        # skip every few frames
        if skip_count == 2:
            skip_count = 0
            continue
        elif skip_count == 0:
            array = frame.to_ndarray(format='bgra')
            h = array.shape[0]
            w = array.shape[1]
            # crop out some of the empty space
            cropped_array = array[h//4:9*h//10, w//3:2*w//3]
            # downsample to shrink the vfx
            downsampled = skimage.transform.resize(cropped_array, (cropped_array.shape[0]//4, cropped_array.shape[1]//4), anti_aliasing=True, preserve_range=False)
            vfx.append(downsampled)
            cv2.imshow("fire", downsampled)
            if cv2.waitKey(1) == ord('q'):
                break
        skip_count +=1
    return np.array(vfx)

if __name__ == "__main__": 
    input_file = "fire_effect.mov"
    effect_arr = get_effect_video(input_file)
    print(effect_arr.shape)
    track(effect_arr)