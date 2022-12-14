# Reference: https://github.com/hukenovs/hagrid
import numpy as np
import cv2
import av
import skimage
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import distance
from scipy import ndimage
import time
import math
import colorsys 
import torchvision
from classifier.models.model import TorchVisionModel
from classifier.models.ssd_mobilenetv3 import SSDMobilenet
from PIL import Image, ImageOps
from torchvision.transforms import functional as f
import torch



targets = [
    "call",
    "dislike",
    "fist",
    "four",
    "like",
    "mute",
    "ok",
    "one",
    "palm",
    "peace",
    "rock",
    "stop",
    "stop inverted",
    "three",
    "two up",
    "two up inverted",
    "three2",
    "peace inverted",
    "no gesture"
]

def segment(frame):
    image = np.float32(frame.reshape((-1,3)))
    max_iter = 5
    epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    K = 5
    attempts = 10
    ret, label, center = cv2.kmeans(image, K, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))
    return res2

def calc_depth_sift(img1, img2):
    img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # compute keypoints and descriptors
    sift = cv2.SIFT_create(contrastThreshold=0.06, edgeThreshold=5) #0.06, 8
    kp_1, des_1 = sift.detectAndCompute(img1_g, None)
    kp_2, des_2 = sift.detectAndCompute(img2_g, None)

    # compute putative matches
    match_thres = 20000
    dist = scipy.spatial.distance.cdist(des_1, des_2,'sqeuclidean') 
    print(min(dist[0]))
    matches = []
    depth = []
    center_1 = np.array([img1.shape[0]/2, img1.shape[1]/2])
    center_2 = np.array([img2.shape[0]/2, img2.shape[1]/2])
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if dist[i][j] < match_thres:
                point_1 = kp_1[i].pt
                point_2 = kp_2[j].pt
                matches.append([point_1[0], point_1[1], point_2[0], point_2[1]])
                depth.append(scipy.spatial.distance.cdist([np.array(point_1) - center_1], [(np.array(point_2) - center_2)//2],'sqeuclidean')[0][0] )
                
    
    print("matches: ", len(matches))
    if len(depth) > 0:
        depth = depth/max(depth)
    #print(depth)
    return (matches, depth)

# calculate sum of squared differences
def ssd(arr1, arr2):
    norm_1 = (arr1 - np.mean(arr1))/np.std(arr1)
    norm_2 = (arr2 - np.mean(arr2))/np.std(arr2)
    return np.sum((norm_1-norm_2)**2)

# calculate sum of absolute differences
def sad(arr1, arr2):
    return np.sum(np.abs(arr1-arr2))

# calculate normalized correlation
def norm_corr(arr1, arr2):
    norm_1 = (arr1 - np.mean(arr1))/np.std(arr1)
    norm_2 = (arr2 - np.mean(arr2))/np.std(arr2)
    return np.sum(norm_1*norm_2)

# calculates depth using horizontal scanlines
def calc_depth(img1, img2):
    # Initialize values
    width1 = img1.shape[1] # num cols
    height1 = img1.shape[0] # num rows
    width2 = img2.shape[1] # num cols
    height2 = img2.shape[0] # num rows
    print("w: ", width1, "h: ", height1)
    window_size = (25,50) # (num horizontal pixels on either side of center, num vertical pixels on either side of center)
    scan_range = (50, height2//50)
    disparity = np.zeros((img1.shape[0], img1.shape[1]))

    # Iterate through image
    for i in range(0, height1, window_size[0]):
        for j in range(0, width1, window_size[1]):
            best_score = -1
            second_best = -1
            for m in range(i, i+1, 1):
                for k in range(max(0, j-scan_range[0]), min(j+scan_range[0], width2),1):
                    
                    # reference image window bounds
                    left_r = max(0, j - window_size[0])
                    right_r = min(width1-1, j+window_size[0])
                    up_r = max(0, i - window_size[1])
                    down_r = min(height1-1, i+window_size[1])

                    # size of window
                    w_l = j - left_r
                    w_r = right_r - j
                    h_u = i - up_r
                    h_d = down_r - i

                    # other image window bounds
                    left_o = max(0, k - w_l)
                    right_o = min(width2-1, k+w_r)
                    up_o = max(0, m - h_u)
                    down_o = min(height2-1, m+h_d)

                    # readjust ref window if needed
                    if left_o == 0:
                        w_l = k
                        left_r = j - w_l
                    if right_o == width2-1:
                        w_r = right_o - k
                        right_r = j + w_r
                    if up_o == 0:
                        h_u = m
                        up_r = i - h_u
                    if down_o == height2-1:
                        h_d = down_o - m
                        down_r = i + h_d

                
                    score = ssd(img1[up_r:down_r+1, left_r:right_r+1], img2[up_o:down_o+1,left_o:right_o+1])
                    if best_score == -1 or score < best_score:
                        second_best = best_score
                        best_score = score
                        disparity[up_r:down_r+1, left_r:right_r+1] = np.full((down_r+1 - up_r, right_r+1 - left_r), j - k)
                
                if np.abs(second_best- best_score) < 0.1 and second_best != -1: 
                        disparity[up_r:down_r+1, left_r:right_r+1] = np.zeros((down_r+1 - up_r, right_r+1 - left_r))
    disparity = disparity/np.amax(disparity)
    #disparity = scipy.ndimage.median_filter(disparity, (50, 50))
    #disparity = scipy.ndimage.gaussian_filter(disparity, 10) 
    print(disparity)
    return disparity

# implements a cartoon filter on the given frame
def cartoon_filter(frame):
    edges = cv2.adaptiveThreshold(scipy.ndimage.gaussian_filter(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)/255
    # add black lines to the frame
    for i in range(3):
        frame[:, :, i] = scipy.ndimage.gaussian_filter(frame[:, :, i], 5)
        frame[:, :, i] = frame[:, :, i]*edges
    # give appearance of cartoon color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = (hsv[:, :, 2]//18) * 18
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame

def web_cam_cal():
    X_3d = np.array([[-52.75, -13.5, 139, 1],
                    [2, 6, 57, 1],
                    [1.75, -7, 72, 1],
                    [-12.5, -11.5, 35, 1],
                    [-7.75, -18, 53.25, 1],
                    [1.75, -14, 44, 1]])
    x_2d = np.array([[500.0, 200.0], 
                        [300.0, 600.0],
                        [500.0, 600.0],
                        [700.0, 200.0],
                        [700.0, 400.0],
                        [700.0, 600.0]])
    A = []
    for i in range(6):
        pt_2d = x_2d[i]
        pt_3d = X_3d[i]
        row1 = np.append(np.append(np.array([0, 0, 0, 0]), -pt_3d), pt_2d[1]*pt_3d)
        row2 = np.append(np.append(pt_3d, np.array([0,0,0,0])), -pt_2d[0]*pt_3d)
        A.append(row1)
        A.append(row2)

    u, s, V = np.linalg.svd(A)
    P = V[len(V) - 1].reshape((3,4))
    
    P_inv = np.linalg.pinv(P)
    return P

def testing():
    video = cv2.VideoCapture(0)
    
    
    while True:
        #frame is in BGR, shape: (1080, 1920, 3)
        ret, frame = video.read()

        if not ret:
            print("Frame not read correctly")
            break
        cartoon = cartoon_filter(np.copy(frame))
        frame = cv2.flip(frame, 1)
        
        # cv2.circle(frame, (300, 200), 5, (255, 0, 0), 3)
        # cv2.circle(frame, (500, 200), 5, (255, 0, 0), 3)
        # cv2.circle(frame, (700, 200), 5, (255, 0, 0), 3)

        # cv2.circle(frame, (300, 400), 5, (255, 0, 0), 3)
        # cv2.circle(frame, (500, 400), 5, (255, 0, 0), 3)
        # cv2.circle(frame, (700, 400), 5, (255, 0, 0), 3)

        # cv2.circle(frame, (300, 600), 5, (255, 0, 0), 3)
        # cv2.circle(frame, (500, 600), 5, (255, 0, 0), 3)
        # cv2.circle(frame, (700, 600), 5, (255, 0, 0), 3)

        # display frame
        cv2.imshow("computer camera", frame)
        cv2.imshow("edges", cartoon)
        if cv2.waitKey(1) == ord('q'):
            break
    

def load_model():
    torch.device('cpu')
    classifier = torchvision.models.resnet18(num_classes = 19)
    classifier.load_state_dict(torch.load("./resnet18_classifier.pth", map_location=torch.device('cpu')))
    classifier.eval()
    classifier.to("cpu")

    detector = SSDMobilenet(num_classes=20)
    detector.load_state_dict("./SSDLite.pth", map_location='cpu')
    detector.eval()

    return detector, classifier

def preprocess(img: np.ndarray):
    # Reference: https://github.com/hukenovs/hagrid
    """
    Preproc image for model input
    Parameters
    ----------
    img: np.ndarray
        input image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)
    width, height = image.size

    image = ImageOps.pad(image, (max(width, height), max(width, height)))
    padded_width, padded_height = image.size
    image = image.resize((320, 320))

    img_tensor = f.pil_to_tensor(image)
    img_tensor = f.convert_image_dtype(img_tensor)
    img_tensor = img_tensor[None, :, :, :]
    return img_tensor, (width, height), (padded_width, padded_height)

def preprocess_classifier(x1,y1,x2,y2, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    w = x2-x1
    h = y2-y1
    size = max(h,w)
    box_hand = np.empty((size, size))
    try:
        box_hand = frame[y1:y1+size, x1:x1+size, :]
    except:
        box_hand = frame[y1:, x1:, :]

    image = Image.fromarray(box_hand)
    image = image.resize((320, 320))

    img_tensor = f.pil_to_tensor(image)
    img_tensor = f.convert_image_dtype(img_tensor)
    img_tensor = img_tensor[None, :, :, :]

    return img_tensor


# gesture detection using convex hull
def gesture(detector, classifier, frame):
    COLOR = (0, 255, 0) #green
    threshold = 0.5
    num_hands = 1
    processed_frame, size, padded_size = preprocess(frame)
    with torch.no_grad():
        output = detector(processed_frame)[0]
    boxes = output["boxes"][:num_hands]
    scores = output["scores"][:num_hands]
    labels = output["labels"][:num_hands]
    x1, x2, y1,y2 = 0,0,0,0
    for i in range(min(num_hands, len(boxes))):
        if scores[i] > threshold:
            width, height = size
            padded_width, padded_height = padded_size
            scale = max(width, height) / 320

            padding_w = abs(padded_width - width) // (2 * scale)
            padding_h = abs(padded_height - height) // (2 * scale)

            x1 = int((boxes[i][0] - padding_w) * scale)
            y1 = int((boxes[i][1] - padding_h) * scale)
            x2 = int((boxes[i][2] - padding_w) * scale)
            y2 = int((boxes[i][3] - padding_h) * scale)

            # boxed_hand_img = preprocess_classifier(x1,y1,x2,y2, frame)

            # labels = classifier(boxed_hand_img).squeeze(0)[:-1]
            # label = torch.argmax(labels).item()

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, thickness=3)
            cv2.putText(frame, targets[int(labels[i])-1], (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)

    center = ((x1+x2)//2, (y1+y2)//2)
    gesture = targets[int(labels[i])-1]

    if gesture in ["call", "like", "one", "mute"]: num_fingers = 1
    elif gesture in ["peace", "peace inverted", "rock", "two up","two up inverted"]: num_fingers = 2
    elif gesture in ["ok", "three", "three2"]: num_fingers = 3
    elif gesture in ["four", "stop", "stop inverted"]: num_fingers = 4
    else: num_fingers = 5

    finger_point = np.array([(x1+x2)//2, y1])

    return frame, center, num_fingers, finger_point


# determine local motion
def local_flow(flow):
    # left, right, up, down
    directions = np.array([0, 0, 0, 0])
    magn_thresh = 20
    diff_thresh = 200
    # separate (u,v) into vertical and horizontal components
    for f in flow:
        x_comp = f[2][0]
        y_comp = f[2][1]
        if x_comp < 0:
            directions[0] -= x_comp
        else:
            directions[1] += x_comp
        if y_comp < 0:
            directions[2] -= y_comp
        else:
            directions[3] += y_comp

    
    # if np.count_nonzero(directions > magn_thresh)>=4 and np.abs(max(directions) - min(directions)) < diff_thresh:
    #     print("zoom")
    # else:
    #     print("up: ", directions[2], "down: ", directions[3], "left: ", directions[0], "right: ", directions[1])
    key = ["L", "R", "U", "D"]
    if max(directions) > magn_thresh:
        #print(key[np.argmax(directions)])
        return key[np.argmax(directions)]
    return "N"

# calculate optical flow of a video stream and local flow around center point given
# note: center = (x,y)
def optical_flow(prev_frame, frame, center, display):
    window_size = (25, 25) #height x width

    # convert to grayscale
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)/255
    frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255

    # smooth image
    frame_g = scipy.ndimage.gaussian_filter(frame_g, 3)
    prev_frame = scipy.ndimage.gaussian_filter(prev_frame, 3)

    Ix = scipy.ndimage.convolve(frame_g, [[2,0,-2]])
    Iy = scipy.ndimage.convolve(frame_g, [[2],[0], [-2]])
    It = frame_g - prev_frame
    
    # for each window calculate (u,v)
    height = frame_g.shape[0]
    width = frame_g.shape[1]
    flow = []
    # iterate through each window
    for i in range(0, height - window_size[0], window_size[0]):
        for j in range(0, width - window_size[1], window_size[1]):
            A = np.append(Ix[i: i + window_size[0], j: j + window_size[1]].flatten(), Iy[i: i + window_size[0], j: j + window_size[1]].flatten())
            A = A.reshape((window_size[0] * window_size[1],2), order = 'F')
            b = -(It[i: i + window_size[0], j: j + window_size[1]].flatten())
            #solution, residuals, rank, s = np.linalg.lstsq(A, b)
            #print(solution)
            At_A = np.matmul(np.transpose(A), A)
            if np.linalg.det(At_A) != 0:
                solution = np.matmul(np.matmul(np.linalg.inv(At_A), np.transpose(A)), b)
                #print(solution)
                flow.append((i + window_size[0]//2, j + window_size[1]//2, solution))
    
    # local flow
    local_subset = []
    upper_left = (max(0, center[0] - 250), max(0, center[1] - 50))
    bottom_right = (min(width-1, center[0] + 250), min(height-1, center[1] + 300))
    #cv2.rectangle(display, upper_left, bottom_right, (0, 255, 0), 3)

    # draw arrows on image
    for f in flow:
        start_point = (int(f[1]), int(f[0])) #x, y
        end_point = (int(f[1] + f[2][0]), int(f[0] + f[2][1]))
        #print(end_point)
        if start_point[0] > upper_left[0] and start_point[0] < bottom_right[0] and start_point[1] > upper_left[1] and start_point[1] < bottom_right[1]:
            local_subset.append(f)
            display = cv2.arrowedLine(display, start_point, end_point, (255, 0, 0), 2, tipLength=0.25)
    
    direction = local_flow(local_subset)
    return direction

    

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

# make the vfx flame tilt to the side
def effect_tilt(effect, tilt):
    #if no tilt frame does not change
    if tilt == 0:
        return effect
    w = effect.shape[1]
    h = effect.shape[0]
    channels = effect.shape[2]
    tilt_abs = np.abs(tilt)
    max_offset = int(((h-1)*tilt_abs)**0.75)
    output = np.zeros((h,w + 2*max_offset,channels))
    
    offset = 0
    for i in range(h):
        #offset += (0.75)**(i*tilt_abs)
        offset = ((h-i-1)*tilt_abs)**0.75
        # tilts to the right
        if tilt < 0:
            output[i, max_offset + int(offset) : max_offset + int(offset) + w, :] = effect[i, :, :]
        # tilts to the left
        else:
            output[i, output.shape[1]-int(offset)-w-max_offset : output.shape[1]-int(offset)-max_offset, :] = effect[i, :, :]
    return output

# overlay effect on frame at location center
def overlay_effect(frame, effect, center):
    vfx_shape = effect.shape
    # calculate bounds of effect
    right_b = min(vfx_shape[1]//2, frame.shape[1]-center[0])
    left_b = min(vfx_shape[1] - vfx_shape[1]//2, center[0])
    top_b = min(vfx_shape[0], center[1])
    # add in vfx using alpha channel
    for i in range(3):
        frame[center[1] - top_b: center[1], center[0]-left_b:center[0]+right_b, i] = frame[center[1] - top_b: center[1], center[0]-left_b:center[0]+right_b, i]* (1 - effect[vfx_shape[0] - top_b:, vfx_shape[1] - vfx_shape[1]//2 - left_b:vfx_shape[1] - vfx_shape[1]//2 + right_b, 3])
        
        frame[center[1] - top_b: center[1], center[0]-left_b:center[0]+right_b, i] = frame[center[1] - top_b: center[1], center[0]-left_b:center[0]+right_b, i]+ effect[vfx_shape[0] - top_b:, vfx_shape[1] - vfx_shape[1]//2 - left_b:vfx_shape[1] - vfx_shape[1]//2 + right_b, i]*255*effect[vfx_shape[0] - top_b:, vfx_shape[1] - vfx_shape[1]//2 - left_b:vfx_shape[1] - vfx_shape[1]//2 + right_b, 3]

# track a green shape and add fireball effect
def track(effect):
    video = cv2.VideoCapture(0)
    # give camera time to connect
    time.sleep(1)

    #video_usb = cv2.VideoCapture(1)
    vfx_num = 0
    vfx_length = effect.shape[0]
    
    #x, y (col, row)
    center = [0,0] 
    prev_center = [0,0]
    # if the center hasn't been detected for a certain amount of time stop displaying flame
    no_center_count = 0
    no_center_limit = 5
    
    # need previous frame for optical flow
    ret, prev_frame = video.read()
    prev_frame = cv2.flip(prev_frame, 1)
    # history of flow directions
    direction = np.full(6, "N")
    # how much the effect should be tilted
    tilt_num = 0
    # coloring effect (hsv)
    color_effect = 0
    while True:
        #frame is in BGR, shape: (720, 1280, 3)
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)
        # keep a copy of the raw unaltered frame
        raw_frame = frame.copy()
        #ret_usb, frame_usb = video_usb.read()
        #print(frame.shape)
        if not ret:
            print("Frame not read correctly")
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Threshold for pink
        lower_bound = np.array([160, 75, 120])
        upper_bound = np.array([175, 255, 200])
        # Threshold for green
        #lower_bound = np.array([30, 50, 120])
        #upper_bound = np.array([40, 200, 255])
        threshold_frame = cv2.inRange(hsv, lower_bound, upper_bound)

        # detect shape
        contours, hiearchy  = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        prev_center = center.copy()
        for c in range(len(contours)):
            if (cv2.contourArea(contours[c]) > 25):
                cv2.drawContours(frame, [contours[c]], 0, (0, 0, 255), 5)
                # find the center of the shape
                M = cv2.moments(contours[c])
                center[0] = int(M['m10']/M['m00'])
                center[1] = int(M['m01']/M['m00'])
                no_center_count = 0
        cv2.circle(frame, (center[0],center[1]), 5, (255, 0, 0), 3)

        if center[0] > 0 and no_center_count < no_center_limit:
            # calculate optical flow around center point
            direction = np.insert(direction, 0, optical_flow(prev_frame, raw_frame, center, frame))
            direction = direction[:6]
            # add effect to video at center point
            flame = effect[vfx_num].copy()
            flame[:,:,0] =  flame[:,:,0] + color_effect/255
            flame[:,:,2] =  flame[:,:,2] - color_effect/255
            print(color_effect)
            overlay_effect(frame, effect_tilt(flame, tilt_num), center)
   
            #print(math.dist(prev_center, center))
            


        # increment to next frame in vfx
        vfx_num = (vfx_num+1)%vfx_length
        # update previous frame for optical flow
        prev_frame = raw_frame
        # no center count continuously increments and is only reset if a center is found
        no_center_count +=1

        # determine if the effect should tilt or resize depending on optical flow and movement of the center
        center_thres = 7.5
        if math.dist(prev_center, center) > center_thres:
            # if center is moving and optical flow is left/right then the effect tilts
            if np.count_nonzero(direction[:5] == "L") >=3:
                tilt_num = max(tilt_num - 0.25, -3.5) 
            elif np.count_nonzero(direction[:5] == "R")>=3:
                tilt_num = min(tilt_num + 0.25, 3.5) 
            else:
                # stopped going left but still moving
                if tilt_num < 0:
                    tilt_num = min(tilt_num + 0.75, 0) 
                # stopped going right but still moving
                elif tilt_num > 0:
                    tilt_num = max(tilt_num - 0.75, 0) 
        else:
            # if center is not moving and optical flow is up/down then the effect should resize
            #  stopped moving so effect should stop tilting
            if tilt_num < 0:
                tilt_num = min(tilt_num + 0.75, 0) 
            elif tilt_num > 0:
                tilt_num = max(tilt_num - 0.75, 0)  

            # optical flow is up or down
            if np.count_nonzero(direction[:5] == "U") >=3:
                color_effect = min(color_effect+15, 225)
            elif np.count_nonzero(direction[:5] == "D")>=3:
                color_effect = max(color_effect-15, 0)


        # display frame
        cv2.imshow("computer camera", frame)
        #cv2.imshow("color", threshold_frame)
        #cv2.imshow("usb camera", frame_usb)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__": 

    '''
    # process vfx video and save np array to txt file
    input_file = "fire_effect.mov"

    effect_arr = get_effect_video(input_file)

    save_arr = effect_arr.reshape(effect_arr.size)
    save_arr = np.insert(save_arr, 0, effect_arr.shape)
    np.savetxt("fire_effect.txt", save_arr)
    '''
    
    # load fire effect
    effect = np.loadtxt("fire_effect.txt")
    effect = effect[4:].reshape((int(effect[0]), int(effect[1]), int(effect[2]), int(effect[3])))
    # print(effect_arr.shape)

    # # for t in range(10):
    # #     for f in range(effect_arr.shape[0]):
    # #         output = effect_tilt(effect_arr[f], t)
    # #         cv2.imshow("tilted", output)
    # #         cv2.waitKey(50)
    # track(effect_arr)

    # history of flow directions
    direction = np.full(6, "N")
    # how much the effect should be tilted
    tilt_num = 0
    # coloring effect (hsv)
    color_effect = 0
    # history of gestures
    gesture_list = np.full(6, 10)

    # if the center hasn't been detected for a certain amount of time stop displaying flame
    no_center_count = 0
    no_center_limit = 5
    prev_center = [0,0]

    vfx_num = 0
    vfx_length = effect.shape[0]
    
    # flag for cartoon filter
    cartoon = False

    video = cv2.VideoCapture(0)
    # need previous frame for optical flow
    ret, prev_frame = video.read()
    prev_frame = cv2.flip(prev_frame, 1)

    detector, classifier = load_model()
    while True:
        ret, frame = video.read()
        if not ret:
            print("Frame not read correctly")
            break
        
        
        # get gesture
        frame = cv2.flip(frame, 1)
        raw_frame = frame.copy()
        frame_gesture, center_gesture, num_fingers, finger_point = gesture(detector, classifier, frame.copy())
        gesture_list = np.insert(gesture_list, 0, num_fingers)
        gesture_list = gesture_list[:6]

        frame_flow = frame.copy()
        # 1 finger -> allow tilt
        if num_fingers == 1:
            # calculate optical flow around center point
            direction = np.insert(direction, 0, optical_flow(prev_frame, raw_frame, finger_point, frame_flow))
            direction = direction[:6]
            #print(direction)
            # add effect to video at center point
            center = finger_point
            flame = effect[vfx_num].copy()
            flame[:,:,0] =  flame[:,:,0] + color_effect/255
            flame[:,:,2] =  flame[:,:,2] - color_effect/255
            #print(color_effect)
            if cartoon:
                frame = cartoon_filter(frame)
            overlay_effect(frame, effect_tilt(flame, tilt_num), center)

            # calculate tilt
            center_thres = 7.5
            if math.dist(prev_center, center) > center_thres:
                # if center is moving and optical flow is left/right then the effect tilts
                if np.count_nonzero(direction[:5] == "L") >=3:
                    tilt_num = max(tilt_num - 0.25, -3.5) 
                elif np.count_nonzero(direction[:5] == "R")>=3:
                    tilt_num = min(tilt_num + 0.25, 3.5) 
                else:
                    # stopped going left but still moving
                    if tilt_num < 0:
                        tilt_num = min(tilt_num + 0.75, 0) 
                    # stopped going right but still moving
                    elif tilt_num > 0:
                        tilt_num = max(tilt_num - 0.75, 0) 
            else:
                # if center is not moving and optical flow is up/down then the effect should resize
                #  stopped moving so effect should stop tilting
                if tilt_num < 0:
                    tilt_num = min(tilt_num + 0.75, 0) 
                elif tilt_num > 0:
                    tilt_num = max(tilt_num - 0.75, 0) 
            
            # update previous center, used to check for movement (filter out noise)
            prev_center =  center

        # 2 finger -> change color
        elif num_fingers == 2:
            # center in between both fingers
            center = finger_point
            # calculate optical flow around center point
            direction = np.insert(direction, 0, optical_flow(prev_frame, raw_frame, center, frame_flow))
            direction = direction[:6]
            #print(direction)
            flame = effect[vfx_num].copy()
            flame[:,:,0] =  flame[:,:,0] + color_effect/255
            flame[:,:,2] =  flame[:,:,2] - color_effect/255
            #print(color_effect)
            if cartoon:
                frame = cartoon_filter(frame)
            overlay_effect(frame, flame, center)

            # optical flow is up or down
            if np.count_nonzero(direction[:5] == "U") >=3:
                color_effect = min(color_effect+15, 225)
            elif np.count_nonzero(direction[:5] == "D")>=3:
                color_effect = max(color_effect-15, 0)

        # >=3 finger -> toggle cartoon filter
        elif num_fingers >= 3:
            # was not >=3 and then turned to >=3 gesture
            if np.count_nonzero(gesture_list[1:]<3) >=3 and np.count_nonzero(gesture_list[:5] >=3) >=3:
                cartoon = not cartoon
            if cartoon:
                frame = cartoon_filter(frame)

        # increment to next frame in vfx
        vfx_num = (vfx_num+1)%vfx_length
        # update previous frame for optical flow
        prev_frame = raw_frame
        # no center count continuously increments and is only reset if a center is found
        # no_center_count +=1
        
        # display
        cv2.imshow("gesture", frame_gesture)
        cv2.imshow("camera", frame)
        cv2.imshow("optical flow", frame_flow)
        if cv2.waitKey(1) == ord('q'):
            break