import numpy as np
import cv2

cap = cv2.VideoCapture('lanes.mp4')

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), (0,255,0), 3)
    except:
        pass

def roi(img, vetrs):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [verts], 255)
    roiImg = cv2.bitwise_and(img, mask)
    return roiImg

def find_largest(img):
    c = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    height = np.size(img, 0)
    width = np.size(img, 1)

    defcenter = (int(width/2),int(height/2),int(2),0)

    if len(c) > 0:
        mc = max(c, key=cv2.contourArea)
        (x,y),radius = cv2.minEnclosingCircle(mc)
        if radius < 5:
            return defcenter
        center = (int(x),int(y),int(radius),1)
        return center
    return defcenter


def lane_detection(img, out, ystart, xstart, width, kheight, kwidth, length):
    # left_crop = img[ystart-kheight:ystart, xstart:xstart+kwidth]
    # right_crop = img[ystart-kheight:ystart, xstart+width-kwidth:xstart+width]

    new_img = img.copy()

    left_line = np.zeros((length, 5))
    right_line = np.zeros((length, 5))

    L_bottom_crop = new_img[ystart-kheight:ystart, xstart:xstart+width/2]
    R_bottom_crop = new_img[ystart-kheight:ystart, xstart+width/2:xstart+width]
    L_bottom_center = find_largest(L_bottom_crop)
    R_bottom_center = find_largest(R_bottom_crop)

    # cv2.line(img, (0,np.size(img, 1)),(np.size(img, 0),np.size(img, 1)), (255,0,0), 2)

    L_prevX = xstart
    if L_bottom_center[3] == 1:
        L_prevX = xstart+L_bottom_center[0]-kwidth/3
    
    R_prevX = xstart + width - kwidth
    if R_bottom_center[3] == 1:
        R_prevX = xstart+width/2+R_bottom_center[0]-kwidth/3

    for i in range(0,length):
        yMin = ystart-kheight*(i+1)
        yMax = ystart-kheight*(i)

        L_crop = new_img[yMin:yMax, L_prevX:L_prevX+kwidth]
        L_local_center = find_largest(L_crop)
        L_X = L_prevX + kwidth/6
        if L_local_center[3] == 1:
            L_X = L_prevX - (L_local_center[0]/2 - L_local_center[0])

        R_crop = new_img[yMin:yMax, R_prevX:R_prevX+kwidth]
        R_local_center = find_largest(R_crop)
        R_X = R_prevX - kwidth/6
        if R_local_center[3] == 1:
            R_X = R_prevX - (R_local_center[0]/2 - R_local_center[0]) - kwidth / 3

        # cv2.rectangle(out,(L_X,yMin),(L_X+kwidth,yMax),(255,0,0),3)
        # cv2.rectangle(out,(R_X,yMin),(R_X+kwidth,yMax),(255,0,0),3)
        left_line[i] = [L_prevX+kwidth/2,L_X+kwidth/2,yMin,yMax,L_X]
        right_line[i] = [R_prevX+kwidth/2,R_X+kwidth/2,yMin,yMax,R_X]

        L_prevX = L_X
        R_prevX = R_X
    return (left_line, right_line)

while(cap.isOpened()):
    ret, frame = cap.read()

    verts = np.array([[360,570],[490,450],[790,450],[920,570]])

    contrast = 250
    brightness = 200

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    grayContrast = np.int16(gray.copy())
    grayContrast = grayContrast*(contrast/127+1) - contrast + brightness
    grayContrast = np.uint8(np.clip(grayContrast, 0, 255))

    ret,grayThresh = cv2.threshold(grayContrast,200,255,cv2.THRESH_BINARY)

    roiGray = roi(grayContrast, verts)
    roiThresh = roi(grayThresh, verts)

    kwidth = 75
    lines = lane_detection(roiThresh, roiGray, 570, 360, 560, 10, kwidth, 10)

    final = gray.copy()

    for line in lines[0]: # Draw left lane
        cv2.line(final,(int(line[1]),int(line[2])),(int(line[0]),int(line[3])),(255,0,0),3)
        cv2.rectangle(grayContrast, (int(line[4]-kwidth/4),int(line[2])),(int(line[4]+kwidth-kwidth/4),int(line[3])),(255,0,0),3)

    for line in lines[1]: # Draw right lane
        cv2.line(final,(int(line[1]),int(line[2])),(int(line[0]),int(line[3])),(255,0,0),3)
        cv2.rectangle(grayContrast, (int(line[4]),int(line[2])),(int(line[4]+kwidth),int(line[3])),(255,0,0),3)

    visT = np.concatenate((gray, grayThresh,grayContrast), axis=1)
    visB = np.concatenate((roiGray, roiThresh,final), axis=1)
    vis = np.concatenate((visT, visB), axis=0)
    small = cv2.resize(vis, (0,0), fx=0.4, fy=0.4)

    cv2.imshow('Lanes',small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()