#Importing libraries
import cv2
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np

cap=cv2.VideoCapture('solidWhiteRight.mp4')

def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        image = np.copy(image) # don't want to modify the original
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    image=frame
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.array([  0, 200,   0])
    upper = np.array([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    lower = np.array([ 10,   0, 100])
    upper = np.array([255, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    gray_image=cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)

    kernel_size=15
    gaussian_images = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)

    low_threshold=50
    high_threshold=150
    edge_images=cv2.Canny(gaussian_images, low_threshold, high_threshold)

    rows, cols = edge_images.shape[:2]
    bottom_left  = [cols*0.01, rows*0.76]
    top_left     = [cols*0.06, rows*0.63]
    bottom_right = [cols*0.7, rows*0.76]
    top_right    = [cols*0.65, rows*0.63] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    mask = np.zeros_like(edge_images)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension 
    
    filtered_img=cv2.bitwise_and(edge_images, mask)
    list_of_lines=list(cv2.HoughLinesP(filtered_img, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300))

    out_frame=draw_lines(image, list_of_lines)
    cv2.imshow('frame',out_frame)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()



    