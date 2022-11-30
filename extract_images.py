import cv2
vidcap = cv2.VideoCapture('/home/simon/Desktop/drone_footage/DJI_0798.mp4')
success,image = vidcap.read()
count = 0
while success:
    if count % 30 == 0:
        cv2.imwrite("/home/simon/Desktop/drone_footage/images/DJI_0798_frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

