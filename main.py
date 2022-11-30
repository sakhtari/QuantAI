import threading
import numpy as np
import cv2
import torch
import subprocess
import threading
import csv 
import pandas as pd
import matplotlib.colors


color_df = pd.read_csv('color_classes.csv')

# command and params for ffmpeg
rtmp_url = "rtmp://localhost/cv/stream"
cap = cv2.VideoCapture("rtmp://localhost/live/stream") # check this

# gather video info to ffmpeg
fps = int(cap.get(cv2.CAP_PROP_FPS))
#fps = 15
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 10000)

command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(width, height),
           '-r', str(fps),
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           rtmp_url]


p = subprocess.Popen(command, stdin=subprocess.PIPE)


def send_img(frame):
    p.stdin.write(frame.tobytes())


def draw_rectangles(frame, df):
    if df.empty:
        return frame
    
    thickness = 2
    for idx, row in df.iterrows():
        top_left = (int(row['xmin']), int(row['ymin']))
        bottom_right = (int(row['xmax']), int(row['ymax']))
        color_hex = (color_df.loc[row["class"]]["hex"]).lstrip('#')
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        frame = cv2.rectangle(frame, top_left, bottom_right, color_rgb, thickness)
        try:
            desc = "{} {}%".format(row['name'], int(row['confidence']*100))
            cv2.putText(frame, desc, (int(row['xmin']), int(row['ymin'])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, thickness)
        except:
            pass
            
    return frame


#model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # local model

while(True):
    # Capture frame-by-frame
    try:
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        #cv2.imshow('frame',gray)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        filename = 'test.jpg'
        #cv2.imwrite(filename, frame)

        imgs = [filename]  
        # Inference
        results = model(frame)
        # Results
        #results.save()
        #results.show()  # or .show()
        
        #results.xyxy[0]  # img1 predictions (tensor)
        #bboxes = results.pandas().xyxy[0]
        df = results.pandas().xyxy[0]
        frame = draw_rectangles(frame, df)

        #try:  
        x = threading.Thread(target=send_img, args=(frame,)) 
        x.start()

    except:
        print('error....continuing..')

    #except:
    #    pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

