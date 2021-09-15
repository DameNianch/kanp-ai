import glob
import cv2
import os

video_dir = "../../../mldata/videos"
image_dir = "../../../mldata/train"

class_name = ["cola", "calpico"]

for i_class in class_name:
    count = 0
    for j_video in glob.glob(os.path.join(video_dir, i_class, "*.mp4")):
        cap = cv2.VideoCapture(j_video)
        ret = True
        while ret:
            ret, image = cap.read()
            if ret:
                image = cv2.resize(image, dsize=(240, 240))
                print(os.path.join(image_dir, i_class, j_video + str(count) + ".jpg"))
                cv2.imwrite(os.path.join(image_dir, i_class, str(count) + ".jpg"), image)
                count += 1
