import cv2 as cv
import pathlib
import os

print(os.getcwd())

BASE_DATA_DIR = "opencv/data"
CATS = "/cat images"
VIDEOS = "/videos"

video = "/alice.mp4"

video_path = BASE_DATA_DIR + VIDEOS + video

if pathlib.Path(video_path).exists():
	cap = cv.VideoCapture(video_path)#numbers are used for camera devices

	while True:
		isTrue, frame = cap.read()
		cv.imshow("Video", frame)

		if cv.waitKey(20) & 0xFF == ord("d"):
			break

	cap.release()
	cv.destroyAllWindows()


