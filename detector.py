import cv2
import sys
import os

face_cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

scale_factor = 1.1
min_neighbors = 3
min_size = (30, 30)
flags = cv2.cv.CV_HAAR_SCALE_IMAGE

image_path = sys.argv[1]
image = cv2.imread(image_path)

faces = face_cascade.detectMultiScale(image, scaleFactor = scale_factor, minNeighbors = min_neighbors,
	minSize = min_size, flags = flags)

idx = 0
prefix = image_path.split(".")[0]
for( x, y, w, h ) in faces:
	cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
	#face = cv2.crop()
	facename = "%s_%s.jpg" % (prefix, idx)
	idx = idx + 1
	#outfname = "/tmp/%s.faces.jpg" % os.path.basename(infname)
outfname = "%s_faces.jpg" % prefix
cv2.imwrite(os.path.expanduser(outfname), image)