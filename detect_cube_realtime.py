import cv2,sys
from detect_cube import CubeDetector
import numpy as np


if __name__ == '__main__':
    if len(sys.argv) == 2:
        cap = cv2.VideoCapture(sys.argv[1])
    else:
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('DISPLAY')
    cube_detector = CubeDetector()
    while cap.isOpened():
	ret, image = cap.read()
        rows,cols,channels = image.shape

        images = cube_detector.get_face(image)
	cv2.imshow('DISPLAY', cv2.resize(images[-1][-1], (cols,rows)))

	if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('test.png', image)
	    break

    cap.release()
    cv2.destroyAllWindows()
