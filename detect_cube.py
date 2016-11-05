import cv2, sys, time, random
import numpy as np
import math

# http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged


def find_shapes(img):
    rows,cols,channels = img.shape
    img_small = cv2.resize(img, (cols/10, rows/10))
    edges = auto_canny(img_small)
    cv2.imwrite('auto_edges.png', edges)
    edges = cv2.dilate(edges, np.ones((3,3)), iterations=1)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2,2)), iterations=1)
    edges = 255-edges
    cv2.imwrite('auto_edges_dilated.png', edges)
    contour_img, contours, hierarchy = cv2.findContours(edges.copy(),\
                            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = img_small.copy()
    for contour in contours:
        color = (random.randint(0, 255), random.randint(0,255), random.randint(0,255))
        cv2.drawContours(img_copy, [contour], 0, color, 1)
    cv2.imwrite('contours.png', img_copy)

    img_copy = img_small.copy()
    new_img = np.zeros((rows/10, cols/10))
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        #for point in contour:
        #    if point[0][0] == 59 and point[0][1] == 42:
        #        print contour
        #        print approx
        #        color = (random.randint(0, 255), random.randint(0,255), random.randint(0,255))
        #        cv2.drawContours(img_copy, [approx], 0, color, -1)

        
        # 4 sides
        if len(approx) == 4 or len(approx) == 5 or len(approx) == 6:
            # only keep contours that are square/circular enough in nature
            M = cv2.moments(approx)
            if M["m00"] == 0:
                continue
	    cX = int(M["m10"] / M["m00"])
	    cY = int(M["m01"] / M["m00"])
            
            distances = []
            for p in approx:
                distances.append(math.sqrt((p[0][0] - cX)**2 + (p[0][1] - cY)**2))

            if np.std(distances) < 1:
                color = (random.randint(0, 255), random.randint(0,255), random.randint(0,255))
                cv2.drawContours(img_copy, [approx], 0, color, -1)
                cv2.rectangle(new_img, (cX-2, cY-2), (cX+2, cY+2), (255,0,0), -1)
            
            #(x, y, w, h) = cv2.boundingRect(approx)
	    #ar = w / float(h)
            #if ar >= 0.75 and ar <= 1.25:
            #    color = (random.randint(0, 255), random.randint(0,255), random.randint(0,255))
            #    cv2.drawContours(img_copy, [contour], 0, color, -1)



    cv2.imwrite('approximated_squares.png', img_copy)
    cv2.imwrite('approximated_centers.png', new_img)

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    find_shapes(img)
