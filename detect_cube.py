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

# http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

# calculate the angle formed by line (pt1, pt2) and (pt2, pt3) and the difference between
# the lengths of the two lines
# some code from http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def angle_and_lengths(pt1, pt2, pt3):
    v1 = np.array(pt1) - np.array(pt2)
    v2 = np.array(pt3) - np.array(pt2)
    length_diff = abs(np.linalg.norm(v1) - np.linalg.norm(v2))
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))), length_diff
    
# pick centers that actually correspond to cube faces
def pick_centers(centers):
    corners = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            for k in range(j+1, len(centers)):
                # check if the 3 points form a corner with a near 90 degree angle
                angle_1, ld_1 = angle_and_lengths(centers[i], centers[j], centers[k])
                angle_2, ld_2 = angle_and_lengths(centers[j], centers[i], centers[k])
                angle_3, ld_3 = angle_and_lengths(centers[i], centers[k], centers[j])

                angle_thresh = 5
                length_thresh = 5
                if angle_1 > 90-angle_thresh and angle_1 < 90+angle_thresh and ld_1 < length_thresh:
                    corners.append([centers[i], centers[j], centers[k]])
                elif angle_2 > 90-angle_thresh and angle_2 < 90+angle_thresh and ld_2 < length_thresh:
                    corners.append([centers[j], centers[i], centers[k]])
                elif angle_3 > 90-angle_thresh and angle_3 < 90+angle_thresh and ld_3 < length_thresh:
                    corners.append([centers[i], centers[k], centers[j]])

    return corners

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
    new_img = np.zeros((rows/10, cols/10, 3))
    centers = []
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
                centers.append((cX, cY))
            
            #(x, y, w, h) = cv2.boundingRect(approx)
	    #ar = w / float(h)
            #if ar >= 0.75 and ar <= 1.25:
            #    color = (random.randint(0, 255), random.randint(0,255), random.randint(0,255))
            #    cv2.drawContours(img_copy, [contour], 0, color, -1)

    corners = pick_centers(centers)
    for corner in corners:
        color = (random.randint(0, 255), random.randint(0,255), random.randint(0,255))
        cv2.line(new_img, corner[0], corner[1], color, 1)
        cv2.line(new_img, corner[1], corner[2], color, 1)

    cv2.imwrite('approximated_squares.png', img_copy)
    cv2.imwrite('approximated_centers.png', new_img)

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    find_shapes(img)
