import cv2, sys, time, random
import numpy as np
import math
from sklearn.cluster import MiniBatchKMeans
from skimage.segmentation import slic
from skimage import io

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

def calc_intersection(p1, p2):
    # we only want perpendicular intersections
    diff = abs(p1[1] - p2[1])
    thresh = math.pi/32
    if diff > math.pi/2 + thresh or diff < math.pi/2 - thresh:
        return [[float("inf")], [float("inf")]]
    
    A = np.array([[math.cos(p1[1]), math.sin(p1[1])],
                  [math.cos(p2[1]), math.sin(p2[1])]])
    # check this otherwise we get a numpy error since its a matrix with 2 duplicate columns
    if A[0][0] == A[1][0] and A[0][1] == A[1][1]:
        return [[float("inf")], [float("inf")]]
    
    b = np.array([[p1[0]], [p2[0]]])
    
    # Solve AX = b with X = A^-1b
    A_inv = np.linalg.inv(A)
    X = np.dot(A_inv, b)
    # X = [[x], [y]], reshape to [x, y]
    return X.ravel()

def find_clusters(points):
    dist_thresh = 30
    clusters = []
    for p in points:
        found_cluster = False
        for i in range(len(clusters)):
            center = clusters[i][0]
            if math.hypot(center[0] - p[0], center[1] - p[1]) < dist_thresh:
                clusters[i][1].append(p)
                clusters[i][0] = np.mean(clusters[i][1], axis=0)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([p, [p]])
    return np.array(clusters)[:, 0]
        

def find_all_intersections(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.png', gray)
    #edges = cv2.Canny(gray, 0, 100)
    edges = auto_canny(img)
    cv2.imwrite('edges.png', edges)
    rows,cols,channels = img.shape
    lines = cv2.HoughLines(edges,1,np.pi/180,40)
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
    cv2.imwrite('found_lines.png', img)
    intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            intersect = calc_intersection(lines[i][0], lines[j][0])
            if intersect[0] >= 0 and intersect[0] <= cols and intersect[1] >= 0\
               and intersect[1] <= rows:
                intersections.append(intersect)
    for intersect in intersections:
        cv2.circle(img,(int(intersect[0]), int(intersect[1])),3,255,-1)

    #clusterer = AffinityPropagation(damping=0.95)
    #clusterer.fit(intersections)
    #centers = clusterer.cluster_centers_
    centers = find_clusters(intersections)
    for center in centers:
        cv2.circle(img, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
    return img

#def find_edges(img):
    #blur = cv2.blur(img, (3,3))
    #rows,cols,channels = blur.shape
    #imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(blur,100,200)
#    edges = auto_canny(img)
#    cv2.imwrite('edges.png', edges)
    #kernel = np.ones((5,5),np.uint8)
    #edges = cv2.dilate(edges, kernel)

    """
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength,maxLineGap)

    for line in lines:
        #for x1,y1,x2,y2 in lines:
        x1,y1,x2,y2=line[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)"""
    
#    return edges


def find_corners(img):
    #blur = cv2.blur(img, (3,3))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
    kp = fast.detect(img,None)
    img2 = cv2.drawKeypoints(img, kp, color=(255,0,0), outImage=img)


    # shi-tomasi corner dectection
    """corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)
    """
    
    # harris corner detection
    """gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    img[dst>0.001*dst.max()]=[0,0,255]"""
#    return img2

def find_squares(img):
    yellow = [(15,30), (125,145), (140,170)]
    rows,cols,shape = img.shape
    new_img = np.zeros((rows, cols)) * 255
    for i in range(cols):
        for j in range(rows):
            pixel = img[j][i]
            if pixel[0] > yellow[0][0] and pixel[0] < yellow[0][1] and pixel[1] > yellow[1][0] \
               and pixel[1] < yellow[1][1] and pixel[2] > yellow[2][0] and pixel[2] < yellow[2][1]:
                new_img[j][i] = 255
    return new_img

def segment_img(img):
    #from skimage.data import astronaut
    #img = astronaut()
    rows,cols,channels = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_small = cv2.resize(img, (cols/10, rows/10))
    #clusterer = MiniBatchKMeans(n_clusters = 40)
    #pixels = []
    #for i in range(cols):
    #    for j in range(rows):
    #        pixels.append([i, j])
    #clusters = clusterer.fit(gray)
    #centers = clusters.cluster_centers_
    #cv2.imwrite('clusters.png', centers)
    

    #cv2.imwrite('small_img.png', img_small)
    segments = slic(img_small, n_segments=60, compactness=10)
    io.imshow(segments)
    io.show()
    return segments


def find_lines(img):
    rows,cols,channels = img.shape
    img_small = cv2.resize(img, (cols/5, rows/5), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplac = cv2.Laplacian(gray,cv2.CV_8U)
    laplac = cv2.normalize(laplac, laplac, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #adapt_thresh = cv2.adaptiveThreshold(laplac,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                                     cv2.THRESH_BINARY,5,2)
    #adapt_thresh = auto_canny(laplac)
    ret,adapt_thresh = cv2.threshold(laplac,20,255,cv2.THRESH_BINARY)
    #adapt_thresh = cv2.erode(adapt_thresh, np.ones((2,2)))
    #adapt_thresh = cv2.morphologyEx(adapt_thresh, cv2.MORPH_CLOSE, np.ones((2,2)), iterations=4)
    #edges = auto_canny(img_small)
    #edges = cv2.Canny(cv2.blur(img, (3,3)), 0, 50)
    #edges = cv2.resize(edges, (cols,rows))

    #edges_dilated = cv2.dilate(edges, np.ones((5,5)), iterations=1)
    #edges_opened = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5)))    
    """
    minLineLength = 20
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges_dilated, 1, np.pi/180, 5, minLineLength, maxLineGap)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
    """
    lines = cv2.HoughLines(adapt_thresh,1,np.pi/180,200)
    if lines is not None:
        for line in lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))*5
            y1 = int(y0 + 1000*(a))*5
            x2 = int(x0 - 1000*(-b))*5
            y2 = int(y0 - 1000*(a))*5

            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
    
    
    return img, adapt_thresh, laplac

def lbp(image):
    rows,cols,channels = image.shape
    small_img = cv2.resize(image, (cols/5,rows/5), interpolation=cv2.INTER_AREA)
    num_points = 10
    radius = 3
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, num_points,
			               radius, method="uniform")
    lbp = cv2.normalize(lbp, lbp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #ret, thresh = cv2.threshold(lbp, 127,255,cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(lbp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                   cv2.THRESH_BINARY,11,2)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray,cv2.CV_64F)
    laplacian = cv2.normalize(laplacian, laplacian, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #ret,thresh = cv2.threshold(laplacian, 150,255, cv2.THRESH_BINARY)
    laplacianx64f = cv2.Laplacian(gray,cv2.CV_64F)
    abs_laplacian64f = np.absolute(laplacianx64f)
    laplacian_8u = np.uint8(abs_laplacian64f)

    ret,thresh = cv2.threshold(laplacian_8u, 20,255, cv2.THRESH_BINARY)
    eroded = cv2.erode(thresh, np.ones((2,2)))
    #thresh = cv2.adaptiveThreshold(laplacian_8u,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                               cv2.THRESH_BINARY,11,2)
    
    lines = cv2.HoughLines(eroded,1,np.pi/180,25)
    if lines is not None:
        for line in lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(small_img,(x1,y1),(x2,y2),(255,0,0),1)
    

    return small_img



if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    #start = time.time()
    #edges = find_edges(img)
    #cv2.imwrite('edges.png', edges)
    #end = time.time()
    #print 'took', end - start
    #corners = find_corners(img)
    #cv2.imwrite('cube_edges.png', edges)
    #cv2.imwrite('cube_corers.png', corners)
    #squares = find_squares(img)
    #cv2.imwrite('squares.png', squares)

    
    seg = segment_img(img)
    cv2.imwrite('segmented.png', seg)

    """
    rows,cols,channels = img.shape
    img_small = cv2.resize(img, (cols/10, rows/10))

    points = find_all_intersections(img_small)
    cv2.imwrite('intersections.png', points)
    """
