import cv2, sys, time, random, os
import math, copy, itertools
import numpy as np

from skimage import feature

class CubeDetector:
    def __init__(self):
        self.last_x = None
        self.last_y = None

    # http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    def auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
     
        # return the edged image
        return edged

    def find_bounding_square(self, contour, img_area):
        peri = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        # we should probably change these #s to be some fraction of the rows/cols of the input image
        if float(area)/img_area < 0.002 or float(area)/img_area > 0.02:
            return None
        
        bounding_rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(bounding_rect)
        box = np.int0(box)
        bounding_line1 = math.sqrt((box[0][0]-box[1][0])**2 + (box[0][1]-box[1][1])**2)
        bounding_line2 = math.sqrt((box[1][0]-box[2][0])**2 + (box[1][1]-box[2][1])**2)
        bounding_area = bounding_line1*bounding_line2

        ap_ratio = (float(peri)/4) / math.sqrt(area)
        a_br_diff = bounding_area - area
        ap_thresh = 0.3
        a_br_thresh = 150
        if abs(1.0 - ap_ratio) < ap_thresh and a_br_diff < a_br_thresh:
            return box
        return None

    def contour_center(self, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        M = cv2.moments(approx)
        if M["m00"] == 0:
            return (float('inf'), float('inf'))
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    def prune_centers(self, centers, img_width):
        while True:
            num_centers = len(centers)
            centers = self.prune_centers_helper(centers, img_width)
            if num_centers == len(centers):
                return centers
        return []

    def prune_centers_helper(self, centers, img_width):
        dist_thresh = 0.25
        pruned_centers = []
        if len(centers) < 4:
            return []

        for ci in centers:
            distances = []
            for cj in centers:
                distances.append(abs(float(ci[0]) - cj[0]) + abs(float(ci[1]) - cj[1]))
            distances.sort()
            # check if there are 3 nearby points
            if distances[3]/img_width < dist_thresh:
                pruned_centers.append(ci)
        return pruned_centers

    def group_centers(self, centers, img_width):
        if len(centers) <= 1:
            return centers
        dist_thresh = 0.12
        clusters = []
        for p in centers:
            found_cluster = False
            for i in range(len(clusters)):
                center = clusters[i][0]
                if math.hypot(center[0] - p[0], center[1] - p[1])/img_width < dist_thresh:
                    clusters[i][1].append(p)
                    clusters[i][0] = np.mean(clusters[i][1], axis=0)
                    found_cluster = True
                    break
            if not found_cluster:
                clusters.append([p, [p]])

        # take out the cluster centers
        clusters = [x[1] for x in clusters]
        # try to recombine clusters that are now close to one another
        n_clusters = float('inf')
        while n_clusters != len(clusters) and len(clusters) > 1:
            n_clusters = len(clusters)
            full_break = False
            for i in range(len(clusters)):
                if full_break:
                    break
                for j in range(i+1, len(clusters)):
                    if full_break:
                        break
                    for c1 in clusters[i]:
                        if full_break:
                            break
                        for c2 in clusters[j]:
                            # if 2 clusters are close to one another, merge them
                            if math.hypot(c1[0] - c2[0], c1[1] - c2[1])/img_width < dist_thresh:
                                clusters[i].extend(copy.deepcopy(clusters[j]))
                                clusters.pop(j)
                                full_break = True
                                break
        return clusters

    def tilted_grid_score(self, points):
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        min_y = min(y_vals)
        min_y_index = y_vals.index(min_y)
        max_y = max(y_vals)
        max_y_index = y_vals.index(max_y)

        max_y_x = x_vals[max_y_index]
        min_y_x = x_vals[min_y_index]

        tilted_left = False
        # cube tilted slightly to the left, otherwise it's tilted slightly to the right
        if max_y_x < min_y_x:
            tilted_left = True

        grid_points = [[[0,0] for x in range(3)] for y in range(3)]
        sorted_y = sorted(points, key=lambda x: x[1])
        sorted_x = sorted(points, key=lambda x: x[0])
        if tilted_left:
            grid_points[0][0] = sorted_x.pop(0)
            grid_points[2][2] = sorted_x.pop(-1)
            grid_points[2][0] = sorted_y.pop(0)
            grid_points[0][2] = sorted_y.pop(-1)
            grid_points[0][1] = sorted_x.pop(0)
            grid_points[2][1] = sorted_x.pop(-1)
            grid_points[1][0] = sorted_y.pop(0)
            grid_points[1][2] = sorted_y.pop(-1)
            for p in sorted_y:
                if p not in grid_points[0] and p not in grid_points[1] and p not in grid_points[2]:
                    grid_points[1][1] = p
                    break
        else:
            grid_points[0][2] = sorted_x.pop(0)
            grid_points[2][0] = sorted_x.pop(-1)
            grid_points[0][0] = sorted_y.pop(0)
            grid_points[2][2] = sorted_y.pop(-1)
            grid_points[0][1] = sorted_x.pop(0)
            grid_points[2][1] = sorted_x.pop(-1)
            grid_points[1][0] = sorted_y.pop(0)
            grid_points[1][2] = sorted_y.pop(-1)
            for p in sorted_y:
                if p not in grid_points[0] and p not in grid_points[1] and p not in grid_points[2]:
                    grid_points[1][1] = p
                    break
        # check to make sure we have 9 unique points in the grid (if we don't then the
        # tilted grid assumption must have been wrong and we will get errors when trying to
        # calculate the residuals)
        found_pts = set()
        for col in grid_points:
            for row in col:
                found_pts.add(row)
        if len(found_pts) != 9:
            return float('inf'), []

        # calculate the score as the residuals from a best fit line through each row and column
        # TODO: account for how parallel the lines are to one another and whether they are the
        # same distance apart from one another
        grid_score = 0
        for i in range(3):
            x,y = zip(*grid_points[i])
            _,resid,_,_,_ = np.polyfit(x,y,1,full=True)
            if len(resid) == 1:
                grid_score += resid
        for j in range(3):
            x = [grid_points[0][j][0], grid_points[1][j][0], grid_points[2][j][0]]
            y = [grid_points[0][j][1], grid_points[1][j][1], grid_points[2][j][1]]
            _,resid,_,_,_ = np.polyfit(x,y,1,full=True)
            if len(resid) == 1:
                grid_score += resid

        return grid_score, grid_points

    def straight_grid_score(self, points):    
        cube_dim = 3
        line_distance_std_thresh = 3.0
        grid_points = [[[0,0] for x in range(3)] for y in range(3)] 
        # try assuming a straight 3x3 cube
        # group points by their y values
        sorted_y = sorted(points, key=lambda x: x[1])
        for i in range(cube_dim**2):
            grid_points[i/cube_dim][i%cube_dim][0] = sorted_y[i][0]
            grid_points[i/cube_dim][i%cube_dim][1] = sorted_y[i][1]
        # sort each row of points by their x values
        for i in range(cube_dim):
            sorted_pts = sorted(copy.deepcopy(grid_points[i]), key=lambda x: x[0])
            for j in range(cube_dim):
                grid_points[i][j][0] = sorted_pts[j][0]
                grid_points[i][j][1] = sorted_pts[j][1]

        grid_points = np.array(grid_points)
        line_distances = []
        line_distances.extend((grid_points[1] - grid_points[0])[:,0])
        line_distances.extend((grid_points[2] - grid_points[1])[:,0])
        line_distances.extend(grid_points[:,:,1].T[1] - grid_points[:,:,1].T[0])
        line_distances.extend(grid_points[:,:,1].T[2] - grid_points[:,:,1].T[1])
        if np.std(line_distances) > line_distance_std_thresh:
            return float('inf'), []

        grid_score = 0
        for i in range(cube_dim):
            grid_score += np.std([grid_points[0][i][0], grid_points[1][i][0], grid_points[2][i][0]])
        for j in range(cube_dim):
            grid_score += np.std([grid_points[j][0][1], grid_points[j][1][1], grid_points[j][2][1]])
        return grid_score, grid_points

    def find_shapes(self, img):
        rows,cols,channels = img.shape
        resize_factor = 4
        # use a smaller version of the image
        img_small = cv2.resize(img, (cols/resize_factor,rows/resize_factor), interpolation=cv2.INTER_AREA)
        #img_small = cv2.resize(cv2.blur(img, (3,3)), (cols/5, rows/5), interpolation=cv2.INTER_AREA)
        edges = self.auto_canny(img_small, 0.2)
        edges_dilated = cv2.dilate(edges, np.ones((5,5)), iterations=1)
        edges_dilated = 255-edges_dilated

        # find contours, which should include squares from the cube faces
        contour_img, contours, hierarchy = cv2.findContours(edges_dilated.copy(),\
                                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # draw the contours to help with debugging
        contour_img = img_small.copy()
        for contour in contours:
            color = (random.randint(0, 255), random.randint(0,255), random.randint(0,255))
            cv2.drawContours(contour_img, [contour], 0, color, 1)

        squares_found = img_small.copy()
        centers = []
        img_area = rows/resize_factor*cols/resize_factor
        # for each contour check if the contour seems to be a square shape
        for contour in contours:
            detected_square = self.find_bounding_square(contour, img_area)
            if detected_square is not None:
                cX,cY = self.contour_center(contour)
                cX,cY = cX,cY
                centers.append((cX,cY))
                # draw the square shaped contours to help with debugging
                color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                cv2.drawContours(squares_found, [contour], 0, color, -1)

        # remove centers that aren't close to at least 3 other centers
        pruned_centers = self.prune_centers(centers, cols/resize_factor)

        # cluster the centers into groups where every center in each group is within X distance
        # of the other centers in the group (this function is probably pretty inefficient right
        # now so we should try to fix that)
        grouped_centers = self.group_centers(pruned_centers, cols/resize_factor)
        
        chosen_contour_img = img_small.copy()
        grid_scores = []
        # in each of the groups of centers, look for a grid shape
        # this current logic below assumes only 1 match will exist, we will want to adjust that
        for gc in grouped_centers:
            # we will want to find a way to handle more than 13-14 centers in way that doesn't
            # exponentially explode, for now we simply don't try
            if len(gc) >= 9 and len(gc) <= 14:
                for c in itertools.combinations(gc, 9):
                    # get the grid scores and the grids
                    grid_scores.append(self.straight_grid_score(c))
                    grid_scores.append(self.tilted_grid_score(c))
                grid_scores.sort()
        found_grid = []
        if len(grid_scores) > 0 and grid_scores[0][0] < 50:
            found_grid = grid_scores[0][1]
            for row in grid_scores[0][1]:
                for col in row:
                    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                    cv2.rectangle(chosen_contour_img, (col[0]-4, col[1]-4), (col[0]+4, col[1]+4), color, -1)
        return found_grid, (cv2.resize(edges, (cols,rows)), cv2.resize(edges_dilated, (cols,rows)),\
            cv2.resize(contour_img, (cols,rows)), cv2.resize(squares_found,(cols,rows)),\
            cv2.resize(chosen_contour_img, (cols,rows)))


if __name__ == '__main__':
    cube_detector = CubeDetector()
    runtimes = []
    i = 1
    output_base = 'output_images'
    while os.path.exists(output_base + str(i)):
        i += 1
    os.mkdir('output_images' + str(i))
    os.chdir('output_images' + str(i))
    for subdir, dirs, files in os.walk(os.path.join('..', sys.argv[1])):
        for f in files:
            filename, ext = os.path.splitext(f)
            if ext == '.png':
                print filename
                img = cv2.imread(os.path.join(subdir,f))
                start = time.time()
                images = cube_detector.find_shapes(img)[1]
                end = time.time()
                print "took", end - start, "seconds"
                runtimes.append(end-start)
                """cv2.imwrite('detected_lines'+filename+'.png', images[0])
                cv2.imwrite('adapt_thresh'+filename+'.png', images[1])
                cv2.imwrite('laplacian'+filename+'.png', images[2])"""     
                cv2.imwrite('auto_edges'+filename+'.png', images[0])
                cv2.imwrite('auto_edges_dilated'+filename+'.png', images[1])
                cv2.imwrite('contour'+filename+'.png', images[2])
                cv2.imwrite('found_squares'+filename+'.png', images[3])
                cv2.imwrite('approximated_squares'+filename+'.png', images[4])
