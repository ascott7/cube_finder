import cv2, sys, time, random, os
import math, copy, itertools
import numpy as np

from skimage import feature

class CubeDetector:
    def __init__(self):
        self.img = None
        self.rotated_img = None
        self.angle = None
        self.rows = None
        self.cols = None
        self.resize_factor = 4
        # the orthogonal distance from the center of the piece to its perimeter
        # i.e. height/2
        self.size = None
        self.last_x = None
        self.last_y = None
        self.last_wh = None

    # http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    def _auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        lower=60
        upper=100
        edged = cv2.Canny(image, lower, upper)
     
        # return the edged image
        return edged

    def _find_bounding_square(self, contour, img_area):
        peri = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        if float(area)/img_area < 0.001 or float(area)/img_area > 0.02:
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

    def _contour_center(self, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        M = cv2.moments(approx)
        if M["m00"] == 0:
            return (float('inf'), float('inf'))
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    def _prune_centers(self, centers, img_width):
        while True:
            num_centers = len(centers)
            centers = self._prune_centers_helper(centers, img_width)
            if num_centers == len(centers):
                return centers
        return []

    def _prune_centers_helper(self, centers, img_width):
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

    def _group_centers(self, centers, img_width):
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

    # based on http://gis.stackexchange.com/questions/23587/how-do-i-rotate-the-polygon-about-an-anchor-point-using-python-script
    def _rotate_pts(self, pts,center,angle):
        '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
        pts = np.array(pts)
        center = np.array(center)
        return np.dot(pts-center,np.array([[math.cos(angle),math.sin(angle)],
                                           [-math.sin(angle),math.cos(angle)]]))+center

    def _create_straight_grid(self, points):
        cube_dim = 3
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

        return grid_points

    def _create_tilted_grid(self, points):
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
            grid_points[0][2] = sorted_y.pop(0)
            grid_points[2][0] = sorted_y.pop(-1)
            grid_points[1][0] = sorted_x.pop(0)
            grid_points[1][2] = sorted_x.pop(-1)
            grid_points[0][1] = sorted_y.pop(0)
            grid_points[2][1] = sorted_y.pop(-1)
            for p in sorted_y:
                if p not in grid_points[0] and p not in grid_points[1] and p not in grid_points[2]:
                    grid_points[1][1] = p
                    break
        else:
            grid_points[2][0] = sorted_x.pop(0)
            grid_points[0][2] = sorted_x.pop(-1)
            grid_points[0][0] = sorted_y.pop(0)
            grid_points[2][2] = sorted_y.pop(-1)
            grid_points[1][0] = sorted_x.pop(0)
            grid_points[1][2] = sorted_x.pop(-1)
            grid_points[0][1] = sorted_y.pop(0)
            grid_points[2][1] = sorted_y.pop(-1)
            for p in sorted_y:
                if p not in grid_points[0] and p not in grid_points[1] and p not in grid_points[2]:
                    grid_points[1][1] = p
                    break
        return grid_points
    
    def _straight_grid_score(self, grid_points):
        line_distance_std_thresh = 5
        cube_dim = 3
        line_distances = self._get_line_distances(grid_points)
        if np.std(line_distances) > line_distance_std_thresh:
            return float('inf')

        grid_score = 0
        for i in range(cube_dim):
            grid_score += np.std([grid_points[0][i][0], grid_points[1][i][0], grid_points[2][i][0]])
        for j in range(cube_dim):
            grid_score += np.std([grid_points[j][0][1], grid_points[j][1][1], grid_points[j][2][1]])
        return grid_score

    def _get_best_grid(self, points):
        # try assuming a straight grid
        straight_grid = self._create_straight_grid(points)
        # straighten the straight grid according to the top row to make scores more accurate
        x,y = zip(*straight_grid[0])
        m,b = np.polyfit(x,y,1)
        straight_angle = math.atan(m)
        straightened_straight_grid =  self._rotate_pts(straight_grid, straight_grid[1][1],
                                                       -straight_angle)
        straight_grid_score = self._straight_grid_score(straightened_straight_grid)

        # try assuming a tilted grid
        tilted_grid = self._create_tilted_grid(points)
        # straighten the tilted grid according to the top row to use the same grid score algorithm
        x,y = zip(*tilted_grid[0])
        m,b = np.polyfit(x,y,1)
        tilt_angle = math.atan(m)
        straightened_tilted_grid =  self._rotate_pts(tilted_grid, tilted_grid[1][1], -tilt_angle)
        tilted_grid_score = self._straight_grid_score(straightened_tilted_grid)

        # pick the option that gave a better score
        if tilted_grid_score < straight_grid_score:
            return tilted_grid_score, tilted_grid, tilt_angle
        return straight_grid_score, straight_grid, straight_angle
        

    def _remove_small_contours(self, edges):
        edges = cv2.dilate(edges, np.ones((2,2)))
        # find contours, which should include squares from the cube faces
        contour_img, contours, hierarchy = cv2.findContours(edges.copy(),\
                                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tmp = self.img.copy()
        for contour in contours:
            color = (random.randint(0, 255), random.randint(0,255), random.randint(0,255))
            cv2.drawContours(tmp, [contour], 0, color, 1)
        cv2.imwrite('temp.png', tmp)
        cnt_area_thresh = 100
        for contour in contours:
            #contour_area = cv2.contourArea(contour)
            # if the contour is small, remove it completely
            bounding_rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(bounding_rect)
            box = np.int0(box)
            bounding_line1 = math.sqrt((box[0][0]-box[1][0])**2 + (box[0][1]-box[1][1])**2)
            bounding_line2 = math.sqrt((box[1][0]-box[2][0])**2 + (box[1][1]-box[2][1])**2)
            bounding_area = bounding_line1*bounding_line2

            if bounding_area < cnt_area_thresh:
                cv2.drawContours(edges, [contour], -1, (0,0,0), -1)

        return edges
        
    def _find_shapes(self):
        edges = self._auto_canny(self.img, 0.2)
        edges_pruned = self._remove_small_contours(edges.copy())
        edges_dilated = cv2.dilate(edges_pruned, np.ones((4,4)), iterations=2)
        edges_dilated = 255-edges_dilated

        # find contours, which should include squares from the cube faces
        contour_img, contours, hierarchy = cv2.findContours(edges_dilated.copy(),\
                                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # draw the contours to help with debugging
        contour_img = self.img.copy()
        for contour in contours:
            color = (random.randint(0, 255), random.randint(0,255), random.randint(0,255))
            cv2.drawContours(contour_img, [contour], 0, color, 1)

        squares_found = self.img.copy()
        centers = []
        img_area = self.rows/self.resize_factor*self.cols/self.resize_factor
        # for each contour check if the contour seems to be a square shape
        for contour in contours:
            detected_square = self._find_bounding_square(contour, img_area)
            if detected_square is not None:
                cX,cY = self._contour_center(contour)
                cX,cY = cX,cY
                centers.append((cX,cY))
                # draw the square shaped contours to help with debugging
                color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                cv2.drawContours(squares_found, [contour], 0, color, -1)

        # remove centers that aren't close to at least 3 other centers
        pruned_centers = self._prune_centers(centers, self.cols/self.resize_factor)

        # cluster the centers into groups where every center in each group is within X distance
        # of the other centers in the group (this function is probably pretty inefficient right
        # now so we should try to fix that)
        grouped_centers = self._group_centers(pruned_centers, self.cols/self.resize_factor)
        
        chosen_contour_img = self.img.copy()
        grid_scores = []
        # in each of the groups of centers, look for a grid shape
        # this current logic below assumes only 1 match will exist, we will want to adjust that
        for gc in grouped_centers:
            # we will want to find a way to handle more than 13-14 centers in way that doesn't
            # exponentially explode, for now we simply don't try
            if len(gc) >= 9 and len(gc) <= 13:
                for c in itertools.combinations(gc, 9):
                    # get the grid scores and the grids
                    grid_scores.append(self._get_best_grid(c))
                grid_scores.sort()
        found_grid,angle,score = [], 0, float('inf')
        if len(grid_scores) > 0 and grid_scores[0][0] < 10:
            score,found_grid,angle = grid_scores[0]
            print score
            for row in found_grid:
                for col in row:
                    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                    cv2.rectangle(chosen_contour_img, (col[0]-4, col[1]-4), (col[0]+4, col[1]+4), color, -1)
        return found_grid, angle, [cv2.resize(edges, (self.cols,self.rows)), cv2.resize(edges_pruned, (self.cols,self.rows)), cv2.resize(edges_dilated, (self.cols,self.rows)),\
            cv2.resize(contour_img, (self.cols,self.rows)), cv2.resize(squares_found,(self.cols,self.rows)),\
            cv2.resize(chosen_contour_img, (self.cols,self.rows))]

    def _find_size(self, grid):
        line_distances = self._get_line_distances(grid)
        self.size = int((np.mean(line_distances) / 2) * 0.8)

    def _get_line_distances(self, grid_points):
        grid_points = np.array(grid_points)
        line_distances = []
        line_distances.extend((grid_points[1] - grid_points[0])[:,1])
        line_distances.extend((grid_points[2] - grid_points[1])[:,1])
        line_distances.extend(grid_points[:,:,0].T[1] - grid_points[:,:,0].T[0])
        line_distances.extend(grid_points[:,:,0].T[2] - grid_points[:,:,0].T[1])
        return [abs(x) for x in line_distances]

    """
    def _get_color(self, grid):
        # Hue range [0, 179], Saturation range [0, 255], Value range [0, 255]
        # convert to HSV
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        for line in grid:
            for row, col in line:
                piece = hsv[col-self.size:col+self.size, row-self.size:row+self.size]
                print "piece"
                print (piece)
                print "col, row"
                print col
                print row
                print "size"
                print self.size
                if self._is_red(piece):
                    print "is red"
        return

    def _is_blue(self, piece):
        blue = np.uint8([[[0,0,255]]])
        hue, saturation, value = np.ndarray.flatten(cv2.cvtColor(blue, cv2.COLOR_BGR2HSV))
        blue_lower = np.array([hue-10, 100, 100])
        blue_upper = np.array([hue+10, 255, 255])

        # threshold the image to get only blue colors
        mask = cv2.inRange(piece, blue_lower, blue_upper)

        print (mask)

        return True

    def _is_red(self, piece):
        red = np.uint8([[[255,0,0]]])
        hue, saturation, value = np.ndarray.flatten(cv2.cvtColor(red, cv2.COLOR_BGR2HSV))
        red_lower = np.array([hue-10, 100, 100])
        red_upper = np.array([hue+10, 255, 255])

        # threshold the image to get only red colors
        mask = cv2.inRange(piece, red_lower, red_upper)
        # res = cv2.bitwise_and(frame, frame, mask = mask)

        print (mask)

        return True
    """

    def get_face(self, img):
        self.rows, self.cols, channels = img.shape
        self.resize_factor = 4
        # use a smaller version of the image
        self.img = cv2.resize(img, (self.cols/self.resize_factor, self.rows/self.resize_factor), interpolation=cv2.INTER_AREA)
        #if self.last_x:
        #    print self.last_x, self.last_y, self.last_wh
        #    full_img = self.img
        #    self.img = self.img[max(0,self.last_y-self.last_wh):min(self.last_y+self.last_wh,self.rows),
        #                        max(0,self.last_x-self.last_wh):min(self.last_x+self.last_wh,self.cols)]
         #   cv2.imwrite('cropped_img.png', self.img)
        found_grid, angle, images = self._find_shapes()
        # if we found a grid
        if len(found_grid) > 2:
            #self.last_x = found_grid[1][1][0]
            #self.last_y = found_grid[1][1][1]
            #self.last_wh = found_grid[1][1][0] - found_grid[0][0][0] + 30 # add some padding
            # rotate the grid so the top edge is flat
            straightened_grid =  self._rotate_pts(found_grid, found_grid[1][1], -angle)
            # rotate the image so that the grid points are in their same locations as before
            M = cv2.getRotationMatrix2D(((found_grid[1][1][0],found_grid[1][1][1])),
                                        math.degrees(angle),1)
            self.rotated_img = cv2.warpAffine(self.img,M,(self.cols/self.resize_factor,
                                                          self.rows/self.resize_factor))
            # draw the centers on the rotated image for debugging purposes
            rotated_centers_img = self.rotated_img.copy()
            for row in straightened_grid:
                for col in row:
                    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                    cv2.rectangle(rotated_centers_img, (int(col[0])-4, int(col[1])-4), (int(col[0])+4, int(col[1])+4), color, -1)
        else:
            rotated_centers_img = self.img.copy()
            #self.last_x = None
            #self.last_y = None
            #self.last_wh = None
        images.append(cv2.resize(rotated_centers_img, (self.cols,self.rows)))


        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        colors = []
        for row in found_grid:
            for col in row:
                colors.append(hsv[col[1], col[0]])
        print colors
        # self._find_size(grid)
        # colored_grid = self._get_color(grid)
        # return colored_grid
        # return colored_grid, images
        return found_grid, angle, images

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
                grid,angle,images = cube_detector.get_face(img)
                end = time.time()
                print "took", end - start, "seconds"
                runtimes.append(end-start)
                """cv2.imwrite('detected_lines'+filename+'.png', images[0])
                cv2.imwrite('adapt_thresh'+filename+'.png', images[1])
                cv2.imwrite('laplacian'+filename+'.png', images[2])"""     
                cv2.imwrite('auto_edges'+filename+'.png', images[0])
                cv2.imwrite('auto_edges_pruned'+filename+'.png', images[1])
                cv2.imwrite('auto_edges_dilated'+filename+'.png', images[2])
                cv2.imwrite('contour'+filename+'.png', images[3])
                cv2.imwrite('found_squares'+filename+'.png', images[4])
                cv2.imwrite('approximated_squares'+filename+'.png', images[5])
                cv2.imwrite('rotated_approx_centers'+filename+'.png', images[6])
