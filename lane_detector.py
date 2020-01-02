import cv2
import numpy as np

def nothing(x):
    pass


#The length of detected lane lines

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 1 / 4)


    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

#cap = cv2.VideoCapture(0)
lane1=cv2.imread("lane5.jpeg",1)

cv2.namedWindow("lane")
cv2.createTrackbar("l1","lane",0,255,nothing)
cv2.createTrackbar("l2","lane",0,255,nothing)
cv2.createTrackbar("l3","lane",0,255,nothing)
cv2.createTrackbar("u1","lane",0,255,nothing)
cv2.createTrackbar("u2","lane",0,255,nothing)
cv2.createTrackbar("u3","lane",0,255,nothing)


while(1):

    #ret, lane1 = cap.read()
    lane1 = cv2.resize(lane1, None, fx=.5, fy=.5, interpolation=cv2.INTER_CUBIC)
    print(lane1)


    l1=cv2.getTrackbarPos("l1","lane")
    lo1 = l1
    l2 = cv2.getTrackbarPos("l2", "lane")
    lo2 = l2
    l3 = cv2.getTrackbarPos("l3", "lane")
    lo3 = l3
    u1 = cv2.getTrackbarPos("u1", "lane")
    up1 = u1
    u2 = cv2.getTrackbarPos("u2", "lane")
    up2 = u2
    u3 = cv2.getTrackbarPos("u3", "lane")
    up3 = u3

    hsv = cv2.cvtColor(lane1,cv2.COLOR_BGR2HSV)

    lower = np.array([lo1,lo2,lo3])
    upper = np.array([up1,up2,up3])

    mask = cv2.inRange(hsv,lower,upper)

    res=cv2.bitwise_and(lane1 , lane1 , mask=mask)
    edges = cv2.Canny(mask, 200, 400)
    print(edges)

    height, width = edges.shape
    mask1 = np.zeros_like(edges)

    # cropping the required lanes
    polygon = np.array([[
        (0,271),
        (253,64),
        (381,67),
        (width,214),
        (width,height),
        (0,height)
    ]], np.int32)

    cv2.fillPoly(mask1, polygon, 255)
    cropped = cv2.bitwise_and(edges, mask1)



    #detection of line segments

    rho = 1 # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(edges, rho, angle, min_threshold,
                                    np.array([]), minLineLength=8, maxLineGap=4)






    #average single line in each side

    lane_lines = []


    height, width, _ = lane1.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 2
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                #logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(lane1, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(lane1, right_fit_average))


    #displaying lane lines in the image

    line_image = np.zeros_like(lane1)
    if lane_lines is not None:
        for line in lane_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0,255,0), 5)
    line_image = cv2.addWeighted(lane1, 0.8, line_image, 1, 1)


    #cv2.imshow("lane lines", line_image)


    #steering angle

    if lane_lines[0][0] is None or lane_lines[1][0] is None:
        if lane_lines[0][0] is not None:
            x1,y1,x2,y2 = lane_lines[0][0]
            x_offset = x2-x1
            y_offset = int(height/2)

        else:
            x1, y1, x2, y2 = lane_lines[1][0]
            x_offset = x1-x2
            y_offset = int(height/2)

    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)

    angle_to_mid_radian = np.arctan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / 3.14)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90 # this is the steering angle needed by picar front wheel
    print(steering_angle)



    #displaying heading line

    heading_image = np.zeros_like(lane1)
    height, width, _ = lane1.shape

    steering_angle_radian = steering_angle / 180.0 * 3.14
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / np.tan(steering_angle_radian))
    y2 = int(height / 4)

    cv2.line(heading_image, (x1, y1), (x2, y2), (0,0,255), 3)
    heading_image = cv2.addWeighted(line_image, 0.8, heading_image, 1, 1)






    cv2.imshow("res",heading_image)
    cv2.imshow("mask1", mask1)
    cv2.imshow("edges", edges)
    cv2.imshow("cropped", cropped)


    k=cv2.waitKey(0) & 0xFF
    if k ==27:
        break




print(lane_lines)
cv2.destroyAllWindows()







