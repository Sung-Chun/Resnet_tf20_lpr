import numpy as np
import cv2
import glob


from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

def imread_hangul_filename(filename):
    np_arr = np.fromfile(filename, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)


hough_threshold_list = [200, 250, 300]

def detect_lines(img_filename, th1, th2, show_canny=False):
    img = imread_hangul_filename(img_filename)
    src = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    dst = src.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    #canny = cv2.Canny(gray, 5000, 1500, apertureSize = 5, L2gradient = True)
    #lines = cv2.HoughLines(canny, 0.8, np.pi / 180, 150, srn = 100, stn = 200, min_theta = 0, max_theta = np.pi)

    # Canny edge detection
    canny = cv2.Canny(gray, th1, th2)
    if show_canny:
        cv2.imshow("canny", canny)

    # Hough line transform
    hough_done = False
    for hough_th in hough_threshold_list:
        lines = cv2.HoughLines(canny, 1, np.pi / 180, hough_th, None, 0, 0)
        if lines is None:
            continue
        print(f'  ## [{img_filename}] Num of lines: ({len(lines)})')
        if len(lines) < 4:
            lines = cv2.HoughLines(canny, 1, np.pi / 180, 100, None, 0, 0)
            if lines is None:
                continue
            print(f'  ## [{img_filename}] Num of lines: ({len(lines)})')

        if len(lines) < 40:
            hough_done = True
            break

    if not hough_done:
        print('********************** No hough lines found')
        return

    slope_list = list()
    point_list = list()
    for ll in lines:
        rho, theta = ll[0][0], ll[0][1]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho

        scale = src.shape[0] + src.shape[1]

        x1 = int(x0 + scale * -b)
        y1 = int(y0 + scale * a)
        x2 = int(x0 - scale * -b)
        y2 = int(y0 - scale * a)

        if x1 == x2:
            continue

        point_list.append((x1, y1, x2, y2))
        slope_list.append((y2 - y1) / (x2 - x1))
#        print(f'    s:{slope_list[i]:.3f}')
	
#        cv2.line(dst, (x1, y1), (x2, y2), color=(120, 255, 120), thickness=1)
        #cv2.circle(dst, (x0, y0), 3, (255, 0, 0), 5, cv2.FILLED)

#    print(slope_list)
    mu = np.mean(slope_list)
    sigma = np.var(slope_list)
    print(f'  -- mean({mu:.3f}), var({sigma:.5f})')

    # 분산 값이 1.0보다 크면 클러스터링을 한 번 해서 구분해준다.
    if sigma > 1.0:
        # define dataset
        model = KMeans(n_clusters=2)
        slope_data = np.array(slope_list).reshape(-1, 1)
        model.fit(slope_data)
        yhat = model.predict(slope_data)
        print(yhat)
        for i in range(len(slope_list)):
            p1 = (point_list[i][0], point_list[i][1])
            p2 = (point_list[i][2], point_list[i][3])
            if yhat[i] == 0:
                cv2.line(dst, p1, p2, color=(120, 255, 120), thickness=2)
            else:
                cv2.line(dst, p1, p2, color=(120, 120, 255), thickness=2)
    else:
        for i in range(len(slope_list)):
            p1 = (point_list[i][0], point_list[i][1])
            p2 = (point_list[i][2], point_list[i][3])
            cv2.line(dst, p1, p2, color=(120, 255, 120), thickness=2)

    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_contours(img_filename):
    img = imread_hangul_filename(img_filename)
    src = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        cv2.drawContours(src, [contours[i]], 0, (0, 0, 255), 2)
#        cv2.putText(src, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        print(i, hierarchy[0][i])

    cv2.imshow("src", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    img_file_folder = 'd:\\WORK\\LPR\\yolo_lp_corner_detect\\LP_images'
    img_file_list = glob.iglob(img_file_folder + '/*.jpg*', recursive=True)

    i = 0
    for img_file_path in img_file_list:
        print(f'***  [{img_file_path}]')
        detect_lines(img_file_path, 30, 100)
#        detect_lines(img_file_path, 50, 150)

#        detect_contours(img_file_path)

        i += 1

        if i > 1020:
            break
 
#    detect_lines('d:\\WORK\\LPR\\yolo_lp_corner_detect\\LP_images\\108로9959.JPG', 30, 100)
