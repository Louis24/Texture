import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern


def main():
    """ load video
        gray scale image
        crop -> edge detection
        classification """

    cap = cv2.VideoCapture("test.mp4")
    clf = joblib.load("svc.m")
    previous = None

    while True:
        # ORIGINAL
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        w = frame.shape[1]
        h = frame.shape[0]

        # SCALE
        scale = cv2.resize(frame, (w // 2, h // 2))

        # CROP
        crop = frame[281:631, 769:1217]  # (350, 448)

        # CLASSIFICATION
        lbp = local_binary_pattern(crop, 8, 1, 'default')
        max_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))  # 这里需要做一次增维
        hist = np.expand_dims(hist, axis=0)
        y_hat = clf.predict(hist)[0]

        if y_hat == 0:
            text = "Cloth"

        elif y_hat == 1:
            text = "Soil"

        else:
            text = "Stone"

        # EDGE DETECTION
        # sobelx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=5)
        # sobely = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=5)
        # laplacian = cv2.Laplacian(crop, cv2.CV_32F, ksize=5)
        # canny = cv2.Canny(crop, threshold1=100, threshold2=200, apertureSize=3)  # 边缘检测
        # dft = cv2.dft(np.float32(crop), flags=cv2.DFT_COMPLEX_OUTPUT)  # 傅里叶变换
        # dft_shift = np.fft.fftshift(dft)  # 平移到中心
        # magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))  # 频谱图
        # cv2.imshow('sobelx', sobelx)
        # cv2.imshow('sobely', sobely)
        # cv2.imshow('laplacian', laplacian)
        # cv2.imshow('dft', magnitude_spectrum)

        move = "Static"

        # 判定检测范围 当分类是土的时候 或者石头 判定移动状态

        current = crop
        if previous is None:
            previous = current

        else:
            current = cv2.GaussianBlur(crop, (5, 5), 0)  # 高斯
            delta = cv2.absdiff(previous, current)  # 帧间差分
            thresh = cv2.threshold(delta, 50, 255, cv2.THRESH_BINARY)[1]  # 阈值 thresh=[25,50] 比较好
            thresh = cv2.dilate(thresh, None, iterations=1)  # 膨胀 iterations=2 较好
            thresh = cv2.erode(thresh, None, iterations=1)  # 腐蚀
            contours, hierarchy = cv2.findContours(thresh, 2, 1)

            for c in contours:
                if cv2.contourArea(c, oriented=False) < 750:  # 设置敏感度
                    continue
                else:
                    move = "Dynamic"

            previous = current

        # SHOW
        cv2.imshow('scale', scale)
        cv2.putText(crop, text, org=(125, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255, 255, 255),
                    thickness=2)
        cv2.putText(crop, move, org=(125, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255, 255, 255),
                    thickness=2)
        cv2.imshow('crop', crop)
        # cv2.imshow('edge', canny)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
