# -*-coding: utf-8-*-

import cv2
import numpy as np
import math


def stretch(img):
    max = float(img.max())
    min = float(img.min())

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = (255 / (max - min)) * img[i, j] - (255 * min) / (max - min)

    return img


def dobinaryzation(img):
    max = float(img.max())
    min = float(img.min())

    x = max - ((max - min) / 2)
    ret, threshedimg = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)

    return threshedimg


def find_retangle(contour):
    y, x = [], []

    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])

    return [min(y), min(x), max(y), max(x)]


def locate_license(img, orgimg):
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找出最大的三个区域
    blocks = []
    for c in contours:
        # 找出轮廓的左上点和右下点，由此计算它的面积和长宽比
        r = find_retangle(c)
        a = (r[2] - r[0]) * (r[3] - r[1])
        s = (r[2] - r[0]) / (r[3] - r[1])

        blocks.append([r, a, s])

    # 选出面积最大的3个区域
    blocks = sorted(blocks, key=lambda b: b[2])[-3:]

    # 使用颜色识别判断找出最像车牌的区域
    maxweight  = 0
    maxindex = -1
    for i in range(len(blocks)):
        b = orgimg[blocks[i][0][1]:blocks[i][0][3], blocks[i][0][0]:blocks[i][0][2]]
        # RGB转HSV
        hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        # 蓝色车牌范围
        lower = np.array([100, 50, 50])
        upper = np.array([140, 255, 255])
        # 根据阈值构建掩模
        mask = cv2.inRange(hsv, lower, upper)

        # 统计权值
        w1 = 0
        for m in mask:
            w1 += m / 255

        w2 = 0
        for w in w1:
            w2 += w

        # 选出最大权值的区域
        if w2 > maxweight:
            maxindex = i
            maxweight = w2

    return blocks[maxindex][0]


def find_license(img):
    '''预处理'''
    # 压缩图像
    img = cv2.resize(img, (400, int(400 * img.shape[0] / img.shape[1])))

    # RGB转灰色
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 灰度拉伸
    stretchedimg = stretch(grayimg)

    # 进行开运算，用来去噪声
    r = 16
    h = w = r * 2 + 1
    kernel = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(kernel, (r, r), r, 1, -1)

    openingimg = cv2.morphologyEx(stretchedimg, cv2.MORPH_OPEN, kernel)
    strtimg = cv2.absdiff(stretchedimg, openingimg)

    # 图像二值化
    binaryimg = dobinaryzation(strtimg)

    # 使用Canny函数做边缘检测
    cannyimg = cv2.Canny(binaryimg, binaryimg.shape[0], binaryimg.shape[1])

    ''' 消除小区域，保留大块区域，从而定位车牌'''
    # 进行闭运算
    kernel = np.ones((5, 19), np.uint8)
    closingimg = cv2.morphologyEx(cannyimg, cv2.MORPH_CLOSE, kernel)

    # 进行开运算
    openingimg = cv2.morphologyEx(closingimg, cv2.MORPH_OPEN, kernel)

    # 再次进行开运算
    kernel = np.ones((11, 5), np.uint8)
    openingimg = cv2.morphologyEx(openingimg, cv2.MORPH_OPEN, kernel)

    # 消除小区域，定位车牌位置
    rect = locate_license(openingimg, img)

    return rect, img


if __name__ == '__main__':
    # 读取图片
    orgimg = cv2.imread('./data/test_pr/che2.jpg')
    rect, img = find_license(orgimg)

    # 框出车牌
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
    cv2.imshow('img', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

