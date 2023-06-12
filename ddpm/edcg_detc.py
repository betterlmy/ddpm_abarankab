import cv2
import numpy as np
import matplotlib.pyplot as plt

img_list = ["F:\Code\edgeDet\image_roi_128\LIDC_0000.png", "F:\Code\edgeDet\image_roi_128\LIDC_0062.png",
            "F:\Code\edgeDet\image_roi_128\LIDC_0096.png", "F:\Code\edgeDet\image_roi_128\LIDC_0408.png",
            "F:\Code\edgeDet\image_roi_128\LIDC_0619.png"]

plt.rcParams["figure.figsize"] = (16.0, 8.0)


# canny边缘检测
def canny(img, size, threshold1, threshold2):
    img = cv2.GaussianBlur(img, (size, size), 0)
    canny_img = cv2.Canny(img, threshold1, threshold2)
    return canny_img


# sobel
def sobely(img, ksize):
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    return sobely


# sobel
def sobelx(img, ksize):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    return sobelx


# Laplacian边缘检测
def Laplacian(img):
    img_temp = cv2.Laplacian(img, cv2.CV_16S)
    img_Laplacian = cv2.convertScaleAbs(img_temp)
    return img_Laplacian


l_title = ['Original', 'sobelx_size3', 'sobely_size3', 'sobelx_size5', 'sobely_size5', 'canny_img_size3',
           'canny_img_size5', 'Laplacian_img']
count = 0
for i in range(5):
    original_img = cv2.imread(img_list[i], 0)
    sobelx_size3, sobely_size3 = sobel(original_img, 3)
    sobelx_size5, sobely_size5 = sobel(original_img, 5)
    canny_img_size3 = canny(original_img, 3, 50, 150)
    canny_img_size5 = canny(original_img, 5, 50, 150)
    Laplacian_img = Laplacian(original_img)
    l_img = [original_img, sobelx_size3, sobely_size3, sobelx_size5, sobely_size5, canny_img_size3, canny_img_size5,
             Laplacian_img]
    for j in range(8):
        if i == 0:
            count = count + 1
            plt.subplot(5, 8, count), plt.imshow(l_img[j], cmap='gray')
            plt.title(l_title[j]), plt.xticks([]), plt.yticks([])
        else:
            count = count + 1
            plt.subplot(5, 8, count), plt.imshow(l_img[j], cmap='gray')
            plt.xticks([]), plt.yticks([])
# plt.show()  

plt.savefig("x.svg")

# original_img = cv2.imread(img_list[0], 0)
# sobelx_size3,sobely_size3 = sobel(original_img,3)
# sobelx_size5,sobely_size5 = sobel(original_img,5)
# canny_img_size3 = canny(original_img,3,50,150)
# canny_img_size5 = canny(original_img,5,50,150)
# Laplacian_img = Laplacian(original_img)
# l_img = [original_img,sobelx_size3,sobely_size3,sobelx_size5,sobely_size5,canny_img_size3,canny_img_size5,Laplacian_img]
# for i in range(8):
#     count = count+1
#     plt.subplot(2,4,count),plt.imshow(l_img[i],cmap = 'gray')
#     plt.title(l_title[i]), plt.xticks([]), plt.yticks([])
# plt.show()   

cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.title('Canny1'), plt.xticks([]), plt.yticks([])

# # 画图
# plt.subplot(1,3,1),plt.imshow(original_img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])

# plt.subplot(1,3,2),plt.imshow(canny1,cmap = 'gray')
# plt.title('Canny1'), plt.xticks([]), plt.yticks([])


# plt.subplot(1,3,3),plt.imshow(canny2,cmap = 'gray')
# plt.title('Canny2'), plt.xticks([]), plt.yticks([])
# plt.show()

# cv2.imshow("Original", original_img)
# cv2.imshow("Canny1", canny1)
# cv2.imshow("Canny2", canny2)
