'''
文档说明

种子识别：整体图像分割算法（单张图片）
文档内容：img文件夹存放待处理的种子图片，pythonFile是一些淘汰了的测试代码，Count.py是项目的核心代码，
        processImg文件夹中是过程中产生的照片用于后期浏览（其中生成顺序已经标号，便于查询），resultImg
        文件夹中存储的是对图片处理后的子种子图片，是我们想要得到的结果。
使用流程：修改root_initial和root_result中的路径可以修改路径，其他图片存储路径为相对路径一般不需要做
        修改。其中processImg是否存储和修改可以通过对每组代码的其中四行代码修改实现。
算法介绍：1.遍历整张图片对整张图片的背景进行处理，将背景处理为黑色，处理后效果为：1_initialImage.jpg
        2.二值化处理，直观变化为图片变为黑白图像，处理后效果为：2_threshImage.jpg
        3.双边滤波+填充“美化”图片,初步去噪，处理后效果为：3.1_blurImage.jpg & 3.2_filloutImage.jpg
        4.开运算：腐蚀图像去除噪点，再进行膨胀处理恢复图像，此时噪点大多被消除（但此方法容易造成图片畸形）；
          于此对应，先膨胀再腐蚀可以有效地使图片更加圆滑，但会保留大量噪点，效果：4_erodeImage.jpg &
          5_dilateImage
        5.边缘噪声填充处理，处理后效果为：6_extendedImage.png  这一步还有最重要的一点就是产生ndarray
          数据集来进行后续选区裁切处理
        6.使用cv2库的画像裁切函数，获得x,y,w,h四个坐标确认选区，再对这部分选区裁切（裁切时通过变换扩大一
        下选区），得到结果
使用前需调整：27，28，30，42，162
'''
import cv2
import numpy as np
from PIL import Image

'''读写相关：从root根目录中读取文件，并将图片设置为1200*1200像素大小'''
root_initial = "E:/201902202/Second_semester_of_freshman_year/test/SeedSearch/img/riceGan"
root_result = "E:/201902202/Second_semester_of_freshman_year/test/SeedSearch/resultImg/riceImgGan_"
ret = ['.jpg', '.png', '.jepg']
img = cv2.imread(root_initial + '/gen_sample_1' + ret[0], cv2.IMREAD_COLOR)
# img = cv2.resize(img, (1800, 1800))


'''处理背景，将背景区域处理为黑色(暴力修改方法，留待改进)'''
def delect(image):

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                #img[i][j][k] = 255

                if int(image[i][j][0]) + int(image[i][j][1]) + int(image[i][j][2]) < 180*3:
                    image[i][j] = 0

    return image

img1 = delect(img)
cv2.imwrite('processImg/1_initialImage.jpg', img1)  #保存图像
# cv2.namedWindow("initialImage_1",0);    #设置窗口名称大小
# cv2.resizeWindow("initialImage_1", 640, 480);
# cv2.imshow('initialImage_1', img1)    #显示图片
# cv2.waitKey(0)      #保持窗口不自动关闭


'''缩放'''
rows, cols, channels = img1.shape   #图片垂直尺寸,图片水平尺寸，图片通道数
print(rows, cols, channels)     #1200,1200,3


'''二值化'''
ret, thresh1 = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY) #核心代码：二值化处理
# cv2.namedWindow("thresh_2",0);
# cv2.resizeWindow("thresh_2", 640, 480);
# cv2.imshow('thresh_2', thresh1)
cv2.imwrite('processImg/2_threshImage.jpg', thresh1)
# cv2.waitKey(0)


'''双边滤波+填充'''
blur = cv2.bilateralFilter(thresh1, 9, 75, 75)
fill = blur.copy()
h, w = blur.shape[: 2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(fill, mask, (90, 90), (255,255,255));
fill_INV = cv2.bitwise_not(fill)
fill_out = blur | fill_INV
# cv2.namedWindow("filloutImage_3",0);
# cv2.resizeWindow("filloutImage_3", 640, 480);
# cv2.imshow('filloutImage_3', fill_out)
cv2.imwrite('processImg/3.1_blurImage.jpg', blur)
cv2.imwrite('processImg/3.2_filloutImage.jpg', fill_out)
# cv2.waitKey(0)


'''开运算：先腐蚀再膨胀，噪声初步去除'''
erode=cv2.erode(fill_out,None,iterations=5)
#核心代码，实现图片腐蚀，通过修改iterations（迭代）可以适应不同像素大小的图片处理，
#设置为3在1000^2像素效果较好且实践证明当迭代次数增加时，尤其是设置为7以上，几乎不再有效果上的提升
# cv2.namedWindow("erodeImage_4.1",0);
# cv2.resizeWindow("erodeImage_4.1", 640, 480);
# cv2.imshow('erodeImage_4.1', erode)
cv2.imwrite('processImg/4.1_erodeImage.jpg', erode)
# cv2.waitKey(0)

dilate=cv2.dilate(erode, None, iterations=7)
#核心代码，实现图片膨胀
# cv2.namedWindow("dilateImage_5",0);
# cv2.resizeWindow("dilateImage_5", 640, 480);
# cv2.imshow('dilateImage_5', dilate)
cv2.imwrite('processImg/5_dilateImage.jpg', dilate)
# cv2.waitKey(0)

erode=cv2.erode(dilate,None,iterations=2)
#核心代码，实现图片腐蚀
# cv2.namedWindow("erodeImage_4.2",0);
# cv2.resizeWindow("erodeImage_4.2", 640, 480);
# cv2.imshow('erodeImage_4.2', erode)
cv2.imwrite('processImg/4.2_erodeImage.jpg', erode)
# cv2.waitKey(0)


'''去掉边缘颗粒，去除边缘噪声'''
extended = cv2.copyMakeBorder(erode, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255])
#扩充src的边缘，将图像变大。其中 BORDER_CONSTANT：常量法 以value填充（白色）
'''然后从白边上的任意点用黑色填充'''
mh, mw = extended.shape[:2]
mask = np.zeros([mh + 2, mw + 2], np.uint8)
cv2.floodFill(extended, mask, (0, 0), (0, 0, 0),flags=cv2.FLOODFILL_FIXED_RANGE)
#漫水填充算法
# cv2.namedWindow("pImg_6",0);
# cv2.resizeWindow("pImg_6", 640, 480);
# cv2.imshow('pImg_6', extended)
extended = cv2.cvtColor(extended,cv2.COLOR_BGR2GRAY)    #生成灰度图像
print(extended.shape)
cv2.imwrite('processImg/6_extendedImage.png', extended)
# cv2.waitKey(0)


'''遍历替换去掉背景'''
for i in range(rows):
    for j in range(cols):
        if any(erode[i,j]==0):
            img1[i,j]=(0,0,0)   #此处替换颜色，为BGR通道
# cv2.namedWindow('exchangeImg_7', 0)
# cv2.resizeWindow('exchangeImg_7', 640, 480)
# cv2.imshow('exchangeImg_7', img1)
cv2.imwrite('processImg/7_exchangeImage.jpg', img1)
# cv2.waitKey(0)



'''画框切割,寻找每粒种子'''
contours, hierarchy = cv2.findContours(extended, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 检测轮廓
#parameter_1：寻找轮廓的图像   parameter_2：轮廓的检索模式  parameter_3:轮廓的近似办法
#cv2.RETR_EXTERNAL表示只检测外轮廓
#cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息

for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 5)

# cv2.namedWindow('childBox_8', 0)
# cv2.resizeWindow('childBox_8', 640, 480)
# cv2.imshow('childBox_8', img1)
cv2.imwrite('processImg/8_childBox.jpg', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img2 = Image.open("processImg/7_exchangeImage.jpg")
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    out = img2.crop((x-0.1*w, y-0.1*h, x+1.05*w, y+1.05*h))   #根据坐标切割子元素为out
    out.save(root_result + str(i+1)+'.jpg','JPEG')

cv2.destroyAllWindows()