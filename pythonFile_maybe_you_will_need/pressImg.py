from PIL import Image
import cv2
import os

root = "E:/201902202/Second_semester_of_freshman_year/Web/WebTest/WebRealTest/Test_4/img"
ret = ['.jpg', '.png', '.jepg']
name = '/2013007'
img = Image.open(root + name + ret[0])   # 截图
width = img.size[0]   # 获取宽度
height = img.size[1]   # 获取高度
while(width > 800):
    img = img.resize((int(width * 0.4), int(height * 0.4)), Image.ANTIALIAS)
    width = img.size[0]  # 获取宽度
    height = img.size[1]  # 获取高度
img.save(root + name + ret[0])