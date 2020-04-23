# -*- coding:utf8 -*-
import os

path = 'resultImg/'
filelist = os.listdir(path)

counts = 1
for item in filelist:
    # print('item name is ',item)
    if item.endswith('.jpg'):
        name = 'riceImg_' + str(counts)
        src = os.path.join(os.path.abspath(path), item)     # 路径拼接，其中abspath函数可以得到其绝对路径
        dst = os.path.join(os.path.abspath(path), name + '.jpg')    # 修改后的文件路径及名称
        counts += 1
    try:
        os.rename(src, dst)     # 将路径及名称为src的文件名改为dst
        print('rename from %s to %s' % (src, dst))
    except:
        continue
