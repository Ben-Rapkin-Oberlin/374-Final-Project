import cv2
import glob
import re

img_array = []
numbers=re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

count=0
a=sorted(glob.glob('images/protest/*.jpg'), key=numericalSort)
size=len(a)
for filename in a:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    count+=1
    print(count,"/",size)
    