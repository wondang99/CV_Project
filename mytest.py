import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from predict import *
from collections import defaultdict
from tqdm import tqdm
import argparse
from PIL import Image
import time

parser = argparse.ArgumentParser()
parser.add_argument("-img")

args = parser.parse_args()

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]


print('---start test---')
model = resnet50()
print(torch.cuda.is_available())
model.load_state_dict(torch.load('res50_e50_b8.pth'))
model.eval()
model.cuda()
count = 0

# img = Image.open(args.img)
# img_resize = img.resize((int(512), int(512)))
# img_resize.save(args.img)


result = predict_gpu(model,args.img) #result[[left_up,right_bottom,class_name,image_path],]

image = cv2.imread(args.img)
for left_up,right_bottom,class_name,_,prob in result:
    color = Color[VOC_CLASSES.index(class_name)]
    cv2.rectangle(image,left_up,right_bottom,color,2)
    label = class_name+str(round(prob,2))
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    p1 = (left_up[0], left_up[1]- text_size[1])
    cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
    cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)


cv2.imwrite('result.jpg',image)
