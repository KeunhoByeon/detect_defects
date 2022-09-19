import os
import cv2
import numpy as np

data_dir = os.path.expanduser('~/data/KolektorSDD2/train')

p, n = 0, 0

for path, dir, files in os.walk(data_dir):
    for filename in files:
        fn, ext = os.path.splitext(filename)
        if ext not in ('.png', '.jpg', '.jpeg'):
            continue
        if 'GT' in fn:
            continue

        img_path = os.path.join(path, filename)
        gt_path = os.path.join(path, '{}_GT{}'.format(fn, ext))

        if not os.path.isfile(gt_path):
            continue

        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)

        # img = cv2.resize(img, (512, 512))
        # gt = cv2.resize(gt, (512, 512))

        # cv2.imshow('T', np.hstack([img, gt]))
        # cv2.waitKey(0)

        if np.max(gt) > 0:
            p += 1
        else:
            n += 1

print(p, n)
