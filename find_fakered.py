import cv2
import os
import matplotlib.pylab as pl
import numpy as np
from skimage.measure import label, regionprops
from skimage import color, io
import pickle
from roipoly import RoiPoly
import pandas as pd

# create array to store 3 channels labelled area
#fake_red_sign = np.load("fakered.npy")
fake_red_sign = [[], [], []]
file = os.listdir('/Users/wzh13/PycharmProjects/ece2767a1/picture')
for f in file:
    or_image = cv2.imread('./picture/' + f)
    gray_image = cv2.cvtColor(or_image, cv2.COLOR_BGR2GRAY)  # 2  gray space
    rgb_image = cv2.cvtColor(or_image, cv2.COLOR_BGR2RGB)  # 2  rgb space
    ycrcb_image = cv2.cvtColor(or_image, cv2.COLOR_BGR2YCrCb)
    pl.imshow(rgb_image)
    my_roi = RoiPoly(color='r')
    mask = my_roi.get_mask(rgb_image[:, :, 0])
    pl.imshow(mask)
    pl.show()
    sign_position = np.where(mask)
    # print(ycrcb_image.shape)
    y, cr, cb = cv2.split(rgb_image)
    fake_red_sign[0].extend(y[sign_position].tolist())
    fake_red_sign[1].extend(cr[sign_position].tolist())
    fake_red_sign[2].extend(cb[sign_position].tolist())
np.save("fakered.npy", fake_red_sign)