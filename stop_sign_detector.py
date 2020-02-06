'''
ECE276A WI20 HW1
Stop Sign Detector
'''

import os
import cv2
import matplotlib.pylab as pl
import numpy as np
from skimage.measure import label, regionprops


class StopSignDetector:
    def __init__(self):
        '''
            Initilize your stop sign detector with the attributes you need,
            e.g., parameters of your classifier
        '''
        #raise NotImplementedError
        self.w = np.array([[19.11037855], [-30.63823315], [-1.56138415], [12.13021749], [-8.6865113 ], [0.88230556], [-5.2437379], [7.48680817]])
        self.b = -1.7751490742440714


    def x_value(self, value):
        return value[0]

    def sigmoid(self, z):
        return (1.0 / (1.0 + np.exp(-z)))

    def segment_image(self, img):
        '''
            Obtain a segmented image using a color classifier,
            e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
            call other functions in this class if needed
            
            Inputs:
                img - original image
            Outputs:img
                mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        '''
        # YOUR CODE HERE
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y_i = rgb_img.shape[0]
        x_i = rgb_img.shape[1]
        d_tmp = rgb_img.reshape((y_i*x_i, rgb_img.shape[2]))
        d_tmp = d_tmp.T / 255. # dim = (3, #)
        d = np.ones((8, d_tmp.shape[1]))
        d[0] = d_tmp[0] # r
        d[1] = d_tmp[1] # g
        d[2] = d_tmp[2] # b
        d[3]= np.multiply(d_tmp[0], d_tmp[0]) # r^2
        d[4] = np.multiply(d_tmp[1], d_tmp[1]) # g^2
        d[5] = np.multiply(d_tmp[2], d_tmp[2])  # b^2
        d[6] = np.multiply(d_tmp[0], d_tmp[1])  # rg
        d[7] = np.multiply(d_tmp[0], d_tmp[2])  # rg
        #print(d.shape, self.w.shape)
        mask_img = self.sigmoid(np.dot(self.w.T, d) + self.b)
        mask_img = mask_img.reshape(y_i, x_i)
        mask_img = np.where(mask_img > 0.5, 255, 0)
        mask_img = mask_img.astype(np.uint8)
        #pl.imshow(mask_img)
        #pl.show()
        #raise NotImplementedError
        return mask_img

    def get_bounding_box(self, img):
        '''
            Find the bounding box of the stop sign
            call other functions in this class if needed
            
            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
                where (x1, y1) and (x2, y2) are the bottom left and top right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.
                
            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
        # YOUR CODE HERE
        mask_img = self.segment_image(img)
        y_axis=img.shape[0]
        # x_axis=image.shape[1]
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(mask_img, kernel, 1)
        mask_img = cv2.erode(dilation, kernel, 1)
        # pl.imshow(mask_img)
        # pl.show()
        contours, hierarchy = cv2.findContours(mask_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        totalArea = mask_img.shape[0]*mask_img.shape[1]
        # com_contour1=np.load("stop_sign.npy")
        # com_contour2=np.load("good_stop_sign.npy")
        # com_contour3=np.load("stop_sign3.npy")
        oct_contour = []
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # if area < totalArea * 0.0042 or totalArea * 0.1 < area:
            if area < 20 or totalArea * 0.1 < area:
                continue # filter out
            else:
                print(f"\narea : {area}, total : {totalArea} , ratio : {area / totalArea}")
                new_img = img.copy()
                epsilon = 0.03 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                print(f" len of approx {len(approx)}")
                if len(approx) <= 4:
                    continue # filter out

                # cv2.drawContours(new_img, contour, -1, (0, 255, 0), 3)
                # cv2.namedWindow("resized", 0)
                # cv2.resizeWindow("resized", 1280, 860)
                # cv2.imshow("resized", new_img)
                # cv2.waitKey(0)
                (x, y), (a, b), angle = cv2.fitEllipse(contour)
                print(f"a/b : {a/b}")
                if a / b < 0.35:
                    continue # filter out

                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # print(cx, cy)
                dist = cv2.pointPolygonTest(contour, (cx, cy), True)
                print(f"dist/a : {dist/a}")
                if (dist < 0) :
                    continue # filter out
                if len(approx) >= 5 and len(approx) <= 100 or a/b > 0.9 or dist/a >= 0.45:
                    oct_contour.append(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    print([x, y_axis - (y+h), x+w, y_axis - y])
                    if (dist/a < 0.2 and a/b < 0.9):
                        continue # filter out
                    if (area < totalArea * 0.0042 and (a/b < 0.95 or a/b > 0.958)):
                        continue # filter out
                    boxes.append([x, y_axis - (y+h), x+w, y_axis - y])
                    new_img = cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.namedWindow("resized", 0)
                    # cv2.resizeWindow("resized", 1280, 860)
                    # cv2.imshow("resized", new_img)
                    # cv2.waitKey(0)

        boxes.sort(key=self.x_value)
        print(boxes)
        return boxes


if __name__ == '__main__':

    '''
    test
    '''





    #
    # folder = "trainset"
    # my_detector = StopSignDetector()
    # for filename in os.listdir(folder):
        # read one img image
        # img = cv2.imread(os.path.join(folder, filename))
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.destroyAllWindows()

    # Display results:
    # (1) Segmented images
    #	 mask_img = my_detector.segment_image(img)
    # (2) Stop sign bounding box
    #    boxes = my_detector.get_bounding_box(img)
    # The autograder checks your answers to the functions segment_image() and get_bounding_box()
    # Make sure your code runs as expected on the imgset before submitting to Gradescope
