import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.map import HeatMap


class SegmentGenerator:
    def __init__(self, destination_folder, img_file_extension='jpg'):
        self.heatmap_obj = HeatMap()
        self.dest_path = destination_folder
        self.img_file_extension = img_file_extension
        self._validate_paths()

    def _validate_paths(self):
        # Can check whether the number of files in the segmentations is the same as the number of images
        pass

    def _assert_path(self, path, error_message):
        assert os.path.exists(path), error_message

    def _display_image(self, image):
        # plt.axis('off')
        frame = plt.gca()
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])
        plt.imshow(image)
        plt.show()

    def _display_images(self, images):
        plt.figure()
        # plt.figure(figsize=(20, 10))
        columns = 2
        for i, image in enumerate(images):
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
            plt.imshow(image)
        plt.show()

    def segment(self, img_path):

        # Read the image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        xmin = 0
        ymin = 0
        padding = 0
        print 'Input Shape: ', image.shape
        display_images = []
        display_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Get the objectness
        heat_map = self.heatmap_obj.get_map(image)
        heat_map = heat_map.data * ~heat_map.mask

        # self._display_image(heat_map)
        # print heat_map
        objectness_heatmap = cv2.applyColorMap(np.uint8(-heat_map), cv2.COLORMAP_JET)
        display_images.append(objectness_heatmap)
        # self._display_image(objectness_heatmap)


        # Remove the border in the detections
        border = 2
        temp = np.zeros_like(heat_map)
        temp[border:-border, border:-border] = heat_map[border:-border, border:-border]
        heat_map = temp

        # Binary Map
        heat_map[heat_map > 0] = 1
        map_h, map_w = heat_map.shape

        # Flood filling it
        im_floodfill = heat_map.copy()
        h, w = im_floodfill.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        heat_map = heat_map | im_floodfill_inv

        # # Rejecting again if the number of disconnected components are > 3
        # im2, contours, hierarchy = cv2.findContours(heat_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # boundingBoxes = [cv2.boundingRect(c) for c in contours]
        # contour_area = [cv2.contourArea(c) for c in contours]
        # index = np.argmax(contour_area)
        # x, y, w, h = boundingBoxes[index]
        #
        # xmin_tight = int(xmin + x - padding) if int(x - padding) > 0 else xmin
        # ymin_tight = int(ymin + y - padding) if int(y - padding) > 0 else ymin
        # xmax_tight = int(xmin + x + w + padding) if int(x + w + padding) < map_w else xmin + map_w
        # ymax_tight = int(ymin + y + h + padding) if int(y + h + padding) < map_h else ymin + map_h

        # self._display_images(patches)
        mask = heat_map
        mask[mask == 255] = 3
        mask[mask == 0] = 2

        # # changes: START
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask_onlyGC = np.zeros(image.shape[:2], np.uint8)
        rect = (0, 0, image.shape[1]-1, image.shape[0]-1)
        cv2.grabCut(image, mask_onlyGC, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
        mask_onlyGC = np.where((mask_onlyGC == 2) | (mask_onlyGC == 0), 0, 1).astype('uint8')
        img = image * mask_onlyGC[:, :, np.newaxis]
        display_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # self._display_image(img)
        # # changes:END

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask, bgdModel, fgdModel = cv2.grabCut(image, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = image * mask[:, :, np.newaxis]
        display_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # self._display_image(img)

        self._display_images(display_images)
        print 'Output Shape: ', heat_map.shape

if __name__ == '__main__':
    np.set_printoptions(threshold='nan')
    img_db_path = os.path.join('./data/images')
    dest_path = os.path.join('./data/segmentations')

    image_path = '/home/joseph/Dataset/voc_2012/VOCdevkit/VOC2012/JPEGImages/2007_001568.jpg'
    # image_path = '/home/joseph/Dataset/voc_2012/VOCdevkit/VOC2012/JPEGImages/2007_001423.jpg' # Person centerstage
    # image_path = '/home/joseph/Dataset/voc_2012/VOCdevkit/VOC2012/JPEGImages/2007_001416.jpg' # The good goats
    # image_path = '/home/joseph/Dataset/voc_2012/VOCdevkit/VOC2012/JPEGImages/2007_001397.jpg' # The best image
    # image_path = '/home/joseph/Dataset/voc_2012/VOCdevkit/VOC2012/JPEGImages/2007_001311.jpg' # Cyclists
    # image_path = '/home/joseph/Dataset/voc_2012/VOCdevkit/VOC2012/JPEGImages/2007_001289.jpg' # Bird
    # image_path = '/home/joseph/Dataset/voc_2012/VOCdevkit/VOC2012/SegmentationClass/2007_000925.jpg'

    sg = SegmentGenerator(dest_path)
    sg.segment(image_path)
    np.set_printoptions(threshold='nan', linewidth=999)
