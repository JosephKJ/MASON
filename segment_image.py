import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.map import HeatMap
from lib.gc_executor import GC_executor
import scipy
from mpl_toolkits.mplot3d import Axes3D

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
        columns = 4
        for i, image in enumerate(images):
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
            plt.imshow(image)
        plt.show()

    def _save_images(self, images, name):
        plt.figure(figsize=(20, 10))
        columns = 5
        for i, image in enumerate(images):
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
            plt.imshow(image)
        plt.savefig(name, bbox_inches='tight')

    def _draw_contour(self, im):
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[4]
        img = cv2.drawContours(im, [cnt], -1, (0, 255, 0), 3)
        self._display_image(img)

    def _get_surface_plot(self, heat_map):
        # downscaling has a "smoothing" effect
        heat_map = scipy.misc.imresize(heat_map, 0.50, interp='cubic')

        # create the x and y coordinate arrays (here we just use pixel indices)
        xx, yy = np.mgrid[0:heat_map.shape[0], 0:heat_map.shape[1]]

        # create the figure
        fig = plt.figure()
        ax = Axes3D(fig)
        # ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, heat_map, rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0)
        plt.show()

    def segment(self, img_path):

        # Read the image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        display_images = []
        display_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Get the objectness
        heat_map = self.heatmap_obj.get_map(image)
        heat_map = heat_map.data * ~heat_map.mask
        objectness_heatmap = cv2.applyColorMap(np.uint8(heat_map), cv2.COLORMAP_JET)
        display_images.append(cv2.cvtColor(objectness_heatmap, cv2.COLOR_BGR2RGB))

        gc = GC_executor()

        # Doing grabcut using heat_map
        img = gc.grab_cut_with_patch(np.copy(image), np.copy(heat_map))
        # Doing grabcut without using heat_map
        img_gc_only = gc.grab_cut_without_patch(np.copy(image))
        display_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        display_images.append(cv2.cvtColor(img_gc_only, cv2.COLOR_BGR2RGB))

        print 'Displaying image...'
        self._display_images(display_images)

if __name__ == '__main__':
    img_db_path = os.path.join('./data/images')
    dest_path = os.path.join('./data/segmentations')

    image_path = './demo/2007_000363.jpg'

    sg = SegmentGenerator(dest_path)
    sg.segment(image_path)

    print 'Done.'
