import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from lib.map import HeatMap
from lib.plot_annotation import PlotAnnotation

class Enhancer:
    def __init__(self, path_to_images, path_to_annotations, path_to_enhanced_annotations, img_file_extension='jpg'):
        self.heatmap_obj = HeatMap()
        self.img_path = path_to_images
        self.annotation_path = path_to_annotations
        self.dest_annotation_path = path_to_enhanced_annotations
        self.img_file_extension = img_file_extension
        self._validate_paths()

    def _validate_paths(self):
        # Can check whether the number of files in the annotations is the same as the number of images
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
        columns = 5
        for i, image in enumerate(images):
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
            plt.imshow(image)
        plt.show()

    def enhance(self):
        # Read each annotation
        file_count = 0
        for annotation_file in os.listdir(self.annotation_path):
            if os.path.isfile(os.path.join(self.annotation_path, annotation_file)):

                # Read the corresponding image
                file_name, _ = annotation_file.split('.')
                image_path = os.path.join(self.img_path, file_name + '.' + self.img_file_extension)
                self._assert_path(image_path, 'The corresponding image file for annotation not found at: ' + image_path)

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Parse the xml annotation
                annotation_xml = open(os.path.join(self.annotation_path, annotation_file), 'r')
                tree = ET.parse(annotation_xml)
                root = tree.getroot()
                intitial_annotation_count = len(root)
                # For each bb-annotation in annotation:
                patches = []
                heatmaps = []
                padding = 5
                for annotation in root.findall('./object'):
                    xmin = int(annotation.find('./bndbox/xmin').text)
                    ymin = int(annotation.find('./bndbox/ymin').text)
                    xmax = int(annotation.find('./bndbox/xmax').text)
                    ymax = int(annotation.find('./bndbox/ymax').text)

                    # Crop the patch
                    patch = image[ymin:ymax, xmin:xmax]
                    patches.append(patch)

                    # Get the objectness
                    heat_map = self.heatmap_obj.get_map(patch)
                    heat_map = heat_map.data * ~heat_map.mask

                    # Remove the border in the detections
                    border = 2
                    temp = np.zeros_like(heat_map)
                    temp[border:-border, border:-border] = heat_map[border:-border, border:-border]
                    heat_map = temp

                    # Retain only valid Annotations
                    if np.max(heat_map) > 200:
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

                        # Rejecting again if the number of disconnected components are > 3
                        im2, contours, hierarchy = cv2.findContours(heat_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        if len(contours) >= 3:
                            heatmaps.append(np.zeros((2, 2)))
                            root.remove(annotation)
                        else:
                            boundingBoxes = [cv2.boundingRect(c) for c in contours]
                            contour_area = [cv2.contourArea(c) for c in contours]
                            index = np.argmax(contour_area)
                            x, y, w, h = boundingBoxes[index]

                            xmin_tight = int(xmin + x - padding) if int(x - padding) > 0 else xmin
                            ymin_tight = int(ymin + y - padding) if int(y - padding) > 0 else ymin
                            xmax_tight = int(xmin + x + w + padding) if int(x + w + padding) < map_w else xmin + map_w
                            ymax_tight = int(ymin + y + h + padding) if int(y + h + padding) < map_h else ymin + map_h

                            annotation.find('./bndbox/xmin').text = str(xmin_tight)
                            annotation.find('./bndbox/ymin').text = str(ymin_tight)
                            annotation.find('./bndbox/xmax').text = str(xmax_tight)
                            annotation.find('./bndbox/ymax').text = str(ymax_tight)

                            heatmaps.append(heat_map)
                    else:
                        heatmaps.append(np.zeros((2,2)))
                        root.remove(annotation)

                # Write back the annotation
                tree.write(os.path.join(self.dest_annotation_path, annotation_file))
                print intitial_annotation_count-len(root), ' annotations removed.'

                # Plot annotation
                p = PlotAnnotation(self.img_path, self.dest_annotation_path, file_name)
                p.plot_annotation()
                p.save_annotated_image('./data/annotated_images/enhanced_' + file_name + '.png')

                file_count += 1
                print 'Done with: ', file_count

                # self._display_images(patches)
                # self._display_images(heatmaps)

if __name__ == '__main__':
    np.set_printoptions(threshold='nan')
    img_db_path = os.path.join('./data/images')
    annotation_path = os.path.join('./data/annotations')
    dest_annotation_path = os.path.join('./data/enhanced_annotations')

    e = Enhancer(img_db_path, annotation_path, dest_annotation_path)
    e.enhance()
    np.set_printoptions(threshold='nan')
