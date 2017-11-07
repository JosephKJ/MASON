import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

import _init_path
import caffe

caffe_root = os.path.join(os.path.dirname(__file__), '..', '..', 'caffe')


class HeatMap:
    def __init__(self, arch='VGG16'):
        caffe.set_device(0)
        caffe.set_mode_gpu()
        # caffe.set_mode_cpu()

        # Loading the caffe model
        if arch == 'CaffeNet':
            model_def = os.path.join(os.path.dirname(__file__), '../model/caffenet_deploy_conv_only.prototxt')
            model_weights = os.path.join(os.path.dirname(__file__), '../model/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
        elif arch == 'VGG16':
            model_def = os.path.join(os.path.dirname(__file__), '../model/VGG_ILSVRC_16_layers_conv_only.prototxt')
            model_weights = os.path.join(os.path.dirname(__file__), '../model/VGG_ILSVRC_16_layers.caffemodel')
        elif arch == 'ResNet-50':
            model_def = os.path.join(os.path.dirname(__file__), '../model/ResNet-50-deploy.prototxt')
            model_weights = os.path.join(os.path.dirname(__file__), '../model/ResNet-50-model.caffemodel')
        elif arch == 'ResNet-101':
            model_def = os.path.join(os.path.dirname(__file__), '../model/ResNet-101-deploy.prototxt')
            model_weights = os.path.join(os.path.dirname(__file__), '../model/ResNet-101-model.caffemodel')

        self.net = caffe.Net(model_def, model_weights, caffe.TEST)
        self.arch = arch

    def get_map(self, image, verbose=False):

        # image = scipy.misc.imresize(image, 8.0, interp='bicubic')
        image_shape = image.shape

        # Image Pre-processing
        transformer = caffe.io.Transformer({'data': (10, 3, image_shape[0], image_shape[1])})
        transformer.set_transpose('data', (2, 0, 1)) # Channel, Height, Width
        transformer.set_channel_swap('data', (2, 1, 0)) # The model expexts BGR
        transformed_image = transformer.preprocess('data', image)

        # Performing Forward Pass
        net = self.net
        net.blobs['data'].reshape(1, 3, image_shape[0], image_shape[1])
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()

        # Extracting the feature map
        if self.arch == 'CaffeNet':
            feat = net.blobs['conv5'].data[0]
        elif self.arch == 'VGG16':
            feat = net.blobs['conv5_3'].data[0]
        elif self.arch == 'ResNet-50':
            feat = net.blobs['res3a'].data[0]
        elif self.arch == 'ResNet-101':
            feat = net.blobs['res3a'].data[0]

        # Generating the mask
        feature_sum = np.sum(feat, axis=0)
        np.set_printoptions(threshold='nan')
        feature_sum = (255 * (feature_sum - np.min(feature_sum)) / np.ptp(feature_sum)).astype(int)
        # feature_sum = (feature_sum - np.min(feature_sum)) / np.ptp(feature_sum)
        threshold = feature_sum.mean() + 20
        # threshold = 0
        feature_sum = np.ma.masked_where(feature_sum <= threshold, feature_sum)

        # Scaling the map to the input image size.
        feature_sum = scipy.misc.imresize(feature_sum, image_shape, interp='bicubic')
        feature_sum = np.ma.masked_where(feature_sum <= threshold, feature_sum)

        if verbose:
            print feature_sum

        return feature_sum

    def display_image(self, image):
        # plt.axis('off')
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    print('Inside Main.')

    hm = HeatMap()

    image_path = os.path.join('/home/cs17mtech01001/workspace/SDD-RFCN-python/data/detections/bookstore_video0_9500_hr_bc_pedestrian_2.png')
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hMap = hm.get_map(img, verbose=True)
    hm.display_image(hMap)

    # for i in range(1, 4):
    #     hMap = hm.get_map(img, verbose=True, layer_name='res3b'+str(i))
    #     hm.display_image(hMap)

    # for index in range(0, 23):
    #     image_path = os.path.join(
    #         '/home/cs17mtech01001/workspace/SDD-RFCN-python/data/detections/bookstore_video0_9500_hr_bc_pedestrian_'+str(index)+'.png')
    #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     hMap = hm.get_map(img, verbose=False)
    #     print 'Shape of the map', hMap.shape
    #     # hm.display_image(hMap)