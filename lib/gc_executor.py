import numpy as np
import matplotlib.pyplot as plt
import cv2


class GC_executor:
    def __init__(self):
        pass

    def _display_image(self, image):
        # plt.axis('off')
        frame = plt.gca()
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])
        plt.imshow(image)
        plt.show()

    def grab_cut_with_patch(self, patch, heat_map):
        # Grabcut mask
        # DRAW_BG = {'color': BLACK, 'val': 0}
        # DRAW_FG = {'color': WHITE, 'val': 1}
        # DRAW_PR_FG = {'color': GREEN, 'val': 3}
        # DRAW_PR_BG = {'color': RED, 'val': 2}

        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)

        mean = np.mean(heat_map[heat_map != 0])
        heat_map_high_prob = np.where((heat_map > mean), 1, 0).astype('uint8')
        heat_map_low_prob = np.where((heat_map > 0), 3, 0).astype('uint8')
        mask = heat_map_high_prob + heat_map_low_prob
        mask[mask == 4] = 1
        mask[mask == 0] = 2

        mask, bgdModel, fgdModel = cv2.grabCut(patch, mask, None, self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_MASK)

        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = patch * mask[:, :, np.newaxis]
        return img

    def grab_cut_without_patch(self, patch):
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        mask_onlyGC = np.zeros(patch.shape[:2], np.uint8)
        rect = (0, 0, patch.shape[1] - 1, patch.shape[0] - 1)

        cv2.grabCut(patch, mask_onlyGC, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

        mask_onlyGC = np.where((mask_onlyGC == 2) | (mask_onlyGC == 0), 0, 1).astype('uint8')
        img = patch * mask_onlyGC[:, :, np.newaxis]
        return img