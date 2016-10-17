import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from detectbycolor import DetectByColor

class DetectByColorRegion(DetectByColor):
    def __init__(self):
        self.image_path = '../data/laneline.jpg'
        return
   
    def threshold_by_region(self, image):
        region_select = np.copy(image)
        left_bottom = [60,539]
        right_bottom = [900,539]
        apex = [500,300]
        
        # Fit lines to identify the region of interest
        fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
        fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
        fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
        
        # Find the region inside the lines
        XX, YY = np.meshgrid(np.arange(0,self.xsize), np.arange(0,self.ysize))
        region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
        # Find where image is both colored right and in the region
        region_select[~region_thresholds] = [0,0,0]
        
        return region_select
    def draw_meshgrid(self, image):
        x_size = image.shape[1]
        y_size = image.shape[0]
        top_xs = np.arange(0, x_size)
        top_ys = np.zeros(x_size).astype(np.uint8)
        bottom_xs = np.arange(0, x_size)
        bottom_ys = np.zeros(x_size).astype(np.uint8)
        bottom_ys.fill(y_size-1)
        
        for i in range(x_size):
            cv2.line(image, (top_xs[i], top_ys[i]), (bottom_xs[i], bottom_ys[i]), (255,0,0))
        return image
    def run(self):
        image = self.load_image()
        color_select = self.threshold_by_color(image)
        region_slect = self.threshold_by_region(color_select)
        region_slect = self.draw_meshgrid(region_slect)
        # Display the image                 
        plt.imshow(region_slect)
        plt.show()

        return






if __name__ == "__main__":   
    obj= DetectByColorRegion()
    obj.run()