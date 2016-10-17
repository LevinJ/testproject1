import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class DetectByColor:
    def __init__(self):
        self.image_path = '../data/laneline.jpg'
        return
    def load_image(self):
        # Read in the image and print out some stats
        image = mpimg.imread(self.image_path)
        print('This image is: ',type(image), 'with dimensions:', image.shape)
        
        # Grab the x and y size and make a copy of the image
        self.ysize = image.shape[0]
        self.xsize = image.shape[1]
        
        return image
    def threshold_by_color(self, image):
        color_select = np.copy(image)
        # Define our color selection criteria
        red_threshold = 200
        green_threshold = 200
        blue_threshold = 200
        rgb_threshold = [red_threshold, green_threshold, blue_threshold]
        
        
        # Use a "bitwise or" to identify pixels below the threshold
        thresholds = (image[:,:,0] < rgb_threshold[0]) \
                    | (image[:,:,1] < rgb_threshold[1]) \
                    | (image[:,:,2] < rgb_threshold[2])
        
        color_select[thresholds] = [0,0,0]
        
        return color_select
    def run(self):
        image = self.load_image()
        color_select = self.threshold_by_color(image)
        # Display the image                 
        plt.imshow(color_select)
        plt.show()

        return






if __name__ == "__main__":   
    obj= DetectByColor()
    obj.run()