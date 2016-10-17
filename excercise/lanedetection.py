import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np


class LaneDetection:
    
    def __init__(self):
        self.image_path = '../test_images/solidWhiteRight.jpg'
        return
    def load_image(self):
        #reading in an image
        image = mpimg.imread(self.image_path)
        #printing out some stats and plotting
        print('This image is:', type(image), 'with dimesions:', image.shape)
        return image
    
    def run(self):
        image = self.load_image()
        plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
        plt.show()
        
        return






if __name__ == "__main__":   
    obj= LaneDetection()
    obj.run()