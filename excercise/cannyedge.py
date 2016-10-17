import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

class CannyEdge:
    def __init__(self):
        self.image_path = '../data/exit_ramp.png'
        return
    def load_image(self):
        # Read in the image and print out some stats
        image = mpimg.imread(self.image_path)
        print('This image is: ',type(image), 'with dimensions:', image.shape)
        
        # Grab the x and y size and make a copy of the image
        self.ysize = image.shape[0]
        self.xsize = image.shape[1]
        
        return image
    def gray_scale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #grayscale conversion
       
        return gray
    
    def canny(self, gray):
        #Gaussian blurring
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
        
        blur_gray = (blur_gray * 255).astype(np.uint8)
        
        low_threshold = 50
        high_threshold = 150
        
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        return edges
    def run(self):
        image = self.load_image()
        gray = self.gray_scale(image)
        edges = self.canny(gray)
        
        plt.imshow(edges, cmap='gray')
        plt.show()

        return






if __name__ == "__main__":   
    obj= CannyEdge()
    obj.run()