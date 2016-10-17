import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from cannyedge import CannyEdge

class Houghline(CannyEdge):
    
    def __init__(self):
        CannyEdge.__init__(self)
        self.image_path = '../data/exit_ramp.png'
        return
    def crop_roi(self, edges):
        #next we'll create a masked edges image using cv2.fillPoly()
        mask = np.zeros_like(edges)   
        ignore_mask_color = 255   
        
        #this time we are defining a four sided polygon to mask
#         vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
        vertices = np.array([[(10,539),(460,290), (480,290), (930,539)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)
        return masked_edges
    def get_hough_lines(self, edges):
        rho = 1
        theta = np.pi/180
        threshold = 3
        min_line_length = 10
        max_line_gap = 5
        
        color_edges = np.dstack((edges, edges, edges)) 
        line_image = np.copy(color_edges)*0 #creating a blank to draw lines on
        
       
        masked_edges = self.crop_roi(edges)
        #run Hough on edge detected image

        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
        #iterate over the output "lines" and draw lines on the blank
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
                
        #creating a "color" binary image to combine with line image
        
        #drawing the lines on the edge image
        combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
        
        return combo
#     def threshold_by_region(self, image):
#         region_select = np.copy(image)
#         left_bottom = [10,539]
#         right_bottom = [930,539]
#         apex = [470,300]
#         
#         # Fit lines to identify the region of interest
#         fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
#         fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
#         fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
#         
#         # Find the region inside the lines
#         XX, YY = np.meshgrid(np.arange(0,self.xsize), np.arange(0,self.ysize))
#         region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
#                     (YY > (XX*fit_right[0] + fit_right[1])) & \
#                     (YY < (XX*fit_bottom[0] + fit_bottom[1]))
#         # Find where image is both colored right and in the region
#         region_select[~region_thresholds] = 0
#         
#         return region_select
    def run(self):
        image = self.load_image()
        gray = self.gray_scale(image)
        edges = self.canny(gray)
        combo = self.get_hough_lines(edges)
        plt.imshow(combo, cmap='gray')
        plt.show()

        return






if __name__ == "__main__":   
    obj= Houghline()
    obj.run()