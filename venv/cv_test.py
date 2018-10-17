import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import glob2
import os
import shutil
import pprint

def main():
    #input_image = 'S40-RX-75-B375-C72-4_TotalPhaseF1.png'
    #input_image = 'S40-RX75-B375-C52-3_TotalPhaseF1.png'
    results = {}
    lens_count_search = re.compile(r'-(\d)_')
    for file in glob2.glob('**\*TotalPhaseF1.png'):
        try:
            crib, area_ratio = get_lens_edge(file)
            lens_count = re.search(lens_count_search,os.path.basename(file))
            if 0.5 > area_ratio:
                test_result = 'PASS'
            if area_ratio < 0.5:
                test_result = 'FAIL'
            else:
                test_result = 'ERROR'
            results.update({str(crib)+ '-' + lens_count.group(1): [area_ratio, test_result]})
        #cv2.imshow(file, crop)
        except TypeError:
            print('File ' + file + ' did not have a found circle.')
            pass
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)
    #processed_image = prep_circle_find(input_image,2)
    #cropped, crop_circle = get_lens_edge(input_image)
    #plt.imshow(cropped)
    #plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def prep_circle_find(unprocessed_image,strength):
    grayscale_image = unprocessed_image#cv2.cvtColor(unprocessed_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.blur(grayscale_image, (strength + 2, strength + 2))
    ret, thresholded_image = cv2.threshold(blurred_image, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((strength+2,strength+2),np.uint8)
    eroded_image = cv2.erode(thresholded_image,kernel,iterations=strength-1)
    dilated_image = cv2.dilate(eroded_image,kernel,iterations=strength+1)
    return dilated_image
#
def get_lens_edge(filename):
    pixels_per_mm = 20
    crib_search = re.search(re.compile(r'C(\d\d)'),filename)
    crib = int(crib_search.group(1))
    print(crib)
    radius = crib*pixels_per_mm/2
    unprocessed_image = cv2.imread(filename,0)
    preprocessed_image = prep_circle_find(unprocessed_image,2)
    height,width = unprocessed_image.shape
    mask = np.zeros((height,width), np.uint8)
    edges = cv2.Canny(preprocessed_image, 150, 200)
    try:
        found_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1000, param1 = 50, param2 = 30, minRadius=int(.98*radius), maxRadius = int(1.05*radius))
        for each_circle in found_circles[0,:]:
            cv2.circle(mask,(each_circle[0],each_circle[1]),each_circle[2],(255,255,255),thickness=-1)
        masked_data = cv2.bitwise_and(unprocessed_image, unprocessed_image, mask=mask)
        contours = cv2.findContours(masked_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        #apply masking circle to find edge defect
        cv2.circle(masked_data, (each_circle[0], each_circle[1]), int(radius * .92), (0, 0, 0), thickness=-1)
        crop = masked_data[y:y + h, x:x + w]
        cv2.imwrite('cropped_results\\' + re.sub(r'.png', '', os.path.basename(filename)) + '_analyzed.png', crop)
        white_pixel_count = np.sum(crop > 150)  # number of pixels, i.e. area, that are white
        pi = np.pi
        outer_radius = int(radius)  # average of the height and width of the image divided by 2 to get radius
        inner_radius = int(.95 * radius)
        #print(outer_radius)
        #print(inner_radius)
        area = pi / 2 * (outer_radius ** 2 - inner_radius ** 2)
        #print(area)
        print(white_pixel_count)
        area_ratio = white_pixel_count/area
        #print(area_ratio)
        return crib, area_ratio
    except:
        cv2.imwrite('cropped_results\\' + re.sub(r'.png', '', os.path.basename(filename)) + '_failed.png', unprocessed_image)
        pass




if __name__ == "__main__":
    main()
