import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import glob2
import os
import shutil
import pprint

def test():
    test_image = cv2.imread('S40-RX-75-B375-C72-4_TotalPhaseF1.png')
    processed_test_image = defect_amplify(test_image)
    plt.imshow(processed_test_image)
    plt.show()

def defect_amplify(image):
    blur_crop = cv2.blur(image, (4, 4))
    erode_crop = cv2.erode(blur_crop, (3, 3), iterations=1)
    ret_crop, thresh_crop = cv2.threshold(erode_crop, 190, 255, cv2.THRESH_BINARY)
    return thresh_crop

def main():
    # input_image = 'S40-RX-75-B375-C72-4_TotalPhaseF1.png'
    # input_image = 'S40-RX75-B375-C52-3_TotalPhaseF1.png'
    results = {}
    results_compare = {}
    lens_count_search = re.compile(r'-(\d)_')
    lens_rx_search = re.compile(r'(RX[-]?[0]?\d\d)')
    #print('-----------------------')
    for index in range(1):
        for file in glob2.glob('**\*_TotalPhaseF1.png'): #*_TotalPhaseF1.png'):  # S40-RX75-B375-C75-
            try:
                crib, area_ratio, radii, thresh_crop, crop, preprocessed_image = get_lens_edge(file)
                lens_count = re.search(lens_count_search, os.path.basename(file))
                lens_rx = re.search(lens_rx_search, os.path.basename(file))
            except TypeError:
                # test_result = 'ERROR'
                # area_ratio = -1
                # results.update({str(crib) + '-' + lens_count.group(1): [area_ratio, test_result]})
                print('File ' + file + ' did not have a found circle.')
                pass
            if (area_ratio > 0.650):
                test_result = 'PASS'
            if (area_ratio < 0.650):
                test_result = 'FAIL'
            print('Test result: ' + test_result)
            plt.imshow(thresh_crop)
            plt.show()

            results.update({str(crib) + '-' + lens_rx.group(1) + '-' + lens_count.group(1): [area_ratio, test_result]})
            cv2.imwrite('cropped_results\\' + re.sub(r'.png', '', os.path.basename(file)) + test_result +  '_analyzed.png',
                        thresh_crop)
            cv2.imwrite('cropped_results\\' + re.sub(r'.png', '', os.path.basename(file)) + test_result + '_unanalyzed.png',
                        crop)

        #results_compare.update({'Test' + str(index): results})
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)
# processed_image = prep_circle_find(input_image,2)
# cropped, crop_circle = get_lens_edge(input_image)
# plt.imshow(cropped)
# plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def prep_circle_find(unprocessed_image, strength):
    grayscale_image = unprocessed_image  # cv2.cvtColor(unprocessed_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.blur(grayscale_image, (strength + 4, strength + 4))
    ret, thresholded_image = cv2.threshold(blurred_image, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((strength + 2, strength + 2), np.uint8)
    eroded_image = cv2.erode(thresholded_image, kernel, iterations=strength - 1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=strength + 4)
    return dilated_image


#
def get_lens_edge(filename):
    #lens details preparation
    pixels_per_mm = 20.6
    crib_search = re.search(re.compile(r'C(\d\d)'), filename)
    crib = int(crib_search.group(1))
    print('--------------------')
    print('Lens crib diameter: ' + str(crib))
    radius = crib * pixels_per_mm / 2

    #image preprocessing
    unprocessed_image = cv2.imread(filename, 0)
    height, width = unprocessed_image.shape
    mask = np.zeros((height, width), np.uint8)
    preprocessed_image = prep_circle_find(unprocessed_image, 2)
    edges = cv2.Canny(preprocessed_image, 150, 200)

    #find circles and process image
    try:
        radii = []
        radius_reduction = .95
        found_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 500, param1=50, param2=30,
                                         minRadius=int(.98 * radius), maxRadius=int(1.05 * radius))
        for each_circle in found_circles[0, :]:
            radii.append((each_circle[2]-radius)/radius)

            print('predicted circle radius = ' + str(radius))
            print('found circle radius = ' + str(each_circle[2]))
            outer_radius = int(each_circle[2]*.97)#int((radius+each_circle[2])/2*radius_reduction)
            inner_radius = int(outer_radius*(.93))
            cv2.circle(mask, (each_circle[0], each_circle[1]), outer_radius, (255, 255, 255), thickness=-1)
            cv2.circle(mask, (each_circle[0], each_circle[1]), inner_radius, (0, 0, 0), thickness=-1)
            #cv2.circle(preprocessed_image, (each_circle[0], each_circle[1]), each_circle[2], (255, 125, 0), thickness=3)

        masked_data = cv2.bitwise_and(unprocessed_image, unprocessed_image, mask=mask)
        masked_preprocessed_data = cv2.bitwise_and(preprocessed_image, preprocessed_image, mask=mask)



        #masked_data = cv2.bitwise_and(unprocessed_image, unprocessed_image, mask=mask)
        contours = cv2.findContours(masked_preprocessed_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        crop = masked_data[y:y + h, x:x + w]
        blur_crop = cv2.blur(crop, (4, 4))
        erode_crop = cv2.erode(blur_crop, (3, 3), iterations=1)
        ret_crop, thresh_crop = cv2.threshold(erode_crop, 190, 255, cv2.THRESH_BINARY)
        #thresh_crop = masked_preprocessed_data[y:y + h, x:x + w]
        #plt.figure(1)
        #plt.imshow(edges)
        #plt.figure(2)
        #plt.imshow(preprocessed_image)
        #plt.figure(3)
        #plt.imshow(crop)
        #plt.show()

        white_pixel_count = np.sum(thresh_crop == 255)  # number of pixels, i.e. area, that are white
        pi = np.pi
        crop_height, crop_width = crop.shape
        area = pi * (outer_radius ** 2 - inner_radius ** 2)
        # print(area)
        #print(white_pixel_count)
        area_ratio = white_pixel_count / area
        # print(area_ratio)
        return crib, area_ratio, radii, thresh_crop, crop, preprocessed_image
    except TypeError:
        cv2.imwrite('cropped_results\\' + re.sub(r'.png', '', os.path.basename(filename)) + '_failed_no_circle_found.png',
                    unprocessed_image)
        pass


if __name__ == "__main__":
    main()
