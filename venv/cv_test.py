import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import glob2
import os
import shutil
import pprint

# todo restructure code so that it takes in lens info from the .csv database and locates jobs based on that information

def main():
    results = {}
    results_compare = {}
    lens_count_search = re.compile(r'-(\d)_')
    lens_rx_search = re.compile(r'(RX[-]?[0]?\d\d)')
    pixels_per_mm = 20.6
    passing_threshold = 0.65
    print('-----------------------')
    for file in glob2.glob('**\*_TotalPhaseF1.png'):
        try:
            crib_search = re.search(re.compile(r'C(\d\d)'), file)
            crib = int(crib_search.group(1))
            edge_area, radii, thresh_crop, crop, preprocessed_image = get_lens_edge(file,crib,pixels_per_mm)
            lens_count = re.search(lens_count_search, os.path.basename(file))
            lens_rx = re.search(lens_rx_search, os.path.basename(file))
            test_result, area_ratio = annular_area_test(thresh_crop,edge_area,passing_threshold)
            print('Lens crib diameter: ' + str(crib))
            print('The minimum ratio of white pixels to the theoretical number of white pixels was ' + str(
                min(area_ratio)))
            print('Test result: ' + test_result)

            results.update(
                {str(crib) + '-' + lens_rx.group(1) + '-' + lens_count.group(1): [area_ratio, test_result]})
            cv2.imwrite(
                'cropped_results\\' + re.sub(r'.png', '', os.path.basename(file)) + '_' + test_result + '_analyzed.png', thresh_crop)
            cv2.imwrite('cropped_results\\' + re.sub(r'.png', '', os.path.basename(file)) + '_' + test_result + '_unanalyzed.png', crop)
        except TypeError:
            print('File ' + file + ' did not have a found circle.')
            pass
    print('-----------------------')

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#this function is used to test single instances of new code
def test():
    test_image = cv2.imread('MR7 B375\C57\Rx -0.75\S40-RX-75-B375-C57-4\S40-RX-75-B375-C57-4_TotalPhaseF1.png')
    processed_test_image = defect_amplify(test_image)
    quadrants = []
    quadrants = return_image_quadrants(processed_test_image)
    for i in range(len(quadrants)):
        plt.figure(i)
        plt.imshow(quadrants[i])
    white_pixels = []
    white_pixels = white_pixel_ratio(quadrants, 1)
    for j in range(len(white_pixels)):
        white_pixels[j] = white_pixels[j]
    plt.show()
  # this function controls the
def preprocess_defect_image(image):
    blur_crop = cv2.blur(image, (3, 3))
    erode_crop = cv2.erode(blur_crop, (3, 3), iterations=2)
    ret_crop, thresh_crop = cv2.threshold(erode_crop, 190, 255, cv2.THRESH_BINARY)
    return thresh_crop

def annular_area_test(image,theoretical_area,passing_threshold):
    quadrants = []
    quadrants = return_image_quadrants(image)
    area_ratio = []
    area_ratio = white_pixel_ratio(quadrants, theoretical_area)
    if min(area_ratio) < passing_threshold:

        test_result = 'FAIL'
    else:
        print('Test passed. The minimum area ratio was ' + str(min(area_ratio)))
        test_result = 'PASS'
    return test_result, area_ratio


def prep_circle_find(unprocessed_image, strength):
    grayscale_image = unprocessed_image  # cv2.cvtColor(unprocessed_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.blur(grayscale_image, (strength + 4, strength + 4))
    ret, thresholded_image = cv2.threshold(blurred_image, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((strength + 2, strength + 2), np.uint8)
    dilated_image = cv2.dilate(thresholded_image, kernel, iterations=strength + 4)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=strength - 1)
    return dilated_image

def get_lens_edge(filename,crib,pixels_per_mm):
    # lens details preparation
    radius = crib * pixels_per_mm / 2

    # image preprocessing
    unprocessed_image = cv2.imread(filename, 0)
    height, width = unprocessed_image.shape
    mask = np.zeros((height, width), np.uint8)
    preprocessed_image = prep_circle_find(unprocessed_image, 2)
    edges = cv2.Canny(preprocessed_image, 150, 200)

    # find circles and process image
    try:
        radii = []
        found_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 500, param1=50, param2=30,
                                         minRadius=int(.98 * radius), maxRadius=int(1.05 * radius))
        for each_circle in found_circles[0, :]:
            radii.append((each_circle[2] - radius) / radius)
            ##define outer and inner radius
            outer_radius = int(each_circle[2] - 10 - (.25 * (radius - 520)))
            inner_radius = int(outer_radius * (.95))
            cv2.circle(mask, (each_circle[0], each_circle[1]), outer_radius, (255, 255, 255), thickness=-1)
            cv2.circle(mask, (each_circle[0], each_circle[1]), inner_radius, (0, 0, 0), thickness=-1)
            #debug print statements for edge shape analysis values
            # print('predicted circle radius = ' + str(radius))
            # print('found circle radius = ' + str(each_circle[2]))
            #print('Adjusted outer radius: ' + str(outer_radius))
            #print('Adjusted inner radius: ' + str(inner_radius))
            ##-----------------------------
        masked_data = cv2.bitwise_and(unprocessed_image, unprocessed_image, mask=mask)
        masked_preprocessed_data = cv2.bitwise_and(preprocessed_image, preprocessed_image, mask=mask)
        contours = cv2.findContours(masked_preprocessed_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x, y, w, h = cv2.boundingRect(contours[0])
        crop = masked_data[y:y + h, x:x + w]
        blur_crop = cv2.blur(crop, (4, 4))
        erode_crop = cv2.erode(blur_crop, (3, 3), iterations=1)
        ret_crop, thresh_crop = cv2.threshold(erode_crop, 190, 255, cv2.THRESH_BINARY)

        pi = np.pi
        edge_area = pi / 4 * ((outer_radius) ** 2 - inner_radius ** 2)

        return edge_area, radii, thresh_crop, crop, preprocessed_image
    except TypeError:
        cv2.imwrite(
            'cropped_results\\' + re.sub(r'.png', '', os.path.basename(filename)) + '_failed_no_circle_found.png',
            unprocessed_image)
        pass

def white_pixel_ratio(image_list, theoretical_white_area):
    pixel_count_list = []
    for i in range(len(image_list)):
        white_pixel_count = np.sum(image_list[i] == 255)
        pixel_count_list.append(white_pixel_count / theoretical_white_area)
        # print(theoretical_white_area)
        # print(white_pixel_count)
    return pixel_count_list

def return_image_quadrants(image):
    h, w = image.shape
    half_width = int(w / 2)
    remaining_width = w - half_width
    half_height = int(h / 2)
    remaining_height = h - half_height
    quadrants = []
    quadrants.append(image[half_height:h, half_width:w])
    quadrants.append(image[half_height:h, 0:half_width])
    quadrants.append(image[0:half_height, 0:half_width])
    quadrants.append(image[0:half_height, half_width:w])
    return quadrants


if __name__ == "__main__":
    main()
