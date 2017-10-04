"""" utils.py """
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import elementary_line_segment as esl


def read_image(filepath, bReadWithOpenCV=False):
    """ Reads an image from a file
    Args:
        The path to the image file
    Returns:
        image
    """
    if bReadWithOpenCV:
        return cv2.imread(filepath)
    return mpimg.imread(filepath)


def write_image(filepath, img, BGR=False):
    """ Writes an image to a file
    Args:
        filepath: Where to write the image
        img: The image to write
    """

    if BGR:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filepath, rgb)
    else:
        cv2.imwrite(filepath, img)


def blank_img(img):
    """ Creates a new blank image the same size as img
    Args:
        img: The image with the desired dimensions
    """
    return np.copy(img) * 0


def boolean_mask(img, mask):
    """ Applies the boolean mask to the image """
    masked = np.copy(img)
    if len(img.shape) is 2:
        masked[mask] = 0
    else:
        masked[mask] = [0, 0, 0]
    return masked


def color_threshold(img, rgb_min=[200, 200, 200], rgb_max=[255, 255, 255]):
    """ Returns an image thresholded by the color criteria """
    # Mask pixels below the threshold
    color_thresholds = (img[:, :, 0] < rgb_min[0]) | (img[:, :, 1] < rgb_min[1]) | (img[:, :, 2] < rgb_min[2]) | (
        img[:, :, 0] > rgb_max[0]) | (img[:, :, 1] > rgb_max[1]) | (img[:, :, 2] > rgb_max[2])
    return color_thresholds


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def divide_segments(segments):
    """ Divides the lines into two sets """
    cluster_assignments = esl.get_cluster_assignments(segments, num_clusters=2)
    set_1 = []
    set_2 = []
    for i, seg in enumerate(segments):
        if np.equal(cluster_assignments[i], 0):
            set_1.append(seg)
        else:
            set_2.append(seg)

    # compare the average slope to determine which set of lines is on the left
    # return left first, right second
    return (set_1, set_2) if esl.get_average_slope(set_1) > esl.get_average_slope(set_2) else (set_2, set_1)


def remove_unlikely_candidates(segments):
    """ Returns a new list with unlikely candidates removed """
    survivors = []
    for e in segments:
        if abs(e.get_slope()) > 0.5 and abs(e.get_slope()) < 2.0:
            survivors.append(e)
    return survivors


def draw_lines(img, lines, color=[255, 0, 0], thickness=2, y_2=None):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if lines is None:
        return

    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    # return
    # convert cv2 lines to esl segments
    segments = [esl.ESL(x1, y1, x2, y2) for l in lines for x1, y1, x2, y2 in l]

    segments = remove_unlikely_candidates(segments)
    if len(segments) <= 1:
        return

    # group them into left and right sets
    left_set, right_set = divide_segments(segments)

    (height, _, _) = img.shape

    m_left = esl.get_average_slope(left_set)
    m_right = esl.get_average_slope(right_set)
    b_left = esl.get_average_bias(left_set)
    b_right = esl.get_average_bias(right_set)

    # Recall y=mx+b, so x = (y-b)/m
    y_1 = height
    if y_2 is None:
        y_2 = int(round(height * 3 / 5))

    if m_left == float("inf") or m_left == float("-inf") or b_left == float("inf") or b_left == float("-inf"):
        x1_left = int(round(esl.get_average_x(left_set)))
        x2_left = x1_left
    else:
        x1_left = int(round((y_1 - b_left) / m_left))
        x2_left = int(round((y_2 - b_left) / m_left))
    if m_right == float("inf") or m_right == float("-inf") or b_right == float("inf") or b_right == float("-inf"):
        x1_right = int(round(esl.get_average_x(right_set)))
        x2_right = x1_right
    else:
        x1_right = int(round((y_1 - b_right) / m_right))
        x2_right = int(round((y_2 - b_right) / m_right))

    col_left = [255, 0, 0]
    col_right = [0, 0, 255]
    cv2.line(img, (x1_left, y_1), (x2_left, y_2), col_left, thickness)
    cv2.line(img, (x1_right, y_1), (x2_right, y_2), col_right, thickness)


def hough_lines(img, rho=2, theta=np.pi / 180, threshold=15, min_line_len=40, max_line_gap=20):
    """ Detect lines in an image of edges
    Args:
        img: the output of a Canny transform.
        rho: distance resolution in pixels of the Hough grid
        theta: angular resolution in radians of the Hough grid
        threshold: min num of votes (intersections in Hough grid cell)
        min_line_len:  minimum number of pixels making up a line
        max_line_gap: maximum gap in pixels between connectable line segments

    Returns:
        an image with hough lines drawn
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows (Python 3 supports math symbols):

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    if len(initial_img.shape) is 2:
        initial_img = cv2.cvtColor(initial_img, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(initial_img, α, img, β, λ)
