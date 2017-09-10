""" main.py """
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import utils as utils
import os

FLAGS = None


def lane_detection_pipelines(img):
    """ Detect lane lines in an input image.
    Args:
        img: The original unmodified input image.
    Returns:
        A new image with lines drawn.
    """
    # calculate shape
    (height, width, num_channels) = img.shape

    # convert image to gray scale
    gray = utils.grayscale(img)

    # blur image with Gaussian smoothing
    blur = utils.gaussian_blur(gray, 5)

    # Define our parameters for Canny and apply
    edges = utils.canny(blur, 50, 150)

    # mask everything but the region of interest
    vertices = np.array([[(0, height), (width * 6 / 13, height * 3 / 5),
                          (width * 7 / 13, height * 3 / 5), (width, height)]], dtype=np.int32)
    masked_edges = utils.region_of_interest(edges, vertices)

    # Apply Hough transform
    detected_lines = utils.hough_lines(
        img=masked_edges,
        rho=2,  # distance resolution in pixels of the Hough grid
        theta=np.pi / 180,  # angular resolution in radians of the Hough grid
        threshold=15,  # min num of votes (intersections in Hough grid cell)
        min_line_len=40,  # minimum number of pixels making up a line
        max_line_gap=20    # maximum gap in pixels between connectable line segments
    )

    # Overlay the detected lines on the original img
    output = utils.weighted_img(detected_lines, img)
    return output


def main():
    """ Main method """

    # Create an output dir if necessary
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    # Read the input_files into a list
    filenames = os.listdir(FLAGS.input_dir)

    for img_file in filenames:
        # open image
        input_img = utils.read_image(FLAGS.input_dir + img_file)
        # process image
        output_img = lane_detection_pipelines(input_img)
        # save the output image
        utils.write_image(FLAGS.output_dir + img_file, output_img)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        '--input_dir', '-i',
        type=str,
        default="./test_images/",
        help='Directory to read input files.'
    )

    PARSER.add_argument(
        '--output_dir', '-o',
        type=str,
        default="./test_images_output/",
        help='Directory to write output files.'
    )
    PARSER.add_argument(
        '--float_arg',
        type=float,
        default=0.5,
        help='Help message.'
    )
    PARSER.add_argument(
        '--int_arg',
        type=int,
        default=1,
        help='Help message.'
    )

    FLAGS, _ = PARSER.parse_known_args()
    main()
