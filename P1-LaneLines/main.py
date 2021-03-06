""" main.py """
import argparse
import os
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
from moviepy.editor import VideoFileClip
import utils as utils

FLAGS = None


def lane_detection_pipeline(img):
    """ Detect lane lines in an input image.
    Args:
        img: The original unmodified input image.
    Returns:
        A new image with lines drawn.
    """
    # calculate shape
    (height, width, num_channels) = img.shape

    white_mask = utils.color_threshold(img, rgb_min=[150, 150, 150])
    yellow_mask = utils.color_threshold(
        img, rgb_min=[150, 150, 0], rgb_max=[255, 255, 165])
    color_mask = white_mask & yellow_mask
    # yellow_white = utils.boolean_mask(img, color_mask)

    # convert image to gray scale
    gray = utils.grayscale(img)

    # blur image with Gaussian smoothing
    blur = utils.gaussian_blur(gray, 5)

    # Define our parameters for Canny and apply
    edges = utils.canny(blur, 50, 150)

    # Only keep the portions that originally had white or yellow
    color_masked = utils.boolean_mask(edges, color_mask)

    # mask everything but the region of interest
    vertices = np.array([[(0, height), (width * 6 / 13, height * 3 / 5),
                          (width * 7 / 13, height * 3 / 5), (width, height)]], dtype=np.int32)
    masked = utils.region_of_interest(color_masked, vertices)

    # Apply Hough transform
    detected_lines = utils.hough_lines(
        img=masked,
        rho=2,  # distance resolution in pixels of the Hough grid
        theta=np.pi / 180,  # angular resolution in radians of the Hough grid
        threshold=15,  # min num of votes (intersections in Hough grid cell)
        min_line_len=40,  # minimum number of pixels making up a line
        max_line_gap=20    # maximum gap in pixels between connectable line segments
    )

    # Overlay the detected lines on the original img
    output = utils.weighted_img(detected_lines, img)

    return output


def run_image_test(input_dir, output_dir):
    """ Performs lane detection on the test images """
    # Create an output dir if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the input_files into a list
    filenames = os.listdir(input_dir)

    for img_file in filenames:
        # open image
        input_img = utils.read_image(input_dir + img_file)
        # process image
        output_img = lane_detection_pipeline(input_img)
        # save the output image
        utils.write_image(output_dir + img_file, output_img, BGR=True)


def run_video_test(input_dir, output_dir):
    """ Performs lane detection on the test videos """
    # Create an output dir if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the input_files into a list
    filenames = os.listdir(input_dir)

    for video_file in filenames:
        output_file = output_dir + video_file
        input_file = input_dir + video_file
        # clip1 = VideoFileClip(input_file).subclip(0, 5)
        clip1 = VideoFileClip(input_file)
        output_clip = clip1.fl_image(lane_detection_pipeline)
        output_clip.write_videofile(output_file, audio=False)
        clip1.reader.close()
        clip1.audio.reader.close_proc()


def main():
    """ Main method """
    run_image_test(FLAGS.input_image_dir, FLAGS.output_image_dir)

    run_video_test(FLAGS.input_video_dir, FLAGS.output_video_dir)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        '--input_image_dir',
        type=str,
        default="./test_images/",
        help='Directory to read input images.'
    )

    PARSER.add_argument(
        '--output_image_dir',
        type=str,
        default="./test_images_output/",
        help='Directory to write output images.'
    )

    PARSER.add_argument(
        '--input_video_dir',
        type=str,
        default="./test_videos/",
        help='Directory to read input videos.'
    )

    PARSER.add_argument(
        '--output_video_dir', '-o',
        type=str,
        default="./test_video_output/",
        help='Directory to write output videos.'
    )

    FLAGS, _ = PARSER.parse_known_args()
    main()
