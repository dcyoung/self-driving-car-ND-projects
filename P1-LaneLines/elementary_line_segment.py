""" elementary_line_segment.py """
import math
import numpy as np
from sklearn.cluster import KMeans


class ESL:
    """ Elementary Line Segment """

    def __init__(self, x1, y1, x2, y2):
        """ Constructor """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.delta_x = self.x2 - self.x1
        if self.delta_x is 0:
            self.delta_x += np.finfo(float).eps
        self.delta_y = self.y2 - self.y1

    def get_slope(self):
        """ Computes the slope (m) from line equation y=mx+b """
        return self.delta_y / self.delta_x

    def get_bias(self):
        """ Computes the bias (b) from line equation y=mx+b """
        return self.y1 - self.get_slope() * self.x1

    def get_polar(self):
        """ Returns the polar representation of the line
        rho: closest distance from origin to the line.
        theta: angle away from horizontal
        """
        theta = math.atan2(self.delta_y, self.delta_x)
        rho = self.x1 * math.sin(theta) + self.y1 * math.cos(theta)
        return (rho, theta)


def get_average_x(segments):
    """ Returns the average x coordinate among the segments """
    x_avg = 0
    for e in segments:
        x_avg += (e.x1 + e.x2)
    return x_avg / (2 * len(segments))


def get_cluster_assignments(segments, num_clusters=2):
    """ Clusters the elementary line segments based on their polar form """
    num_segments = len(segments)
    polar_lines = np.empty((num_segments, 2))

    # Package the line's polar forms nicely in a 2d array
    for i, e in enumerate(segments):
        polar_lines[i, :] = e.get_polar()

    # Perform k-means clustering on the polar form with 2 clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(polar_lines)
    return kmeans.labels_


def get_average_slope(segments):
    """ Averages the slope m (y = mx + b) among the segments """
    avg_slope = 0
    for e in segments:
        avg_slope += e.get_slope()
    return avg_slope / len(segments)


def get_average_bias(segments):
    """ Averages the bias b (y = mx + b) among the segments """
    avg_bias = 0
    for e in segments:
        avg_bias += e.get_bias()
    return avg_bias / len(segments)


def main():
    """ Main method """
    # Create some elementary line segments
    segments = []
    segments.append(ESL(0, 0, 1, 1))
    segments.append(ESL(1, 1, 2, 2))
    segments.append(ESL(0, 0, -1, 1))
    segments.append(ESL(-1, 1, -2, 2))

    # Perform k-means clustering on the polar form with 2 clusters
    print(get_cluster_assignments(segments))


if __name__ == '__main__':
    main()
