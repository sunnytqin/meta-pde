"""Simple example on how to log scalars and images to tensorboard without tensor ops.
License: Copyleft
"""
__author__ = "Michael Gygli"

import tensorflow as tf
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import numpy as np
import os
import random


class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)
        image_dir = os.path.join(log_dir, "images")
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)

    def log_images(self, tag, images, step):
        """Logs a list of images."""
        with self.writer.as_default():
            tf.summary.image(
                tag,
                [tf.io.decode_image(img.getvalue()) for img in images],
                step=step,
                max_outputs=4,
            )

    def log_plots(self, tag, figures, step):
        """Logs a list of plots."""

        images = []
        for nr, fig in enumerate(figures):
            # Write the image to a string
            s = BytesIO()
            plt.figure(fig.number)
            plt.savefig(s, format="png", dpi=800)
            # width, height = fig.get_size_inches() * fig.get_dpi()
            # width = int(width)
            # height = int(height)
            # Create an Image object
            images.append(s)

        self.log_images(tag, images, step)

    def log_histogram(self, tag, values, step, bins=25):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create and write Summary
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step, buckets=bins)
