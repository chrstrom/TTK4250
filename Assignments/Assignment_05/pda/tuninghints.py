import numpy as np
from numpy import ndarray
from typing import Sequence


def tuninghints(measurement_data: Sequence[ndarray],
                association_gt_data: ndarray):
    """Function used to give hints how to tune the filter.

    Args:
        measurement_data (Sequence[ndarray]): the measurements
        association_gt_data (ndarray): the true associations
    """
    number_of_steps = len(association_gt_data)
    number_of_detections = len([a for a in association_gt_data if a != 0])
    total_number_of_clutter = (sum([len(zs) for zs in measurement_data])
                               - number_of_detections)

    z_xmin = min([z[0] for zs in measurement_data for z in zs])
    z_xmax = max([z[0] for zs in measurement_data for z in zs])
    z_ymin = min([z[1] for zs in measurement_data for z in zs])
    z_ymax = max([z[1] for zs in measurement_data for z in zs])

    clutter_density_estimate = "TODO"  # TODO
    detection_probability_estimate = "TODO"  # TODO

    print("Hints from tuninghints.py:")
    print(f"A reasonable clutter density is {clutter_density_estimate}")
    print(f"A reasonable detection probability is "
          f"{detection_probability_estimate}")
