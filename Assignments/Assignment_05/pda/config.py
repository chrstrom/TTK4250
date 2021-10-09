# set to false for speedup.
# Speedup will also occur if argument '-O' is given to python,
# as __debug__ then is False
DEBUG = True and __debug__


sigma_a = 2.6  # acceleration standard deviation
sigma_z = 3.1  # measurement standard deviation

# clutter density, (measurements per m^2, is this reasonable?)
clutter_density = 1e-6

# detection probability, (how often cyan dot appear, is this reasonable?)
detection_prob = 0.5

# gate percentile, (estimated percentage of correct measurements that will be
# accepted by gate function)
gate_percentile = 0.8
