# set to false for speedup.
# Speedup will also occur if argument '-O' is given to python,
# as __debug__ then is False
DEBUG = False and __debug__


sigma_a = 2.2  # acceleration standard deviation
sigma_z = 3.2  # measurement standard deviation

# clutter density, (measurements per m^2, is this reasonable?)
clutter_density = 0.003

# detection probability, (how often cyan dot appear, is this reasonable?)
detection_prob = 0.896

# gate percentile, (estimated percentage of correct measurements that will be
# accepted by gate function)
gate_percentile = 0.999
