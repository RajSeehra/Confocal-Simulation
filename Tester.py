import numpy as np

centre_x = 400
centre_y = 500
a_x = -centre_x  # distance of left edge of screen relative to centre x.
b_x = 1000 - centre_x  # distance of right edge of screen relative to centre x.
a_y = -centre_y  # distance of top edge of screen relative to centre y.
b_y = 1000 - centre_y  # distance of bottom edge of screen relative to centre y.

r = 100
# Produce circle mask, ones grid = to original file and cut out.
y, x = np.ogrid[a_x:b_x, a_y:b_y]  # produces a list which collapses to 0 at the centre in x and y
mask = x * x + y * y <= r * r  #
ones = np.ones((1000, 1000))
ones[mask] = 0