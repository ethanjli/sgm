import numpy as np
cimport numpy as np

cdef extern from "src/disparity_method.h":
    cpdef void init_disparity_method(unsigned char p1, unsigned char p2)
    cpdef void finish_disparity_method()
