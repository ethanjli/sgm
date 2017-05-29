import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libc.string cimport memcpy

cdef extern from "opencv2/opencv.hpp":
    cdef int CV_8UC1

cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat() except +
        void create(int, int, int, void*) except +
        void create(int, int, int)
        void* data
        int type() const
        int cols
        int rows
        int channels()
        Mat clone() const

cdef extern from "src/disparity_method.h":
    cpdef void init_disparity_method(unsigned char p1, unsigned char p2)
    cdef Mat compute_disparity_method(Mat left, Mat right)
    cpdef void finish_disparity_method()

cdef void array2mat(np.ndarray arr, Mat& mat):
    assert(arr.ndim == 2, "ASSERT: Grayscale images only!")
    cdef int rows = arr.shape[0]
    cdef int cols = arr.shape[1]
    cdef int mat_type = CV_8UC1
    cdef unsigned int pixel_size = 1
    mat.create(rows, cols, mat_type)
    memcpy(mat.data, arr.data, rows * cols * pixel_size)

cdef void mat2array(Mat mat, np.ndarray arr):
    cdef int rows = mat.rows
    cdef int cols = mat.cols
    cdef int mat_type = CV_8UC1
    cdef unsigned int pixel_size = 1
    memcpy(arr.data, mat.data, rows * cols * pixel_size)

def compute_disparity(np.ndarray[np.uint8_t, ndim=2] left,
                      np.ndarray[np.uint8_t, ndim=2] right):
    cdef Mat left_mat
    cdef Mat right_mat
    cdef np.ndarray[np.uint8_t, ndim=2] disparity = np.empty((left.shape[0], left.shape[1]), dtype=np.uint8)

    array2mat(left, left_mat)
    array2mat(right, right_mat)
    cdef Mat disparity_mat = compute_disparity_method(left_mat, right_mat)
    mat2array(disparity_mat, disparity)
    return disparity
