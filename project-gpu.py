# Author: Matis Herrmann
# Date modified: 01/05/2023
# Description: Perform parallel prefix sum on a list of numbers using a GPU
# Usage: python project-gpu.py <inputFile> [--tb int] [--independent] [--inclusive]
# Input: <inputFile> - File containing input array
#        [--tb int] - Size of thread block
#        [--independent] - Perform independent scans on sub-arrays
#        [--inclusive] - Perform an inclusive scan
# Output: <outputFile> - Printed values separated by commas of the transformed array

import argparse
import math
import sys

import numpy as np
from numba import cuda

# Constants
VERBOSE = True
MAX_THREAD_BLOCK_SIZE = 1024
THREAD_BLOCKS = 64


def vprint(*args, **kwargs):
    """
    Print if verbose is enabled
    :param args: print arguments
    :param kwargs: print keyword arguments
    :return: None
    """
    if VERBOSE:
        print(*args, **kwargs)


@cuda.jit(device=True)
def to_shared(values, n):
    """
    Copy values to shared memory
    :param values: Array of values to copy
    :param n: Number of values to copy
    :return: None
    """
    pass


@cuda.jit(device=True)
def from_shared(values, n):
    """
    Copy values from shared memory
    :param values: Array of values to copy to
    :param n: Number of values to copy
    :return: None
    """
    pass


@cuda.jit(device=True)
def sweep_up(values, n, m):
    pass


@cuda.jit(device=True)
def sweep_down(values, n, m):
    pass


@cuda.jit
def scan_kernel(values, n):
    return values


def scan_gpu_exclusive(values, independent):
    """
    Perform an exclusive scan on the input array using the Nvidia GPU
    :param values: Array of values to scan
    :param independent: Whether to perform independent scans on sub-arrays
    :return: New array of transformed values
    """

    # Calculate the number of blocks needed
    num_blocks = math.ceil(len(values) / THREAD_BLOCKS)

    # Calculate the closest power of 2 of length of values
    n = 2 ** int(math.log2(len(values) / num_blocks))

    # Copy values to the GPU
    gpu_values = cuda.to_device(values)

    if independent:
        vprint("Performing independent scans on sub-arrays")
        scan_kernel[num_blocks, THREAD_BLOCKS](gpu_values, n)
    else:
        vprint("Performing scan on the entire array")
        scan_kernel[num_blocks, THREAD_BLOCKS](gpu_values, len(values))

    return values


def scan_gpu_inclusive(values, independent):
    return values


def scan_gpu(values, inclusive, independent):
    """
    Perform a scan on the input array using the Nvidia GPU
    :param values: Array of values to scan
    :param inclusive: Whether to perform an inclusive scan
    :param independent: Whether to perform independent scans on sub-arrays
    :return: New array of transformed values
    """
    if inclusive:
        vprint("Performing inclusive scan")
        return scan_gpu_inclusive(values, independent)
    else:
        vprint("Performing exclusive scan")
        return scan_gpu_exclusive(values, independent)


if __name__ == '__main__':
    # Check for correct number of arguments
    if len(sys.argv) < 2:
        # Print usage statement and exit with an error
        sys.stderr.write("Incorrect arguments usage - Please use:\npython project-gpu.py "
                         "<inputFile> [--tb int] [--independent] [--inclusive]")
        exit(1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Perform parallel prefix sum on a list of numbers using a GPU')
    parser.add_argument('input_file', type=str, help='File containing input array')
    parser.add_argument('--tb', type=int, help='Size of thread block')
    parser.add_argument('--independent', action='store_true', help='Perform independent scans on sub-arrays')
    parser.add_argument('--inclusive', action='store_true', help='Perform an inclusive scan')
    args = parser.parse_args()

    # Input file
    with open(args.input_file, 'r') as f:
        array_values = np.fromstring(f.read().strip(), dtype=np.int32, sep=',')
        vprint("Input array: " + str(array_values))

    # Check if input array is empty
    if len(array_values) == 0:
        sys.stderr.write("Error: Input array is empty")
        exit(1)

    # TB size if set, between 1 and 1024
    if args.tb and 1 <= args.tb <= MAX_THREAD_BLOCK_SIZE:
        # Round up to nearest power of 2
        THREAD_BLOCKS = 2 ** int(math.log2(args.tb))
        vprint("Thread block size: " + str(THREAD_BLOCKS))

    # Scanning method depends on command line arguments
    array_values = scan_gpu(array_values, args.inclusive, args.independent)

    # Print input array
    vprint("Result: ")
    print(np.array2string(array_values, separator=",", threshold=array_values.shape[0]) \
          .strip('[]').replace('\n', '').replace(' ', ''))
