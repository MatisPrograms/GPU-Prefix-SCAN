import sys

import numpy as np


def random_array(power_of_two, min_value=-100, max_value=100):
    return np.random.randint(min_value, max_value, 2 ** power_of_two, dtype=np.int32)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <power of 2>")
        sys.exit(2)
    else:
        arr = random_array(int(sys.argv[1]))
        str_arr = np.array2string(arr, separator=",", threshold=arr.shape[0]) \
            .strip('[]').replace('\n', '').replace(' ', '')
        print(str_arr)

        # Ask for input to save the array to a file
        save = input("Would you like to save this array to a file? (y/n): ")
        if save.lower().startswith('y'):
            filename = input("Please enter the filename: ")
            if not filename.endswith('.txt'):
                filename += '.txt'
            with open(filename, 'w') as f:
                f.write(str_arr)
                print("Saved to " + filename)
