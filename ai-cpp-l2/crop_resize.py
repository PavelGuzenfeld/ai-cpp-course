import numpy as np
import cv2
import time
from cpp_image_processor import crop_and_resize
from cpp_image_processor import crop_and_resize_unseq

def crop_and_resize_python(image, crop_start, crop_dim, target_dim):
    """
    Perform cropping and resizing in Python using OpenCV.
    """
    # Crop the image
    x, y = crop_start
    crop_width, crop_height = crop_dim
    cropped_image = image[y:y + crop_height, x:x + crop_width]

    # Resize the cropped image
    resized_image = cv2.resize(cropped_image, (target_dim[1], target_dim[0]))
    return resized_image

def compare_functions():
    # Create a dummy image (2048x1365 RGB image)
    # image_height, image_width = 2048, 1365
    # image = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)
    image_path = "ai-cpp-course/ai-cpp-l2/bmp-2048x1365.bmp"
    image = cv2.imread(image_path)
    if image is None:
        print("error: could not read the image.")
        sys.exit(1)
    # convert to numpy array (if not already)
    image = np.array(image)

    # Define cropping and resizing parameters
    crop_start = (50, 50)
    crop_dim = (200, 200)
    target_dim = (100, 100)

    # Time the Python function
    python_start = time.perf_counter()
    python_result = crop_and_resize_python(image, crop_start, crop_dim, target_dim)
    python_end = time.perf_counter()
    python_time_ms = (python_end - python_start) * 1000  # Convert to milliseconds

    # write the output image
    cv2.imwrite("python_result.bmp", python_result)

    print(f"Python crop_and_resize function took {python_time_ms:.2f} ms")

    # Time the C++ function
    cpp_start = time.perf_counter()
    cpp_result = crop_and_resize(
        image,
        crop_start[0],
        crop_start[1],
        crop_dim[0],
        crop_dim[1],
        target_dim[0],
        target_dim[1]
    )
    cpp_end = time.perf_counter()
    cpp_time_ms = (cpp_end - cpp_start) * 1000  # Convert to milliseconds
    
    # write the output image
    cv2.imwrite("cpp_result.bmp", cpp_result)

    print(f"C++ crop_and_resize function took {cpp_time_ms:.2f} ms")

    # Time the C++ function (unsequenced version)
    cpp_unseq_start = time.perf_counter()
    cpp_unseq_result = crop_and_resize_unseq(
        image,
        crop_start[0],
        crop_start[1],
        crop_dim[0],
        crop_dim[1],
        target_dim[0],
        target_dim[1]
    )
    cpp_unseq_end = time.perf_counter()
    cpp_unseq_time_ms = (cpp_unseq_end - cpp_unseq_start) * 1000  # Convert to milliseconds

    # write the output image
    cv2.imwrite("cpp_unseq_result.bmp", cpp_unseq_result)

    print(f"C++ crop_and_resize_unseq function took {cpp_unseq_time_ms:.2f} ms")

if __name__ == "__main__":
    compare_functions()
