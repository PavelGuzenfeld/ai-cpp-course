import image_shm_dblbuff as shm_nb
from time import sleep, perf_counter
import numpy as np

shm_name = "shared_memory_4k_rgb_atomic_nb"
shm = shm_nb.DoubleBufferShem(shm_name)
print("Consumer created:", shm)

img = shm_nb.Image4K_RGB()
img.timestamp = 111
img.frame_number = 222
data_42 = np.ones((2160, 3840, 3), dtype=np.uint8) * 42
img.set_data(data_42)
shm.store(img)

start = int(perf_counter())
retrieved_image = shm.load()
end = int(perf_counter())

print(f"Load elapsed time: {end - start} ms")

print(f"Retrieved image: {retrieved_image}")
print(f"Retrieved image timestamp: {retrieved_image.timestamp()}")
print(f"Retrieved image frame number: {retrieved_image.frame_number()}")

sleep(1)

print(f"Retrieved image: {retrieved_image}")
print(f"Retrieved image timestamp: {retrieved_image.timestamp()}")
print(f"Retrieved image frame number: {retrieved_image.frame_number()}")

print(f"Retrieved image data shape: {retrieved_image.get_data()}")
    