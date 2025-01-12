import Share_memory_image_producer_consumer as shm
import numpy as np
from time import perf_counter_ns as perf_counter

# Create an Image4K_RGB object
image = shm.Image4K_RGB()

# Set data to 42
image.set_data(np.ones((2160, 3840, 3), dtype=np.uint8) * 42)
image.timestamp = 1234567890

# Retrieve data
retrieved_data = image.get_data()
print(f"Retrieved data shape: {retrieved_data.shape}")
print(f"Retrieved data[0, 0]: {retrieved_data[0, 0]}")
print(f"Retrieved timestamp: {image.timestamp}")

# Producer
def producer_example():
    # Create a producer instance
    shm_name = "shared_memory_4k_rgb"
    producer = shm.ProducerConsumer(shm_name)
    print("ProducerConsumer created:", producer)  # ensure this is a valid object
    
    # Create an Image4K_RGB object
    # image = shm.Image4K_RGB()
    # now = perf_counter()
    # image.timestamp = int(now)
    
    # Publish the Image4K_RGB object
    for _ in range(10):
        image.timestamp = int(perf_counter())
        producer.store(image)
        now = int(perf_counter())
        print("Store elapsed:", (now - image.timestamp) / 1e6, "ms")
    producer.close()


# Main
if __name__ == "__main__":
    producer_example()
