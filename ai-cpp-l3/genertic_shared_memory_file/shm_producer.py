import Share_memory_image_producer_consumer as shm
import Share_memory_image_producer_consumer_nb as shm_nb
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

#Create an image Image4K_RGB_NB object
image_nb = shm_nb.Image4K_RGB()

# Set data to 42
image_nb.set_data(np.ones((2160, 3840, 3), dtype=np.uint8) * 42)
image_nb.timestamp = 1234567890

# Retrieve data
retrieved_data = image_nb.get_data()
print(f"Retrieved data shape: {image_nb.shape}")
print(f"Retrieved data[0, 0]: {retrieved_data[0, 0]}")
print(f"Retrieved timestamp: {image_nb.timestamp}")

# Producer
def producer_example(repeat=10) -> list:
    # Create a producer instance
    shm_name = "shared_memory_4k_rgb"
    producer = shm.ProducerConsumer(shm_name)
    print("ProducerConsumer created:", producer)  # ensure this is a valid object
    
    result = []
    for frame_id in range(repeat):
        image.timestamp = int(perf_counter())
        image.frame_number = frame_id
        print(f"Storing image frame number: {image.frame_number}")
        producer.store(image)
        result.append((int(perf_counter()) - image.timestamp) / 1e6)
    producer.close()
    return result


def atomic_producer_example(repeat=10) -> list:
    # Create a producer instance
    shm_name = "shared_memory_4k_rgb_atomic"
    producer = shm.AtomicProducerConsumer(shm_name)
    print("AtomicProducerConsumer created:", producer)  # ensure this is a valid object

    result = []
    for frame_id in range(repeat):
        image.timestamp = int(perf_counter())
        image.frame_number = frame_id
        producer.store(image)
        result.append((int(perf_counter()) - image.timestamp) / 1e6)
    producer.close()
    return result


def producer_example_nb(repeat=10) -> list:
    # Create a producer instance
    shm_name = "shared_memory_4k_rgb_nb"
    producer = shm_nb.ProducerConsumer.create(shm_name)
    print("ProducerConsumer created:", producer)  # ensure this is a valid object

    result = []
    for frame_id in range(repeat):
        image_nb.timestamp = int(perf_counter())
        image_nb.frame_number = frame_id
        producer.store(image_nb)
        result.append((int(perf_counter()) - image_nb.timestamp) / 1e6)
    producer.close()
    return result


def atomic_producer_example_nb(repeat=10) -> list:
    # Create a producer instance
    shm_name = "shared_memory_4k_rgb_atomic_nb"
    producer = shm_nb.AtomicProducerConsumer.create(shm_name)
    print("AtomicProducerConsumer created:", producer)  # ensure this is a valid object

    result = []
    for frame_id in range(repeat):
        image_nb.timestamp = int(perf_counter())
        image_nb.frame_number = frame_id
        producer.store(image_nb)
        result.append((int(perf_counter()) - image_nb.timestamp) / 1e6)
    producer.close()
    return result

# Main
if __name__ == "__main__":
    REPEAT = 100
    prod_result = producer_example(REPEAT)
    atomic_result = atomic_producer_example(REPEAT)
    prod_result_nb = producer_example_nb(REPEAT)
    atomic_result_nb = atomic_producer_example_nb(REPEAT)


    #Remove outliers (max 2 std)
    prod_result = [i for i in prod_result if i < np.mean(prod_result) + 2 * np.std(prod_result)]
    atomic_result = [i for i in atomic_result if i < np.mean(atomic_result) + 2 * np.std(atomic_result)]
    prod_result_nb = [i for i in prod_result_nb if i < np.mean(prod_result_nb) + 2 * np.std(prod_result_nb)]
    atomic_result_nb = [i for i in atomic_result_nb if i < np.mean(atomic_result_nb) + 2 * np.std(atomic_result_nb)]

    print("Producer mean:", np.mean(prod_result))
    print("Producer std:", np.std(prod_result))
    print("--------------------")
    print("Atomic Producer mean:", np.mean(atomic_result))
    print("Atomic Producer std:", np.std(atomic_result))
    print("Atomic Producer is faster by percentage:", (np.mean(prod_result) - np.mean(atomic_result)) / np.mean(prod_result) * 100, "%")
    print("--------------------")
    print("Producer NB mean:", np.mean(prod_result_nb))
    print("Producer NB std:", np.std(prod_result_nb))
    print("Producer NB is faster by percentage:", (np.mean(prod_result) - np.mean(prod_result_nb)) / np.mean(prod_result) * 100, "%")
    print("--------------------")
    print("Atomic Producer NB mean:", np.mean(atomic_result_nb))
    print("Atomic Producer NB std:", np.std(atomic_result_nb))
    print("Atomic Producer NB is faster by percentage:", (np.mean(prod_result) - np.mean(atomic_result_nb)) / np.mean(prod_result) * 100, "%")
