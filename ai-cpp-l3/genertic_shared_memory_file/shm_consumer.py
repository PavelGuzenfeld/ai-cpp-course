import Share_memory_image_producer_consumer as shm
from time import perf_counter_ns as perf_counter, sleep
import numpy as np

# Consumer
def consumer_example(repeat=10) -> list:
    # Create a consumer instance
    shm_name = "shared_memory_4k_rgb"
    consumer = shm.ProducerConsumer(shm_name)
    
    # Consume the image
    result = []
    for _ in range(repeat):
        retrieved_image = consumer.load()
        frame_id = retrieved_image.frame_number
        elapsed_time = (int(perf_counter()) - retrieved_image.timestamp) / 1e6  # in milliseconds
        result.append(elapsed_time)
        if retrieved_image.frame_number == repeat - 1:
            break
    consumer.close()
    return result

def atomic_consumer_example(repeat=10) -> list:
    # Create a consumer instance
    shm_name = "shared_memory_4k_rgb_atomic"
    consumer = shm.AtomicProducerConsumer(shm_name)
    
    # Consume the image
    result = []
    for _ in range(repeat):
        retrieved_image = consumer.load()
        timestamp = retrieved_image.timestamp
        elapsed_time = (int(perf_counter()) - timestamp) / 1e6  # in milliseconds
        result.append(elapsed_time)
        if retrieved_image.frame_number == repeat - 1:
            break
    consumer.close()
    return result

# Main
if __name__ == "__main__":
    REPEAT = 100
    consumer_result = consumer_example(REPEAT)
    atomic_consumer_result = atomic_consumer_example(REPEAT)

    #Post processing statistics
    print("Consumer elapsed time:", consumer_result)
    print("Atomic Consumer elapsed time:", atomic_consumer_result)
    #Remove outliers (max 2 std)
    consumer_result = [i for i in consumer_result if i < np.mean(consumer_result) + 2 * np.std(consumer_result)]
    atomic_consumer_result = [i for i in atomic_consumer_result if i < np.mean(atomic_consumer_result) + 2 * np.std(atomic_consumer_result)]
    print("Consumer mean:", np.mean(consumer_result))
    print("Atomic Consumer mean:", np.mean(atomic_consumer_result))


