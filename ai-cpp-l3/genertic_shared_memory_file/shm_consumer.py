import Share_memory_image_producer_consumer as shm
import Share_memory_image_producer_consumer_nb as shm_nb
import numpy as np
from time import perf_counter_ns as perf_counter

# Consumer
def consumer_example(repeat=10) -> list:
    # Create a consumer instance
    shm_name = "shared_memory_4k_rgb"
    consumer = shm.ProducerConsumer(shm_name)
    
    # Consume the image
    result = []
    for _ in range(repeat):
        start = int(perf_counter())
        retrieved_image = consumer.load()
        # timestamp = retrieved_image.timestamp
        end = int(perf_counter())
        elapsed_time = (end - start) / 1e6  # in milliseconds
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
        start = int(perf_counter())
        retrieved_image = consumer.load()
        # timestamp = retrieved_image.timestamp
        end = int(perf_counter())
        elapsed_time = (end - start) / 1e6  # in milliseconds
        result.append(elapsed_time)
        if retrieved_image.frame_number == repeat - 1:
            break
    consumer.close()
    return result

def consumer_example_nb(repeat=10) -> list:
    # Create a consumer instance
    shm_name = "shared_memory_4k_rgb_nb"
    consumer = shm_nb.ProducerConsumer.create(shm_name)
    
    # Consume the image
    result = []
    for _ in range(repeat):
        start = int(perf_counter())
        retrieved_image = consumer.load()
        # timestamp = retrieved_image.timestamp
        end = int(perf_counter())
        elapsed_time = (end - start) / 1e6  # in milliseconds
        result.append(elapsed_time)
        if retrieved_image.frame_number == repeat - 1:
            break
    consumer.close()
    return result

def atomic_consumer_example_nb(repeat=10) -> list:
    # Create a consumer instance
    shm_name = "shared_memory_4k_rgb_atomic_nb"
    consumer = shm_nb.AtomicProducerConsumer.create(shm_name)
    
    # Consume the image
    result = []
    for _ in range(repeat):
        start = int(perf_counter())
        retrieved_image = consumer.load()
        # timestamp = retrieved_image.timestamp
        end = int(perf_counter())
        elapsed_time = (end - start) / 1e6  # in milliseconds
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
    consumer_result_nb = consumer_example_nb(REPEAT)
    atomic_consumer_result_nb = atomic_consumer_example_nb(REPEAT)

    #Post processing statistics
    print("Consumer elapsed time:", consumer_result)
    print("Atomic Consumer elapsed time:", atomic_consumer_result)
    print("Consumer NB elapsed time:", consumer_result_nb)
    print("Atomic Consumer NB elapsed time:", atomic_consumer_result_nb)
    #Remove outliers (max 2 std)
    consumer_result = [i for i in consumer_result if i < np.mean(consumer_result) + 2 * np.std(consumer_result)]
    atomic_consumer_result = [i for i in atomic_consumer_result if i < np.mean(atomic_consumer_result) + 2 * np.std(atomic_consumer_result)]
    consumer_result_nb = [i for i in consumer_result_nb if i < np.mean(consumer_result_nb) + 2 * np.std(consumer_result_nb)]
    atomic_consumer_result_nb = [i for i in atomic_consumer_result_nb if i < np.mean(atomic_consumer_result_nb) + 2 * np.std(atomic_consumer_result_nb)]

    print("Consumer mean:", np.mean(consumer_result))
    print("Atomic Consumer mean:", np.mean(atomic_consumer_result))
    print("Consumer NB mean:", np.mean(consumer_result_nb))
    print("Atomic Consumer NB mean:", np.mean(atomic_consumer_result_nb))


