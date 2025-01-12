import Share_memory_image_producer_consumer as shm
from time import perf_counter_ns as perf_counter

# Consumer
def consumer_example():
    # Create a consumer instance
    shm_name = "shared_memory_4k_rgb"
    consumer = shm.ProducerConsumer(shm_name)
    
    # Consume the image
    for _ in range(10):
        retrieved_image = consumer.load()
        timestamp = retrieved_image.timestamp
        now = int(perf_counter())
        print("Image timestamp:", timestamp)
        print("Now timestamp:", now)
        print("Time elapsed:", (now - timestamp), "ns")
        print("Time elapsed ms:", (now - timestamp) / 1e6, "ms")
    consumer.close()

# Main
if __name__ == "__main__":
    # producer_example()
    consumer_example()
    # consumer_with_callback_example()
