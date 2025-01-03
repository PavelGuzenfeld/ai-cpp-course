import Share_memory_image_producer_consumer as shm

# Consumer
def consumer_example():
    # Create a consumer instance
    shm_name = "shared_memory_4k_rgb"
    consumer = shm.ProducerConsumer(shm_name)
    
    # Consume the image
    retrieved_image = consumer.load()
    print("Retrieved image from shared memory")
    
    # Convert the retrieved Image4K_RGB object to a numpy array
    # retrieved_data = retrieved_image.get_data()
    # print(f"Retrieved image shape: {retrieved_data.shape}")
    consumer.close()

# Main
if __name__ == "__main__":
    # producer_example()
    consumer_example()
    # consumer_with_callback_example()
