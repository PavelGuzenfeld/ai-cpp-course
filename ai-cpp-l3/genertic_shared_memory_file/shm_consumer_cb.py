import Share_memory_image_producer_consumer as shm
import numpy as np

# Create an Image4K_RGB object
image = shm.Image4K_RGB()
image.timestamp = 123456789

# Set data
random_data = np.random.randint(0, 256, size=(2160, 3840, 3), dtype=np.uint8)
image.set_data(random_data)

# Retrieve data
retrieved_data = image.get_data()
print(f"Retrieved data shape: {retrieved_data.shape}")

# Producer
def producer_example():
    # Create a producer instance
    shm_name = "shared_memory_4k_rgb"
    producer = shm.ProducerConsumer.create(shm_name)
    print("ProducerConsumer created:", pc)  # ensure this is a valid object

    
    # Create a dummy 4K RGB image using numpy
    height, width, channels = 2160, 3840, 3  # 4K dimensions
    dummy_image = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)
    
    # Create an Image4K_RGB object
    image = shm.Image4K_RGB()
    image.timestamp = 123456789  # Example timestamp
    image.set_data(dummy_image)  # Set the numpy array data into the Image4K_RGB object
    
    # Publish the Image4K_RGB object
    producer.publish(image)
    print("Published image to shared memory")
    producer.close()


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

# Consumer with callback
def consumer_with_callback_example():
    # Create a consumer instance
    shm_name = "shared_memory_4k_rgb"
    consumer = shm.ProducerConsumer(shm_name)
    
    # Define a callback function
    def process_image(image):
        print(f"Processing image with shape: {image.shape}")
    
    # Consume with the callback
    consumer.consume_with_callback(process_image)
    consumer.close()

# Main
if __name__ == "__main__":
    # producer_example()
    consumer_example()
    # consumer_with_callback_example()
