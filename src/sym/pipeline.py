"""Pipeline for generating synthetic particle images and applying flow fields."""
import queue
import threading
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt

# Import your existing modules
from src.sym.processing import process_flow_field_on_gpus
from src.sym.scheduler import FlowFieldScheduler

# Buffer settings
FLOW_BUFFER_MAXSIZE = 5  # Maximum number of flow fields buffered
flow_field_buffer = queue.Queue(maxsize=FLOW_BUFFER_MAXSIZE)

# Optional: buffer for generated images (for testing purposes)
# In a real pipeline, you might directly pass them into your training loop
IMAGE_BUFFER_MAXSIZE = 1000
image_buffer = queue.Queue(maxsize=IMAGE_BUFFER_MAXSIZE)


def training_pipeline(images: jnp.ndarray):
    """Dummy function to represent the training/testing pipeline.

    Replace this with your actual training code.

    Args:
        images (jnp.ndarray): A (num_images, H, W) array of synthetic particle images.
    """
    # print(f"Feeding {images.shape[0]} images into the pipeline.")
    # Clear the image_buffer properly instead of just checking if it's empty.
    while not image_buffer.empty():
        image_buffer.get_nowait()  # remove existing images

    # Simulate processing time
    # time.sleep(0.5)


def main_loop():
    """Main loop.

    1. Retrieves the next flow field from the buffer.
    2. Generates a batch of images with the flow field.
    3. Feeds the images into the training/testing pipeline.
    """
    num_images = 10000  # Number of images to generate per flow field
    batch_count = 0
    while True:
        try:
            flow_field = jnp.array(
                flow_field_buffer.get(timeout=15)
            )  # wait up to 5 seconds
        except queue.Empty:
            print("No more flow fields, exiting main loop.")
            break

        print(f"Processing batch {batch_count}: generating images...")
        try:
            imgs1, imgs2 = process_flow_field_on_gpus(
                flow_field, image_shape=flow_field.shape[:2]
            )
            print("Image buffer size after putting:", image_buffer.qsize() + 2)
            for i in range(num_images):
                # Save the generated images to the image buffer
                # print("Generated images shapes:", imgs1.shape, imgs2.shape)

                # Save the images to disk for debugging
                plt.imsave(f"generated_image_a_{i}.png", imgs1[i], cmap="gray")
                plt.imsave(f"generated_image_b_{i}.png", imgs2[i], cmap="gray")

                image_buffer.put(imgs1[i])
                image_buffer.put(imgs2[i])

            # Save the first image for debugging
            # save_jax_image(imgs1, f"generated_image_{batch_count}_{i}.png")
            # Save the second image for debugging
            # save_jax_image(imgs2, f"generated_image_{batch_count}_{i}_after.png")
            print(f"Batch {batch_count} complete: generated {num_images} images.")
        except Exception as e:
            print(f"Error processing flow field for batch {batch_count}: {e}")

        batch_count += 1


if __name__ == "__main__":
    # Example usage:
    files = [
        "/shared/fluids/channel_full_ts_0004.h5",
        "/shared/fluids/channel_full_ts_0008.h5",
        "/shared/fluids/channel_full_ts_0012.h5",
        "/shared/fluids/channel_full_ts_0016.h5",
    ]

    # Start a thread to load flow fields
    def load_flow_fields(files):
        """Load flow fields from list of files and puts them in a buffer.

        Args:
            files (list): List of HDF5 file paths.
        """
        scheduler = FlowFieldScheduler(files, randomize=True, loop=True)
        for epoch in range(2):  # Example: iterate over two epochs
            for _ in range(len(scheduler)):
                flow_field = next(scheduler)
                # Use a timeout to avoid blocking indefinitely if the buffer is full.
                while True:
                    try:
                        flow_field_buffer.put(flow_field, timeout=1)
                        break  # succeeded in putting; exit loop
                    except queue.Full:
                        time.sleep(0.1)  # brief pause before retrying

    flow_thread = threading.Thread(target=load_flow_fields, args=(files,), daemon=True)
    flow_thread.start()

    # Start the main loop
    main_thread = threading.Thread(target=main_loop)
    time.sleep(0.5)  # Ensure flow_thread is ready before starting main_thread
    print("Starting main loop...")
    # Start the main thread
    main_thread.start()

    time.sleep(5)
    # Start the training pipeline thread
    training_thread = threading.Thread(target=training_pipeline, args=(image_buffer,))
    training_thread.start()

    # Optionally join threads to wait for their completion
    # Note: main_loop runs indefinitely, so join() may block forever.
    try:
        # Wait for the flow thread to finish
        flow_thread.join()
        # Wait for the training thread to finish
        training_thread.join()
        # Joining main_thread is optional if main_loop is infinite.
        main_thread.join()
    except KeyboardInterrupt:
        print("Interrupt received, stopping threads.")
