import os
import threading
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy
from PIL import Image, ImageDraw
import random

# Define the image size and background color
background_color = (255, 255, 255, 0)
gt_files = os.listdir('train_data/original')
os.makedirs('test/masks', exist_ok=True)
os.makedirs('test/damaged', exist_ok=True)

print(len(gt_files))
# Define a function to generate the image for a given title
def generate_image(start, end):
    for title in gt_files[start:end]:
        title, ext = title.split('.')
        background = Image.open(f'train_data/original/{title}.{ext}').convert("RGBA")
        image_size = background.size
        # Create a new image and drawing context
        image = Image.new('RGBA', image_size, background_color)
        draw = ImageDraw.Draw(image)

        # Generate random zig-zag scratches
        for i in range(200):
            # Define the scratch color as a random dark gray
            scratch_color = (
                random.randint(100, 130),
                random.randint(50, 80),
                random.randint(5, 35)
            )

            # Define the scratch width and length
            scratch_width = random.randint(2, 4)
            scratch_length = random.randint(50, 100)

            # Define the scratch position and angle
            x1, y1 = random.randint(0, image_size[0]), random.randint(0, image_size[1])
            angle = random.uniform(-0.5, 0.5)

            # Define the number of line segments for the scratch
            num_segments = random.randint(3, 6)

            # Compute the starting and ending points for each segment
            points = [(x1, y1)]
            for j in range(num_segments):
                dx = random.randint(-scratch_length // 2, scratch_length // 2)
                dy = random.randint(-scratch_width // 2, scratch_width // 2)
                x, y = points[-1]
                points.append((x + dx, y + dy))
            points.append((x1 + scratch_length, y1))

            # Draw the scratch as a series of connected lines
            draw.line(points, fill=scratch_color, width=scratch_width)

        # Save the generated texture
        image.save(f'texture{title}.png')
        image = image.convert('L')
        array = numpy.uint8(numpy.asarray(image) > 150)
        plt.imsave(f'test/masks/{title}.{ext}', array, cmap=cm.gray)

        foreground = Image.open(f'texture{title}.png')

        # Merge the two images using alpha blending
        merged_image = Image.alpha_composite(background, foreground).convert('RGB')

        # Save the merged image
        merged_image.save(f'test/damaged/{title}.{ext}')
        os.remove(f'texture{title}.png')

# Create a thread for each title
threads = []
for i in range(0, 4400, 880):
    t = threading.Thread(target=generate_image, args=(i, i + 880,))
    threads.append(t)

# Start the threads
for t in threads:
    t.start()

# Wait for all threads to finish
for t in threads:
    t.join()
