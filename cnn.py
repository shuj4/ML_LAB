import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Set the parameters for plotting
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')
# Define the kernel
kernel = tf.constant([[-1, -1, -1],
 [-1, 8, -1],
 [-1, -1, -1]], dtype=tf.float32)
# Load the image
image = tf.io.read_file('ml.jpg')
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[500, 500])
# Plot the original image
plt.figure(figsize=(5, 5))
plt.imshow(tf.squeeze(image).numpy(), cmap='gray')
plt.axis('off')
plt.title('Original Gray Scale Image')
plt.show()
# Reformat the image and kernel
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
# Perform convolution
conv_fn = tf.nn.conv2d
image_filter = conv_fn(input=image, filters=kernel, strides=1, padding='SAME')
# Apply ReLU activation
relu_fn = tf.nn.relu
image_detect = relu_fn(image_filter)
# Apply max pooling
pooling_window_shape = (2, 2)
pooling_strides = (2, 2)
image_condense = tf.nn.max_pool2d(input=image_detect,
 ksize=pooling_window_shape,
 strides=pooling_strides,
 padding='SAME')
# Plot the results
plt.figure(figsize=(15, 5))
# Plot the convolved image
plt.subplot(1, 3, 1)
plt.imshow(tf.squeeze(image_filter).numpy(), cmap='gray')
plt.axis('off')
plt.title('Convolution')
# Plot the activated image
plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(image_detect).numpy(), cmap='gray')
plt.axis('off')
plt.title('Activation')
# Plot the pooled image
plt.subplot(1, 3, 3)
plt.imshow(tf.squeeze(image_condense).numpy(), cmap='gray')
plt.axis('off')
plt.title('Pooling')
plt.show()