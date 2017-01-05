import matplotlib.pyplot as plt
import numpy as np
import struct

################################################################################
#  The following code to read MNIST data is copied from Zellyn:
#    https://github.com/zellyn/deeplearning-class-2011/tree/master/ufldf/starter
################################################################################

def load_images(filename):
  with open(filename, 'rb') as f:
    magic = struct.unpack('>i', f.read(4))[0]
    assert magic == 2051, ("Bad magic number(%s) in filename '%s'" % (magic, filename))
    num_images, num_rows, num_cols = struct.unpack('>3i', f.read(12))
    num_bytes = num_images * num_rows * num_cols
    images = np.fromstring(f.read(), dtype='uint8')
  assert images.size == num_bytes, 'Mismatch in dimensions vs data size'
  images = images.reshape([num_cols, num_rows, num_images], order='F')
  images = images.swapaxes(0,1)

  # Reshape to #pixels x #examples
  images = images.reshape([num_cols*num_rows, num_images], order='F')
  # Convert to double and rescale to [0,1]
  images = images / 255.0
  return images

def load_images_row_major(filename):
  """
  Load images into a M x N array where N is the number of images and
  M = P*P is the number of pixels in the image. The image dimension is P x P.
  That is, each column contains an image vector that is the raveled image in
  row-major ordering.
  """
  with open(filename, 'rb') as f:
    magic = struct.unpack('>i', f.read(4))[0]
    assert magic == 2051, ("Bad magic number(%s) in filename '%s'" % (magic, filename))
    num_images, num_rows, num_cols = struct.unpack('>3i', f.read(12))
    num_bytes = num_images * num_rows * num_cols
    images = np.fromstring(f.read(), dtype='uint8')
  assert images.size == num_bytes, 'Mismatch in dimensions vs data size'
  images = images.reshape([num_cols, num_rows, num_images], order='F')
  images = images.swapaxes(0,1)

  # Reshape to #pixels x #examples
  images = images.reshape([num_cols*num_rows, num_images], order='C')
  # Convert to double and rescale to [0,1]
  images = images / 255.0
  return images

def load_images_3d(filename):
  """
  Load images into a image_height x image_width x N 3D array where N is the number of images,
  image_height is the height in pixels and image_width is the width in pixels. 

  Note that each image is a greyscale 2D image of size image_height x image_width.

  Return the 3D array of images.
  """
  with open(filename, 'rb') as f:
    magic = struct.unpack('>i', f.read(4))[0]
    assert magic == 2051, ("Bad magic number(%s) in filename '%s'" % (magic, filename))
    num_images, num_rows, num_cols = struct.unpack('>3i', f.read(12))
    num_bytes = num_images * num_rows * num_cols
    images = np.fromstring(f.read(), dtype='uint8')
  assert images.size == num_bytes, 'Mismatch in dimensions vs data size'
  images = images.reshape([num_cols, num_rows, num_images], order='F')
  images = images.swapaxes(0,2)
  
  # Convert to double and rescale to [0,1]
  images = images / 255.0
  images = images.copy('C') # Change to C ordering for the backing array.
  return images

def load_labels(filename):
  with open(filename, 'rb') as f:
    magic = struct.unpack('>i', f.read(4))[0]
    assert magic == 2049, ("Bad magic number(%s) in filename '%s'" % (magic, filename))
    num_labels = struct.unpack('>i', f.read(4))[0]
    labels = np.fromstring(f.read(), dtype='uint8')
  assert labels.size == num_labels, 'Mismatch in label count'
  return labels

def save_as_figure(arr, filepath="output/frame.png"):
  arr = arr - np.mean(arr)
  L, M = arr.shape
  sz = np.sqrt(L)
  buf = 1

  # Figure out pleasant grid dimensions
  if M == np.floor(np.sqrt(M))**2:
    n = m = np.sqrt(M)
  else:
    n = np.ceil(np.sqrt(M))
    while (M%n) and n < 1.2*np.sqrt(M):
      n += 1
    m = np.ceil(M/n)

  array = np.zeros([buf+m*(sz+buf), buf+n*(sz+buf)])

  k = 0
  for i in range(0, int(m)):
    for j in range(0, int(n)):
      if k>=M:
        continue
      cmax = np.max((arr[:,k]))
      cmin = np.min((arr[:,k]))
      r = buf+i*(sz+buf)
      c = buf+j*(sz+buf)
      array[r:r+sz, c:c+sz] = (arr[:,k].reshape([sz,sz], order='F')-cmin) / (cmax-cmin)
      k = k + 1
#  plt.imshow(array, interpolation='nearest', cmap=plt.cm.gray)
  plt.imshow(array, cmap=plt.cm.gray)
  plt.savefig(filepath)
  print("Saving to ", filepath)
