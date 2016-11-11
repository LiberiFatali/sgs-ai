# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7
"""Export inception model given existing training checkpoints.
"""

import os.path

# This is a placeholder for a Google-internal import.

import tensorflow as tf
import numpy as np
from tensorflow.contrib.session_bundle import exporter
from inception import inception_model


tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/inception_train',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('export_dir', '/tmp/inception_export',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('image_size', 299,
                            """Needs to provide same value as in training.""")
FLAGS = tf.app.flags.FLAGS


#NUM_CLASSES = 1000
#NUM_TOP_CLASSES = 5

#WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
#SYNSET_FILE = os.path.join(WORKING_DIR, 'imagenet_lsvrc_2015_synsets.txt')
#METADATA_FILE = os.path.join(WORKING_DIR, 'imagenet_metadata.txt')

NUM_CLASSES = 1017
#NUM_TOP_CLASSES = 5
NUM_TOP_CLASSES = 1018

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
SYNSET_FILE = os.path.join(WORKING_DIR, 'sgsnet_1017_synsets.txt')
METADATA_FILE = os.path.join(WORKING_DIR, 'sgsnet_1017_metadata.txt')

IMAGE_SIZE = 299


def export():
  # Create index->synset mapping
  synsets = []
  with open(SYNSET_FILE) as f:
    synsets = f.read().splitlines()
  # Create synset->metadata mapping
  texts = {}
  with open(METADATA_FILE) as f:
    for line in f.read().splitlines():
      parts = line.split('\t')
      assert len(parts) == 2
      texts[parts[0]] = parts[1]

  with tf.Graph().as_default():
    # Build inference model.
    # Please refer to Tensorflow inception model for details.

    # Input transformation.
    jpegs = tf.placeholder(tf.string)
    images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)

    # Run inference.
    #logits, _ = inception_model.inference(images, NUM_CLASSES + 1)
    logits, _ = inception_model.inference(images[0], NUM_CLASSES + 1)
    
    # To softmax probabilities
    probs = tf.nn.softmax(logits)
    #
    probs = tf.reduce_mean(probs, 0, keep_dims=True)

    # Transform output to topK result.
    #values, indices = tf.nn.top_k(logits, NUM_TOP_CLASSES)
    values, indices = tf.nn.top_k(probs, NUM_TOP_CLASSES)

    # Create a constant string Tensor where the i'th element is
    # the human readable class description for the i'th index.
    # Note that the 0th index is an unused background class
    # (see inception model definition code).
    class_descriptions = ['unused background']
    for s in synsets:
      class_descriptions.append(texts[s])
    class_tensor = tf.constant(class_descriptions)

    classes = tf.contrib.lookup.index_to_string(tf.to_int64(indices),
                                                mapping=class_tensor)

    # Restore variables from training checkpoint.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    with tf.Session() as sess:
      # Restore variables from training checkpoints.
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, global_step))
      else:
        print('No checkpoint file found at %s' % FLAGS.checkpoint_dir)
        return

      # Export inference model.
      init_op = tf.group(tf.initialize_all_tables(), name='init_op')
      model_exporter = exporter.Exporter(saver)
      signature = exporter.classification_signature(
          input_tensor=jpegs, classes_tensor=classes, scores_tensor=values)
      model_exporter.init(default_graph_signature=signature, init_op=init_op)
      model_exporter.export(FLAGS.export_dir, tf.constant(global_step), sess)
      print('Successfully exported model to %s' % FLAGS.export_dir)


def preprocess_image(image_buffer):
  """Preprocess JPEG encoded bytes to 3D float Tensor."""

  # Decode the string as an RGB JPEG.
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. In other words, the height
  # and width of image is unknown at compile-time.
  image = tf.image.decode_jpeg(image_buffer, channels=3)
  # After this point, all image pixels reside in [0,1)
  # until the very end, when they're rescaled to (-1, 1).  The various
  # adjust_* ops all require this range for dtype float.
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
  image_centered = tf.image.central_crop(image, central_fraction=0.875)
  image_centered = tf.image.resize_images(image_centered, IMAGE_SIZE, 
					IMAGE_SIZE, 
					method=tf.image.ResizeMethod.BILINEAR, 
					align_corners=False)
  # Resize the image to required height and width.  
  image_resized = tf.image.resize_images(image, IMAGE_SIZE, 
					IMAGE_SIZE, 
					method=tf.image.ResizeMethod.BILINEAR, 
					align_corners=False)
  # Resize shorter edge to IMAGE_SIZE, keep aspect ratio
  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_shorter_edge = tf.constant(IMAGE_SIZE)
  height_smaller_than_width = tf.less_equal(height, width)  
  new_height, new_width = tf.cond(
    height_smaller_than_width,
    lambda: (new_shorter_edge, (width / height) * new_shorter_edge),
    lambda: (new_shorter_edge, (height / width) * new_shorter_edge)
  )
  image_resized_keepar = tf.image.resize_images(image, new_height, 
					new_width, 
					method=tf.image.ResizeMethod.BILINEAR, 
					align_corners=False)
    # Compare new width and height
  newheight_smaller_than_newwidth = tf.less_equal(new_height, new_width)
    # Crop left/top view
  image_resized_keepar_lview = tf.image.crop_to_bounding_box(image_resized_keepar, 
							     0, 0, 
							     IMAGE_SIZE, 
							     IMAGE_SIZE)
    # Crop middle view
  offset_height, offset_width = tf.cond(
    newheight_smaller_than_newwidth,
    lambda: (tf.constant(0), tf.floordiv(tf.sub(new_width, IMAGE_SIZE), 2)),
    lambda: (tf.floordiv(tf.sub(new_width, IMAGE_SIZE), 2), tf.constant(0))
  )
  image_resized_keepar_cview = tf.image.crop_to_bounding_box(image_resized_keepar, 
							     offset_height, 
							     offset_width, 
							     IMAGE_SIZE, 
							     IMAGE_SIZE) 
    # Crop right/bottom view
  offset_height, offset_width = tf.cond(
    newheight_smaller_than_newwidth,
    lambda: (tf.constant(0), tf.sub(new_width, IMAGE_SIZE)),
    lambda: (tf.sub(new_height, IMAGE_SIZE), tf.constant(0))
  )
  image_resized_keepar_rview = tf.image.crop_to_bounding_box(image_resized_keepar, 
							     offset_height, 
							     offset_width, 
							     IMAGE_SIZE, 
							     IMAGE_SIZE)

  # Flip left-right the resized image
  image_fliplr = tf.image.flip_left_right(image_resized)
  image_centered_lr = tf.image.flip_left_right(image_centered)
  image_resized_keepar_lview_lr = tf.image.flip_left_right(image_resized_keepar_lview)
  image_resized_keepar_cview_lr = tf.image.flip_left_right(image_resized_keepar_cview)
  image_resized_keepar_rview_lr = tf.image.flip_left_right(image_resized_keepar_rview)
  # Flip up-down the resized image
  image_flipud = tf.image.flip_up_down(image_resized)  
  image_centered_ud = tf.image.flip_up_down(image_centered)
  image_resized_keepar_lview_ud = tf.image.flip_up_down(image_resized_keepar_lview)
  image_resized_keepar_cview_ud = tf.image.flip_up_down(image_resized_keepar_cview)
  image_resized_keepar_rview_ud = tf.image.flip_up_down(image_resized_keepar_rview)
  ## Rotate 90 the resized image
  #image_rot90 = tf.image.rot90(image_resized, k=1)
  ## Rotate 270 the resized image
  #image_rot270 = tf.image.rot90(image_resized, k=3)
  
  # Construct the output crops
  #image_crops = tf.pack([image_centered, image_resized, image_fliplr, 
			 #image_flipud, image_rot90, image_rot270]) 
  image_crops = tf.pack([image_centered, image_resized, image_fliplr, 
			 image_flipud, image_centered_lr, image_centered_ud, 
			 image_resized_keepar_lview, 
			 image_resized_keepar_lview_lr, 
			 image_resized_keepar_lview_ud, 
			 image_resized_keepar_cview, 
			 image_resized_keepar_cview_lr, 
			 image_resized_keepar_cview_ud, 
			 image_resized_keepar_rview, 
			 image_resized_keepar_rview_lr, 
			 image_resized_keepar_rview_ud])
  
  # Finally, rescale to [-1,1] instead of [0, 1)
  #image = tf.sub(image, 0.5)
  #image = tf.mul(image, 2.0)
  #return image 
  
  image_crops = tf.map_fn(lambda x: (x - 0.5) * 2.0, image_crops) 
  return image_crops


def main(unused_argv=None):
  export()


if __name__ == '__main__':
  tf.app.run()
