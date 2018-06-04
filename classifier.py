import tensorflow as tf
import yaml
import numpy as np
import re
import time

config = yaml.safe_load(open("config.yml"))

model_path      = config['training']['storage_path'] + "/retrained_graph.pb"
labels_path     = config['training']['storage_path'] + "/retrained_labels.txt"
output_path     = config['training']['storage_path'] + "/classified"
input_size      = int(re.search('[0-9]+$', config['training']['architecture']).group())
input_mean      = 128
input_std       = 128
patch_width     = config['training']['patch_width']
patch_height    = config['training']['patch_height']

def load_graph(model_path):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_path, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name):
  input_name = "file_reader"
  output_name = "normalized"
  width = input_size
  height = input_size
  num_channels = 3
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  # resized = tf.image.resize_bilinear(dims_expander, [input_size, input_size])
  normalized = tf.divide(tf.subtract(dims_expander, [input_mean]), [input_std])
  patches = tf.extract_image_patches(normalized,
       ksizes=[1, patch_height, patch_width, 1],
       strides=[1, patch_height/4, patch_width/4, 1],
       rates=[1,1,1,1],
       padding="VALID")
  patches_shape = tf.shape(patches)
  patches = tf.reshape(patches, [-1, patch_height, patch_width, num_channels])
  patches = tf.image.resize_images(patches, [height, width])
  patches = tf.reshape(patches, [-1, height, width, num_channels])
  sess = tf.Session()
  return sess.run([patches, patches_shape])

def load_labels():
   label = []
   proto_as_ascii_lines = tf.gfile.GFile(labels_path).readlines()
   for l in proto_as_ascii_lines:
     label.append(l.rstrip())
   return label

def classify(input_path):
    image_batch, batch_shape = read_tensor_from_image_file(input_path)

    graph = load_graph(model_path)
    input_operation = graph.get_operation_by_name("import/Placeholder")
    output_operation = graph.get_operation_by_name("import/final_result")

    positive_index = load_labels().index("positive")

    with tf.Session(graph=graph) as sess:
      start = time.time()
      results = []
      for img in image_batch:
          results.append(
            sess.run(output_operation.outputs[0], {input_operation.outputs[0]: [img]})[0]
          )
      end = time.time()

    h_count = batch_shape[2]

    results = [[
        p[positive_index],
        int((i % h_count) * patch_width / 4 + patch_width / 2),
        int(int(i / h_count) * patch_height / 4 + patch_height / 2)
    ] for i, p in enumerate(results)]

    results.sort(key=lambda tup: tup[0], reverse=True)

    print('\nEvaluation time: {:.3f}s\n'.format(end-start))

    return results
