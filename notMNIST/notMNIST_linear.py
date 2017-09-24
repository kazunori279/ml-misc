
# retrieving training file names
import re
from google.cloud import storage

GCS_BUCKET = "gcp-samples2-misc"
GCS_DATA_DIR = "notMNIST_large_test"

def list_files_and_labels(): 
  file_list = []
  label_list = []
  gcs_client = storage.Client()
  gcs_bucket = gcs_client.get_bucket(GCS_BUCKET)
  gcs_iterator = gcs_bucket.list_blobs(prefix=GCS_DATA_DIR)
  file_pattern = r".*%2F(.)%2F(.*\.png)"
  code_A = ord("A".decode("utf-8")[0])
  for blob in gcs_iterator:
    match = re.match(file_pattern, blob.path)
    if match:
      label = match.group(1)
      label_index = ord(label.decode("utf-8")[0]) - code_A
      filename = match.group(2)
      file_list.append("gs://" + GCS_BUCKET + "/" + GCS_DATA_DIR + "/" + label + "/" + filename)
      label_list.append(label_index) 
  return (file_list, label_list)

# create Dataset
import tensorflow as tf
from tensorflow.python import debug as tf_debug

def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  return image_decoded, label

def create_dataset():
  files, labels = list_files_and_labels()
  files_const = tf.constant(files)
  labels_const = tf.one_hot(tf.constant(labels), depth=10)

  dataset = tf.contrib.data.Dataset.from_tensor_slices((files_const, labels_const))
  dataset = dataset.map(_parse_function)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(2)
  dataset = dataset.repeat(1000)
  return dataset

dataset = create_dataset()
iterator = dataset.make_one_shot_iterator()
next_data, next_label = iterator.get_next()

sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session())

flat_data = tf.div(tf.to_float(tf.reshape(next_data, [2, -1])), 255.0)
sess.run(flat_data) 
