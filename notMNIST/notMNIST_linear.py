
# retrieving training file names
import re
from google.cloud import storage

GCS_BUCKET = "gcp-samples2-misc"
# GCS_DATA_DIR = "notMNIST_large_test"
GCS_DATA_DIR = "notMNIST_small"
#GCS_DATA_DIR = "notMNIST_large"
TRAIN_STEPS = 10
TRAIN_BATCH_SIZE = 100

def list_files_and_labels(): 

  # get all filenames and labels under the dir
  file_list = []
  label_list = []
  gcs_client = storage.Client()
  gcs_bucket = gcs_client.get_bucket(GCS_BUCKET)
  gcs_iterator = gcs_bucket.list_blobs(prefix=GCS_DATA_DIR)
  file_pattern = r".*%2F(.)%2F(.*\.png)"
  code_A = ord("A".decode("utf-8")[0])

  # build a list of filepaths
  for blob in gcs_iterator:
    match = re.match(file_pattern, blob.path)
    if match:
      label = match.group(1)
      label_index = ord(label.decode("utf-8")[0]) - code_A
      filename = match.group(2).replace("%3D", "=")
      file_list.append("gs://" + GCS_BUCKET + "/" + GCS_DATA_DIR + "/" + label + "/" + filename)
      label_list.append(label_index) 

  # shuffle the list
  import random
  shuffled_files = []
  shuffled_labels = []
  shuffled_index = range(len(file_list)) 
  random.shuffle(shuffled_index)
  for i in shuffled_index: 
    shuffled_files.append(file_list[i])
    shuffled_labels.append(label_list[i])
  print("found " + str(len(file_list)) + " files.")
  return (shuffled_files, shuffled_labels)

# create Dataset
import tensorflow as tf
from tensorflow.contrib.data import Dataset as Dataset
from tensorflow.python import debug as tf_debug

def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  return image_decoded, label

def create_dataset(batch_size):
  files, labels = list_files_and_labels()
  files_const = tf.constant(files)
  labels_const = tf.one_hot(tf.constant(labels), depth=10)

  dataset = Dataset.from_tensor_slices((files_const, labels_const))
  dataset = dataset.interleave(
    lambda filename, label: Dataset.from_tensors((filename, label)).map(_parse_function, num_threads=1), cycle_length=10)
#  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(batch_size)
  return dataset

# define model
def model_fn(features, labels, mode):

  # layers
  input_layer = tf.div(tf.to_float(tf.reshape(features, [-1, 784])), 255.0)
  fc0_layer = tf.layers.dense(inputs=input_layer, units=10240, activation=tf.nn.relu)
  fc1_layer = tf.layers.dense(inputs=fc0_layer, units=10240, activation=tf.nn.relu)
  fc2_layer = tf.layers.dense(inputs=fc1_layer, units=10240, activation=tf.nn.relu)
  fc3_layer = tf.layers.dense(inputs=fc2_layer, units=10240, activation=tf.nn.relu)
  fc4_layer = tf.layers.dense(inputs=fc3_layer, units=10240, activation=tf.nn.relu)
  logits_layer = tf.layers.dense(inputs=fc4_layer, units=10) 

  # prediction
  predictions = {
    "classes": tf.argmax(input=logits_layer, axis=1),
    "probabilities": tf.nn.softmax(logits_layer, name="softmax_tensor")
  } 
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # loss and optimizer
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits_layer)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
  if mode == tf.estimator.ModeKeys.TRAIN:
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # evaluation
  eval_metric_ops = { 
    "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# input_fn for training 
train_dataset = None
eval_dataset = None
def train_input_fn():
  global train_dataset
  iterator = train_dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels 

# input_fn for eval
def eval_input_fn():
  global eval_dataset
  iterator = eval_dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels 

# main
def main(argv):
 
  # logging
  tf.logging.set_verbosity(tf.logging.INFO)
  log_hook = tf.train.LoggingTensorHook(tensors={"probabilities": "softmax_tensor"}, every_n_iter=1)
  # debug_hook = tf_debug.LocalCLIDebugHook()

  # read dataset 
  global train_dataset
  print("\ncreating train dataset...")
  train_dataset = create_dataset(TRAIN_BATCH_SIZE)

  # train
  print("\ntraining...")
  estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="/tmp/model")
  estimator.train(input_fn=train_input_fn, steps=TRAIN_STEPS, hooks=[log_hook])
  quit()

  # read dataset 
  global eval_dataset
  print("\ncreating eval dataset...")
  eval_dataset = create_dataset(1)

  # evaluate
  print("\nevaluating...")
  eval_resuts = estimator.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()   
