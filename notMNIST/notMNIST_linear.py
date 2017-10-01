
# retrieving training file names
import re
from google.cloud import storage

GCS_BUCKET = "gcp-samples2-misc"
# GCS_DATA_DIR = "notMNIST_large_test"
GCS_DATA_DIR = "notMNIST_small"
#GCS_DATA_DIR = "notMNIST_large"
TRAIN_STEPS = 100
TRAIN_BATCH_SIZE = 10

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
      filename = match.group(2).replace("%3D", "=")
      file_list.append("gs://" + GCS_BUCKET + "/" + GCS_DATA_DIR + "/" + label + "/" + filename)
      label_list.append(label_index) 
  print("found " + str(len(file_list)) + " files.")
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
#  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(TRAIN_BATCH_SIZE)
  dataset = dataset.repeat(TRAIN_STEPS)
  return dataset

# define model
def model_fn(features, labels, mode):

  # layers
  input_layer = tf.div(tf.to_float(tf.reshape(features, [-1, 784])), 255.0)
  fc0_layer = tf.layers.dense(inputs=input_layer, units=1024, activation=tf.nn.relu)
  logits_layer = tf.layers.dense(inputs=fc0_layer, units=10) 

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
  eval_metrics_ops = { 
    "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)

# input_fn for training 
dataset = None
def train_input_fn():
  global dataset
  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels 

# input_fn for eval
def eval_input_fn():
  global dataset
  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels 

# main
def main(argv):
 
  # logging
  tf.logging.set_verbosity(tf.logging.INFO)
  log_hook = tf.train.LoggingTensorHook(tensors={"probabilities": "softmax_tensor"}, every_n_iter=50)
  # debug_hook = tf_debug.LocalCLIDebugHook()

  # read dataset 
  global dataset
  print("\ncreating dataset...")
  dataset = create_dataset()

  # train
  print("\ntraining...")
  estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="/tmp/model")
  estimator.train(input_fn=train_input_fn, steps=TRAIN_STEPS, hooks=[log_hook])
  quit()

  # evaluate
  print("\nevaluating...")
  eval_resuts = estimator.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()   
