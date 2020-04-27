from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time
import joblib
import math
import numpy
from six.moves import urllib
from six.moves import xrange  
from PIL import Image
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
FLAGS = None
IMAGE_HEIGHT = 128
IMAGE_WEITH = 128
NUM_CHANNELS = 1
NUM_LABELS = 4
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
EVAL_FREQUENCY = 10  # Number of steps between evaluations.

def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_HEIGHT, IMAGE_WEITH, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  Confusion_matrix=sk_confusion_matrix(numpy.argmax(predictions, 1).tolist(), labels.tolist())
  print('Confusion_matrix:')
  print(Confusion_matrix)  

  Se1 = Confusion_matrix[1,1]+Confusion_matrix[2,2]+Confusion_matrix[3,3]
  Se2 = Confusion_matrix[1,1]+Confusion_matrix[1,0]+Confusion_matrix[1,2]+Confusion_matrix[1,3]+Confusion_matrix[2,2]+Confusion_matrix[2,0]+Confusion_matrix[2,1]+Confusion_matrix[2,3]+Confusion_matrix[3,3]+Confusion_matrix[3,0]+Confusion_matrix[3,1]+Confusion_matrix[3,2]
  Se = Se1/Se2
  Sp = Confusion_matrix[0,0]/(Confusion_matrix[0,0]+Confusion_matrix[0,1]+Confusion_matrix[0,2]+Confusion_matrix[0,3]) 
  Acc = (Se+Sp)*100/2

  target_names = ['class 0', 'class 1', 'class 2', 'class 3']

  print()
  accuracy = 100.0-(100.0 *numpy.sum(numpy.argmax(predictions, 1) == labels)/predictions.shape[0])
  
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0]), Acc

def GroupNorm(x, G, eps=1e-05):
    # x: input features with shape [N,H,W,C]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
  N, H, W, C = x.shape
 # N = BATCH_SIZE
  gamma = tf.ones([1, 1, 1, C])
  beta = tf.zeros([1, 1, 1, C])
  x = tf.reshape(x, [-1, G, H*W, C // G])
#
  mean = tf.reduce_mean(x, axis=[2,3], keep_dims=True)

  var = tf.subtract(x,mean)
  var = var*var
  var = tf.reduce_mean(var, axis=[2,3], keep_dims=True)
 
  x = tf.subtract(x,mean) / tf.sqrt(var + eps)
  x = tf.reshape(x, [-1, H, W, C])
  return x * gamma + beta

class ResBlock(object):

  def __init__(self, stride_num=1, downsample=False):
    self.conv1_weights = tf.Variable(
      tf.truncated_normal([1, 1, 64, 64],  # 1x1 filter, depth 64.
                          stddev=0.1,
                          seed=SEED, dtype=data_type()))  

    self.conv2_weights = tf.Variable(
      tf.truncated_normal([3, 3, 64, 64],  # 3x3 filter, depth 64.
                          stddev=0.1,
                          seed=SEED, dtype=data_type()))
    self.conv3_weights = tf.Variable(
      tf.truncated_normal([3, 3, 64, 64],  # 3x3 filter, depth 64.
                          stddev=0.1,
                          seed=SEED, dtype=data_type()))  
    self.stride_num = stride_num
    self.downsample = downsample

  def forward(self, data):
    with tf.name_scope('ResNet'):
    # shortcut = x
      shortcut = data
      # out = self.relu(self.norm1(x))
#      axis = list(range(len(data.get_shape()) - 1))
      with tf.name_scope('BN1'):
        out = GroupNorm(x=data, G=32)
        #mean, variance = tf.nn.moments(data, axis)
        #out = tf.nn.batch_normalization(data, mean, variance, 0, 1, 0.001)
      with tf.name_scope('relu1'):
        out = tf.nn.relu(out)
      #  if self.downsample is not None:
        #   shortcut = self.downsample(out)
      with tf.name_scope('downsample'):
        if self.downsample is True:
          shortcut = tf.nn.conv2d(out,
                                  self.conv1_weights,
                                  strides=[1, self.stride_num, self.stride_num, 1],
                                  padding='SAME')
        #  out = self.conv1(out)
      with tf.name_scope('conv1'):
        out = tf.nn.conv2d(out,
                          self.conv2_weights,
                          strides=[1, self.stride_num, self.stride_num, 1],
                          padding='SAME')
      #  out = self.droupout(out)
      #  out = self.norm2(out)
      with tf.name_scope('BN2'):
        out = GroupNorm(x=out, G=32)

      #  out = self.relu(out) 
      with tf.name_scope('relu2'):
        out = tf.nn.relu(out)   
      #  out = self.conv2(out)
      with tf.name_scope('conv2'):
        out = tf.nn.conv2d(out,
                          self.conv3_weights,
                          strides=[1, 1, 1, 1],
                          padding='SAME')
    return shortcut+out

class BRN(object):

  def __init__(self):
    self.ResNet_0_0 = ResBlock(2, True)
    self.ResNet_0_1 = ResBlock(2, True)
    self.ResNet_1_0 = ResBlock(2, True)
    self.ResNet_1_1 = ResBlock(2, True)
    self.ResNet_0 = ResBlock(1, False)
    self.ResNet_1 = ResBlock(1, False)
    self.ResNet_2 = ResBlock(1, False)
    self.ResNet_3 = ResBlock(1, False)
    self.ResNet_4 = ResBlock(1, False)
    self.ResNet_5 = ResBlock(1, False)
    self.ResNet_6 = ResBlock(1, False)
    self.ResNet_7 = ResBlock(1, False)
    self.ResNet_8 = ResBlock(1, False)
    self.ResNet_9 = ResBlock(1, False)
    self.ResNet_10 = ResBlock(1, False)
    self.ResNet_11 = ResBlock(1, False)
    self.ResNet_12 = ResBlock(1, False)
    self.ResNet_13 = ResBlock(1, False)
    self.ResNet_14 = ResBlock(1, False)
    self.ResNet_15 = ResBlock(1, False)
    self.ResNet_16 = ResBlock(1, False)
    self.ResNet_17 = ResBlock(1, False)
    self.ResNet_18 = ResBlock(1, False)
    self.ResNet_19 = ResBlock(1, False)
    self.ResNet_20 = ResBlock(1, False)
    self.ResNet_21 = ResBlock(1, False)
    self.conv1_weights = tf.Variable(
      tf.truncated_normal([3, 3, NUM_CHANNELS, 64],  
                          stddev=0.1,
                          seed=SEED, dtype=data_type()))  
    self.conv2_weights = tf.Variable(
      tf.truncated_normal([3, 3, NUM_CHANNELS, 64],  
                          stddev=0.1,
                          seed=SEED, dtype=data_type()))  
    self.fc_weights = tf.Variable(tf.truncated_normal([64, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))

    self.fc_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=data_type()))

  def forward(self, stft, mfcc):
    stft = tf.convert_to_tensor(stft)
    mfcc = tf.convert_to_tensor(mfcc)
    with tf.name_scope('BRNcon1'):
      out_s = tf.nn.conv2d(stft,
                        self.conv1_weights,
                        strides= [1, 1, 1, 1],
                        padding='VALID')
    with tf.name_scope('resnetblocks_1'):
      out_s = self.ResNet_0_0.forward(out_s)
    with tf.name_scope('resnetblocks_2'):
      out_s = self.ResNet_0_1.forward(out_s)
    with tf.name_scope('resnetblocks_3'):
      out_s = self.ResNet_0.forward(out_s)
    with tf.name_scope('resnetblocks_4'):
      out_s = self.ResNet_2.forward(out_s)
    with tf.name_scope('resnetblocks_5'):
      out_s = self.ResNet_4.forward(out_s)
    with tf.name_scope('resnetblocks_6'):
      out_s = self.ResNet_6.forward(out_s)
    with tf.name_scope('resnetblocks_7'):
      out_s = self.ResNet_8.forward(out_s)
    with tf.name_scope('resnetblocks_8'):
      out_s = self.ResNet_10.forward(out_s)
    with tf.name_scope('resnetblocks_9'):
      out_s = self.ResNet_12.forward(out_s)
    with tf.name_scope('resnetblocks_10'):
      out_s = self.ResNet_14.forward(out_s)
    with tf.name_scope('resnetblocks_11'):
      out_s = self.ResNet_16.forward(out_s)
    with tf.name_scope('resnetblocks_12'):
      out_s = self.ResNet_18.forward(out_s)
    with tf.name_scope('resnetblocks_13'):
      out_s = self.ResNet_20.forward(out_s)
    with tf.name_scope('brns_bn1'):
      out_s = GroupNorm(x=out_s, G=32)

    with tf.name_scope('brn_relu_1'):
      out_s = tf.nn.relu(out_s)
    with tf.name_scope('brn_pool1'):
      out_s = tf.nn.avg_pool(out_s,
                            ksize=[1,out_s.shape[2],out_s.shape[2],1],
                            strides=[1, 1, 1, 1],
                            padding='VALID')

    with tf.name_scope('BRNcon2'):
      out_m = tf.nn.conv2d(mfcc,
                        self.conv2_weights,
                        strides= [1, 1, 1, 1],
                        padding='VALID')
    with tf.name_scope('resnetblockm_1'):
      out_m = self.ResNet_1_0.forward(out_m)
    with tf.name_scope('resnetblockm_2'):
      out_m = self.ResNet_1_1.forward(out_m)
    with tf.name_scope('resnetblockm_3'):
      out_m = self.ResNet_1.forward(out_m)
    with tf.name_scope('resnetblockm_4'):
      out_m = self.ResNet_3.forward(out_m)  
    with tf.name_scope('resnetblockm_5'):
      out_m = self.ResNet_5.forward(out_m)
    with tf.name_scope('resnetblockm_6'):
      out_m = self.ResNet_7.forward(out_m)
    with tf.name_scope('resnetblockm_7'):
      out_m = self.ResNet_9.forward(out_m)
    with tf.name_scope('resnetblockm_8'):
      out_m = self.ResNet_11.forward(out_m)
    with tf.name_scope('resnetblockm_9'):
      out_m = self.ResNet_13.forward(out_m)
    with tf.name_scope('resnetblockm_10'):
      out_m = self.ResNet_15.forward(out_m)
    with tf.name_scope('resnetblockm_11'):
      out_m = self.ResNet_17.forward(out_m)
    with tf.name_scope('resnetblockm_12'):
      out_m = self.ResNet_19.forward(out_m)
    with tf.name_scope('resnetblockm_13'):
      out_m = self.ResNet_21.forward(out_m)
    with tf.name_scope('brnm_bn1'):
      out_m = GroupNorm(x=out_m, G=32)

    with tf.name_scope('brn_relu_2'):
      out_m = tf.nn.relu(out_m)
    with tf.name_scope('brn_pool2'):
      out_m = tf.nn.avg_pool(out_m,
                            ksize=[1,out_m.shape[2],out_m.shape[2],1],
                            strides=[1, 1, 1, 1],
                            padding='VALID')
    with tf.name_scope('maumul'):
    
      out = tf.multiply(out_s,out_m)
    with tf.name_scope('fc'):

      out_shape = out.get_shape().as_list()
      reshape = tf.reshape(
          out,
          [-1, out_shape[1] * out_shape[2] * out_shape[3]])    
      out = tf.add(tf.matmul(reshape, self.fc_weights), self.fc_biases, name="logits_")

    return out

def main(_):

  def loss_function(weight, logits, labels):
    labels = tf.one_hot(labels,4)
    labels = tf.cast(labels, tf.float32)
    first = tf.reduce_sum(tf.multiply(-labels, logits),1)
    second_0 = tf.add(tf.exp(logits[:,0]),tf.exp(logits[:,1]))
    second_1 = tf.add(tf.exp(logits[:,2]),tf.exp(logits[:,3]))
    log = tf.log(tf.add(second_1,second_0))
    weight = tf.transpose(tf.reduce_sum(tf.multiply(labels, weight),1))
    output = tf.multiply(weight,tf.add(first,log))

    return output

  def normalize(stft):
    stft_1 = numpy.empty([stft.shape[0],128,128])
    stft_2 = numpy.empty([stft_1.shape[0],stft_1.shape[1],stft_1.shape[2],1])
    for i in range(stft_1.shape[0]):
      image = Image.fromarray(stft[i,:,:])
      image = image.resize([128,128])
      stft_1[i,:,:] = numpy.array(image)

      min = numpy.min(stft_1[i,:,:])
      max = numpy.max(stft_1[i,:,:])
      stft_1[i,:,:] = (stft_1[i,:,:]-min)/(max-min)
      stft_2[i,:,:,:] = stft_1[i,:,:].reshape((stft_1.shape[1],stft_1.shape[2],1))
    return stft_2  

  if FLAGS.self_test:
    
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # Get the data.
    
    stft_training, mfcc_training, labels_training = joblib.load(open(FLAGS.input, mode='rb'))
    stft_test, mfcc_test, labels_test = joblib.load(open(FLAGS.test, mode='rb'))

    stft_test = numpy.array(stft_test)
    mfcc_test = numpy.array(mfcc_test)
    labels_test = numpy.array(labels_test)
    stft_test = normalize(stft_test)
    mfcc_test = normalize(mfcc_test)

    stft_training = numpy.array(stft_training)
    mfcc_training = numpy.array(mfcc_training)
    labels_training = numpy.array(labels_training)
    stft_training = normalize(stft_training)
    mfcc_training = normalize(mfcc_training)

    stft_shape = stft_training.shape
    stft_shape = (None, stft_shape[1], stft_shape[2], 1)

    mfcc_shape = mfcc_training.shape
    mfcc_shape = (None, mfcc_shape[1], mfcc_shape[2], 1)

    labels_shape = labels_training.shape
    labels_shape = (None)

    stft_placeholder = tf.placeholder(stft_training.dtype, stft_shape)
    labels_placeholder = tf.placeholder(labels_training.dtype, labels_shape)
    mfcc_placeholder = tf.placeholder(mfcc_training.dtype, mfcc_shape)
    
    dataset_training = tf.data.Dataset.from_tensor_slices((stft_placeholder, mfcc_placeholder, labels_placeholder))
    dataset_training  = dataset_training.apply(
        tf.data.experimental.shuffle_and_repeat(len(stft_training), None))  
    dataset_training  = dataset_training.batch(BATCH_SIZE)
    dataset_training  = dataset_training.prefetch(1)
    iterator_training = dataset_training.make_initializable_iterator()
    next_element_training = iterator_training.get_next()
    num_epochs = FLAGS.epochs

  train_size = labels_training.shape[0]


  stft_holder = tf.placeholder(
        name="stft_holder",
        dtype=data_type(),
        shape=(None, IMAGE_HEIGHT, IMAGE_WEITH, NUM_CHANNELS))
  mfcc_holder = tf.placeholder(
        name="mfcc_holder",
        dtype=data_type(),
        shape=(None, IMAGE_HEIGHT, IMAGE_WEITH, NUM_CHANNELS))
  labels = tf.placeholder(tf.int64, shape=(None,))

  with tf.name_scope('test_input'):
    stft_t = tf.placeholder(
        data_type(),
        shape=(None, IMAGE_HEIGHT, IMAGE_WEITH, NUM_CHANNELS))
    mfcc_t = tf.placeholder(
        data_type(),
        shape=(None, IMAGE_HEIGHT, IMAGE_WEITH, NUM_CHANNELS))

  model = BRN()
  
  logits = model.forward(stft_holder, mfcc_holder)
  out1 = tf.identity(logits,name="out1")

  try:
    scalar_summary = tf.scalar_summary
    SummaryWrite = tf.train.SummaryWrite
    merge_summary = tf.merge_summary
  except:
    scalar_summary = tf.summary.scalar
    SummaryWrite = tf.summary.FileWriter
    merge_summary = tf.summary.merge
  with tf.name_scope('loss'):
    weights = [1.0, 1.7, 4.1, 5.7]
    mid = loss_function(weights, logits=logits, labels=labels)
#    mid = tf.nn.sparse_softmax_cross_entropy_with_logits(
#       labels=labels, logits=logits)

    loss = tf.reduce_sum(mid)
    
    loss_summary = scalar_summary('loss', loss)

    
    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(model.conv1_weights) + tf.nn.l2_loss(model.conv2_weights) +
                    tf.nn.l2_loss(model.fc_weights) + tf.nn.l2_loss(model.fc_biases))
    # Add the regularization term to the loss.
    loss += 0.02 * regularizers

    batch = tf.Variable(0, dtype=data_type())
  # Use simple momentum for the optimization.
  with tf.name_scope('train'):

    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)
  eval_prediction = tf.nn.softmax(model.forward(stft_t, mfcc_t))

  # Create a local session to run the training.
  start_time = time.time()

  def eval_in_batches(stft_data, mfcc_data, sess, type):
    """Get all predictions for a dataset by running it in small batches."""
    size = stft_data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        if type == 'train':
          predictions[begin:end, :] = sess.run(
              train_prediction,
              feed_dict={stft_holder: stft_data[begin:end, ...], mfcc_holder: mfcc_data[begin:end, ...]})
        else: 
          predictions[begin:end, :] = sess.run(
              eval_prediction,
              feed_dict={stft_t: stft_data[begin:end, ...], mfcc_t: mfcc_data[begin:end, ...]})
      else:
        if type == 'train':
          batch_predictions = sess.run(
              train_prediction,
              feed_dict={stft_holder: stft_data[-EVAL_BATCH_SIZE:, ...], mfcc_holder: mfcc_data[-EVAL_BATCH_SIZE:, ...]})
        else:
           batch_predictions = sess.run(
              eval_prediction,
              feed_dict={stft_t: stft_data[-EVAL_BATCH_SIZE:, ...], mfcc_t: mfcc_data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  saver = tf.train.Saver()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  

  with tf.Session(config=config) as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.global_variables_initializer().run()

    merged = tf.summary.merge_all()
    writer = SummaryWrite(FLAGS.logs + 'train', sess.graph)
    print('Initialized!')
    sess.run(iterator_training.initializer, feed_dict={stft_placeholder:stft_training,
                      mfcc_placeholder:mfcc_training,
                      labels_placeholder:labels_training})

    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):

      batch_stft, batch_mfcc, batch_labels = sess.run(next_element_training)
  
      feed_dict = {stft_holder: batch_stft,
                   mfcc_holder: batch_mfcc,
                   labels: batch_labels}
      # Run the optimizer to update weights.

      sess.run(optimizer, feed_dict=feed_dict)
      # print some extra information once reach the evaluation frequency
      if step % EVAL_FREQUENCY == 0:
        # fetch some extra nodes' data
        summary, l = sess.run([merged, loss],
                                      feed_dict=feed_dict)
        writer.add_summary(summary, step)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        rate, acc = error_rate(eval_in_batches(stft_training, mfcc_training, sess, 'train'), labels_training)
        acc_summary = scalar_summary('accuracy', acc)
        print('Step %d (epoch %.2f), Minibatch loss: %.3f, Minibatch error: %.1f%%, Accuracy:%.4f' %
              (step, float(step) * BATCH_SIZE / train_size,
              l,rate, acc))

        
    # Finally print the result!
        sys.stdout.flush()
        test_error, test_acc = error_rate(eval_in_batches(stft_test, mfcc_test, sess, 'test'), labels_test)
        print('Testset error: %.1f%%, Accuracy:%.4f' % (test_error, test_acc))
#        mfcc_ = tf.placeholder(name="mfcc_", dtype=tf.float32, shape=(1, IMAGE_HEIGHT, IMAGE_WEITH, NUM_CHANNELS))
#        stft_ = tf.placeholder(name="logits_", dtype=tf.float32, shape=(1, IMAGE_HEIGHT, IMAGE_WEITH, NUM_CHANNELS))
    converter = tf.lite.TFLiteConverter.from_session(sess, [stft_holder,mfcc_holder], [out1])
    tflite_model = converter.convert()
    open("BRN11.tflite", "wb").write(tflite_model)
    
    saver.save(sess, './local_ckpt5')        
    writer.close()



if __name__ == '__main__':
#  dev = '/gpu:0'
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')
  parser.add_argument(
      '--input',
      default='wavelet_stft.p')
  parser.add_argument(
      '--test',
      default='wavelet_stft_test.p')  
  parser.add_argument(
      '--epochs',
      type=float,
      default=0.2)  
  parser.add_argument(
      '--logs',
      default='')  
  FLAGS, unparsed = parser.parse_known_args()
 # tf.enable_resource_variables()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
