import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import sys

# TODO: Load traffic signs data.
training_data_file = "./train.p"
with open(training_data_file, "rb") as tfile:
    train = pickle.load(tfile)

# TODO: Split data into training and validation sets.
XX_train, yy_train = train['features'], train['labels']
X_train, X_val, y_train, y_val = train_test_split(XX_train, yy_train, test_size = 0.05, random_state=832289)
X_train, y_train = shuffle(X_train, y_train)
X_val, y_val = shuffle(X_val, y_val)

# TODO: Define placeholders and resize operation.
gtsrb_nclasses = 43
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
x_resized = tf.image.resize_images(x, (227, 227))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, gtsrb_nclasses)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(x_resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], gtsrb_nclasses)
fc8W = tf.Variable(tf.random_normal(shape, stddev = 0.01))
fc8b = tf.Variable(tf.zeros(gtsrb_nclasses))
logits = tf.matmul(fc7, fc8W) + fc8b

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
LEARNING_RATE = 1e-3
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
training_op = optimizer.minimize(loss)

# TODO: Train and evaluate the feature extraction model.
BATCH_SIZE = 128
EPOCHS = 10
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def evaluate(X_data, y_data):
    sess = tf.get_default_session()
    tot_accuracy = 0
    for offset in range(0, len(X_data), BATCH_SIZE):
        X_b, y_b = X_data[offset: offset + BATCH_SIZE], y_data[offset: offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_op, feed_dict = {x: X_b, y: y_b})
        tot_accuracy += (accuracy * BATCH_SIZE)
    return tot_accuracy / len(X_data)

# Do the training

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Training...")
    sys.stdout.flush()
    print()
    sys.stdout.flush()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, len(X_train), BATCH_SIZE):
            X_b, y_b = X_train[offset: offset+BATCH_SIZE], y_train[offset: offset+BATCH_SIZE]
            sess.run(training_op, feed_dict = {x: X_b, y: y_b})
        val_accuracy = evaluate(X_val, y_val)
        print("EPOCH {}...".format(i+1))
        sys.stdout.flush()
        print("   -> Validation Accuracy = {:.3f}".format(val_accuracy))
        sys.stdout.flush()
        print()
        sys.stdout.flush()
