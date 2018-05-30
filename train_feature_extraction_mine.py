import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time

nb_classes = 43
epochs = 10
batch_size = 128

# Load traffic signs data.
with open('./train.p', mode='rb') as f:
    data = pickle.load(f)

# Split data into training and validation sets.
x_train, x_val , y_train , y_val = train_test_split(data['features'], data['labels'],test_size=0.33, random_state=0)

# Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(x, (227,227))

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8_W = tf.Variable(tf.truncated_normal(shape, stddev = 1e-2))
fc8_B = tf.Variable(tf.zeros(nb_classes))
logit = tf.nn.xw_plus_b(fc7,fc8_W,fc8_B)


# Define loss, training, accuracy operations.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=y)
loss_op = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op, var_list=[fc8_W, fc8_B])
init_op = tf.global_variables_initializer()

preds = tf.arg_max(logit, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))

# Train and evaluate the feature extraction model.
def eval_on_data(x, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, x.shape[0], batch_size):
        end = offset + batch_size
        X_batch = x[offset:end]
        y_batch = y[offset:end]
        
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: X_batch, y: y_batch})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])
    
    return total_loss/x.shape[0], total_acc/x.shape[0]

with tf.Session() as sess:
    sess.run(init_op)
    
    for i in range(epochs):
        # training
        x_train, y_train = shuffle(x_train, y_train)
        t0 = time.time()
        for offset in range(0, x_train.shape[0], batch_size):
            end = offset + batch_size
            sess.run(train_op, feed_dict={x: x_train[offset:end], y: y_train[offset:end]})
        
        val_loss, val_acc = eval_on_data(x_val, y_val, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")
