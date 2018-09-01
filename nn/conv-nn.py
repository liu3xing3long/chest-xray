
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os, random, time


# In[2]:


# Hyperparameters

batch_size = 8
learning_rate = 0.01
training_epochs = 30
display_step = 10

# To prevent overfitting
dropout = 0.75

summaries_dir = "./logs"


# In[3]:


# Dataset and iterator creation

in_dir = "../input/preprocessed-normalized/"
test_normal_dir = in_dir + "test/NORMAL"
test_pneumonia_dir = in_dir + "test/PNEUMONIA"
train_normal_dir = in_dir + "train/NORMAL"
train_pneumonia_dir = in_dir + "train/PNEUMONIA"
val_normal_dir = in_dir + "val/NORMAL"
val_pneumonia_dir = in_dir + "val/PNEUMONIA"

full_url = np.vectorize(lambda url,prev_url: prev_url+"/"+url)
test_normal_data = pd.DataFrame(full_url(np.array(os.listdir(test_normal_dir)),test_normal_dir), columns=["image_dir"])
test_pneumonia_data = pd.DataFrame(full_url(np.array(os.listdir(test_pneumonia_dir)),test_pneumonia_dir), columns=["image_dir"])
train_normal_data = pd.DataFrame(full_url(np.array(os.listdir(train_normal_dir)),train_normal_dir), columns=["image_dir"])
train_pneumonia_data = pd.DataFrame(full_url(np.array(os.listdir(train_pneumonia_dir)),train_pneumonia_dir), columns=["image_dir"])
val_normal_data = pd.DataFrame(full_url(np.array(os.listdir(val_normal_dir)),val_normal_dir), columns=["image_dir"])
val_pneumonia_data = pd.DataFrame(full_url(np.array(os.listdir(val_pneumonia_dir)),val_pneumonia_dir), columns=["image_dir"])

test_normal_data["class"] = "NORMAL"
test_pneumonia_data["class"] = "PNEUNOMIA"
train_normal_data["class"] = "NORMAL"
train_pneumonia_data["class"] = "PNEUNOMIA"
val_normal_data["class"] = "NORMAL"
val_pneumonia_data["class"] = "PNEUNOMIA"

test_data = test_normal_data.append(test_pneumonia_data)
train_data = train_normal_data.append(train_pneumonia_data)
val_data = val_normal_data.append(val_pneumonia_data)

# Total ammount of landmarks
n_classes = 2

with tf.device('/cpu:0'):

    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_decoded = tf.cast(image_decoded, tf.float32)
        image_decoded.set_shape((256, 256, 1))
        return image_decoded, label

    def _parse_rotate_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_decoded =  tf.contrib.image.rotate(image_decoded, random.uniform(-3,3))
        image_decoded = tf.cast(image_decoded, tf.float32)
        image_decoded.set_shape((256, 256, 1))
        return image_decoded, label

    train_data = tf.data.Dataset.from_tensor_slices(
        (train_data["image_dir"].values, 
         pd.get_dummies(train_data["class"]).values))
    train_data = train_data.shuffle(buffer_size=10000)

    # for a small batch size
    train_data = train_data.map(_parse_rotate_function, num_parallel_calls=4)
    train_data = train_data.batch(batch_size)

    # for a large batch size (hundreds or thousands)
    # dataset = dataset.apply(tf.contrib.data.map_and_batch(
    #    map_func=_parse_function, batch_size=batch_size))

    # with gpu usage
    train_data = train_data.prefetch(1)
    
    test_data = tf.data.Dataset.from_tensor_slices(
        (test_data["image_dir"].values, 
         pd.get_dummies(test_data["class"]).values))
    test_data = test_data.map(_parse_function, num_parallel_calls=4)
    test_data = test_data.batch(batch_size)
    test_data = test_data.prefetch(1)

    val_data = tf.data.Dataset.from_tensor_slices(
        (val_data["image_dir"].values, 
         pd.get_dummies(val_data["class"]).values))
    val_data = val_data.map(_parse_function, num_parallel_calls=4)
    val_data = val_data.batch(1)
    val_data = val_data.prefetch(1)
    
    iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                               train_data.output_shapes)
    x, y = iterator.get_next()

    train_init = iterator.make_initializer(train_data) # Inicializador para train_data
    test_init = iterator.make_initializer(test_data) # Inicializador para test_data
    val_init = iterator.make_initializer(val_data) # Inicializador para test_data


# In[4]:


# Placeholder
# x = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 1])
# y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])

# Visualize input x
tf.summary.image("input", x, batch_size)

def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add        (tf.nn.conv2d(img, w,        strides=[1, 1, 1, 1],        padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img,         ksize=[1, k, k, 1],        strides=[1, k, k, 1],        padding='SAME')

def avg_pool(img, k):
    return tf.nn.avg_pool(img,         ksize=[1, k, k, 1],        strides=[1, k, k, 1],        padding='SAME')

# weights and bias conv layer 1
# wc1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
wc1 = tf.Variable(tf.random_normal([3, 3, 1, 8]))
bc1 = tf.Variable(tf.random_normal([8]))
tf.summary.histogram("weights", wc1)
tf.summary.histogram("bias", bc1)

# conv layer
conv1 = conv2d(x,wc1,bc1)
tf.summary.histogram("activations", conv1)

# Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 128*128 matrix.
conv1 = max_pool(conv1, k=2)
# conv1 = avg_pool(conv1, k=2)

# dropout to reduce overfitting
keep_prob = tf. placeholder(tf.float32)
conv1 = tf.nn.dropout(conv1,keep_prob)

# weights and bias conv layer 2
wc2 = tf.Variable(tf.random_normal([3, 3, 8, 16]))
bc2 = tf.Variable(tf.random_normal([16]))
tf.summary.histogram("weights", wc2)
tf.summary.histogram("bias", bc2)

# conv layer
conv2 = conv2d(conv1,wc2,bc2)
tf.summary.histogram("activations", conv2)

# Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 64*64 matrix.
conv2 = max_pool(conv2, k=4)
# conv2 = avg_pool(conv2, k=2)

# dropout to reduce overfitting
conv2 = tf.nn.dropout(conv2, keep_prob)

"""
# weights and bias conv layer 3
wc3 = tf.Variable(tf.random_normal([1, 1, 64, 64]))
bc3 = tf.Variable(tf.random_normal([64]))
tf.summary.histogram("weights", wc3)
tf.summary.histogram("bias", bc3)

# conv layer
conv3 = conv2d(conv2,wc3,bc3)
tf.summary.histogram("activations", conv3)

# Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 16*16 matrix.
conv3 = max_pool(conv3, k=4)

# dropout to reduce overfitting
conv3 = tf.nn.dropout(conv3, keep_prob)

"""

# weights and bias fc 1
wd1 = tf.Variable(tf.random_normal([32*32*16, 128]))
bd1 = tf.Variable(tf.random_normal([128]))
tf.summary.histogram("weights", wd1)
tf.summary.histogram("bias", bd1)

# fc 1
dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
# dense1 = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1),bd1))
tf.summary.histogram("activations", dense1)
dense1 = tf.nn.dropout(dense1, keep_prob)

"""
# weights and bias fc 2
wd2 = tf.Variable(tf.random_normal([256, 512]))
bd2 = tf.Variable(tf.random_normal([512]))
tf.summary.histogram("weights", wd2)
tf.summary.histogram("bias", bd2)

# fc 2
dense2 = tf.reshape(dense1, [-1, wd2.get_shape().as_list()[0]])
dense2 = tf.nn.relu(tf.add(tf.matmul(dense2, wd2),bd2))
tf.summary.histogram("activations", dense2)
dense2 = tf.nn.dropout(dense2, keep_prob)

"""
# weights and bias out
wout = tf.Variable(tf.random_normal([128, n_classes]))
bout = tf.Variable(tf.random_normal([n_classes]))
tf.summary.histogram("weights", wout)
tf.summary.histogram("bias", bout)

# prediction
pred = tf.add(tf.matmul(dense1, wout), bout)
# pred = tf.add(tf.matmul(dense2, wout), bout)
tf.summary.histogram("activations", pred)


with tf.name_scope("cross_entropy"):
    # softmax
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    tf.summary.scalar("cross_entropy", cost)
    
# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


with tf.name_scope("accuracy"):
    # Accuracy
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    
    
# Get all summary
summ = tf.summary.merge_all()


# In[ ]:


# Session start

init = tf.global_variables_initializer()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(summaries_dir + '/test', sess.graph)
    
    # Required to get the filename matching to run.
    sess.run(init)
    
    step = 1
    # Compute epochs.
    for i in range(training_epochs):
        print("epoch: {}".format(i))
        epoch_start = time.time()
        sess.run(train_init)
        try:
            while True:
                _, acc, loss = sess.run([optimizer, accuracy, cost], feed_dict={keep_prob: dropout}) 

                if step % display_step == 0:
#                    acc = sess.run(accuracy, feed_dict={keep_prob: 1.})
#                    loss = sess.run(cost, feed_dict={keep_prob: 1.})
#                    train_writer.add_summary(loss, step)
                    print("step: {}".format(step))
                    print("accuracy: {}".format(acc))
                    print("loss: {}".format(loss))
                    print("\n")
                step += 1
        except tf.errors.OutOfRangeError:
            
            print("epoch finished in {} seconds".format(time.time() - epoch_start))
            # Validate

            print("Validation\n")
            sess.run(val_init)
            avg_acc = 0
            avg_loss = 0
            steps=0
            try:
                while True:
                    acc, loss = sess.run([accuracy, cost], feed_dict={keep_prob: 1.})
                    avg_acc += acc
                    avg_loss += loss
                    steps += 1
            except tf.errors.OutOfRangeError:
                print("Average validation set accuracy over {} iterations is {:.2f}%".format(steps,(avg_acc / steps) * 100))
                print("Average validation set loss over {} iterations is {:.2f}".format(steps,(avg_loss / steps)))
    # Test            

    print("Test\n")
    sess.run(test_init)
    avg_acc = 0
    avg_loss = 0
    steps=0
    try:
        while True:
            acc, loss = sess.run([accuracy, cost], feed_dict={keep_prob: 1.})
            avg_acc += acc
            avg_loss += loss
            steps += 1
#            test_writer.add_summary(loss, step)
            print("accuracy: {}".format(acc))
            print("loss: {}".format(loss))
            print("\n")
    except tf.errors.OutOfRangeError:
        print("Average validation set accuracy over {} iterations is {:.2f}%".format(steps,(avg_acc / steps) * 100))
        print("Average validation set loss over {} iterations is {:.2f}".format(steps,(avg_loss / steps)))

