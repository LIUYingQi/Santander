import tensorflow as tf
import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

data_path = "input/"

#################################################################################################################
#######  input normalization
#################################################################################################################

f_train_x = open(data_path+'trainset.pkl','rb')
f_train_y = open(data_path+'label.pkl','rb')

trainset = pickle.load(f_train_x)
label = pickle.load(f_train_y)

print trainset.shape
print label.shape

f_train_x.close()
f_train_y.close()

train_y = np.zeros((trainset.shape[0],22),np.int8)
for i in range(trainset.shape[0]):
    train_y[i,label[i]] = 1

stander = StandardScaler()
train_x = stander.fit_transform(trainset)

print train_x.shape
print train_y.shape

#################################################################################################################
#######  define model
#################################################################################################################

# initial for weight
def init_weight(shape):
    return tf.Variable(tf.random_normal(shape,stddev=1.0))


# define X Y
X = tf.placeholder(tf.float32,[None,train_x.shape[1]])
Y = tf.placeholder(tf.float32,[None,train_y.shape[1]])

# get lstm size and output HI

W1 = init_weight([train_x.shape[1],50])
B1 = init_weight([50])

W2 = init_weight([50,35])
B2 = init_weight([35])

W = init_weight([35,22])
B = init_weight([22])

# get the last output

layer1 = tf.matmul(X, W1) + B1

layer2 = tf.matmul(layer1, W2) + B2

output = tf.matmul(layer2, W) + B

py_x = tf.nn.softmax(output)
cost = -tf.reduce_sum(Y * tf.log(py_x + 1e-9))
train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)
predict = tf.argmax(py_x , 1)

# 'Saver' op to save and restore all the variables
model_path = "model.ckpt"
saver = tf.train.Saver()

with tf.Session() as sess:

    # model value initialization
    tf.initialize_all_variables().run()

    predict_label = sess.run(predict, feed_dict={X: train_x, Y: train_y})
    # learning_steps
    while accuracy_score(label,predict_label) < 0.9:
        print '                                                            '
        print '############################################################'
        print '                                                            '

        cost_item,_ = sess.run([cost,train_op],feed_dict={X:train_x,Y:train_y})
        print cost_item
        predict_prob = sess.run(py_x,feed_dict={X:train_x})
        predict_label = sess.run(predict,feed_dict={X:train_x})

        print predict_label.shape
        print label.shape
        print accuracy_score(label,predict_label)

    save_path = saver.save(sess, model_path)
