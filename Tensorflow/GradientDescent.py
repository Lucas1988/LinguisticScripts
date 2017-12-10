'''
python GradientDescent.py Data2.csv test.csv RawResult.txt 0.0001 1000000
'''
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import sys

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Load training data
dataframe = pd.read_csv(sys.argv[1])
# Removing features we don't care about
dataframe = dataframe.drop(['id'], axis=1)

# Load test data
dataframe2 = pd.read_csv(sys.argv[2])
# Removing features we don't care about
dataframe2 = dataframe2.drop(['id'], axis=1)

# Define output file
outfile = open(sys.argv[3], 'w+')

# Prepare data for TensorFlow

# Convert training set features to input tensor
inputX = dataframe.loc[:, ['feat_1','feat_2','feat_3','feat_4','feat_5','feat_6','feat_7','feat_8','feat_9','feat_10','feat_11','feat_12','feat_13','feat_14','feat_15','feat_16','feat_17','feat_18','feat_19','feat_20','feat_21','feat_22','feat_23','feat_24','feat_25','feat_26','feat_27','feat_28','feat_29','feat_30','feat_31','feat_32','feat_33','feat_34','feat_35','feat_36','feat_37','feat_38','feat_39','feat_40','feat_41','feat_42','feat_43','feat_44','feat_45','feat_46','feat_47','feat_48','feat_49','feat_50','feat_51','feat_52','feat_53','feat_54','feat_55','feat_56','feat_57','feat_58','feat_59','feat_60','feat_61','feat_62','feat_63','feat_64','feat_65','feat_66','feat_67','feat_68','feat_69','feat_70','feat_71','feat_72','feat_73','feat_74','feat_75','feat_76','feat_77','feat_78','feat_79','feat_80','feat_81','feat_82','feat_83','feat_84','feat_85','feat_86','feat_87','feat_88','feat_89','feat_90','feat_91','feat_92','feat_93']].as_matrix()
# Convert training set labels to input tensor
inputY = dataframe.loc[:, ['class_1','class_2','class_3','class_4','class_5','class_6','class_7','class_8','class_9',]].as_matrix()

# Convert test set feautures to input tensor
TestInputX = dataframe2.loc[:, ['feat_1','feat_2','feat_3','feat_4','feat_5','feat_6','feat_7','feat_8','feat_9','feat_10','feat_11','feat_12','feat_13','feat_14','feat_15','feat_16','feat_17','feat_18','feat_19','feat_20','feat_21','feat_22','feat_23','feat_24','feat_25','feat_26','feat_27','feat_28','feat_29','feat_30','feat_31','feat_32','feat_33','feat_34','feat_35','feat_36','feat_37','feat_38','feat_39','feat_40','feat_41','feat_42','feat_43','feat_44','feat_45','feat_46','feat_47','feat_48','feat_49','feat_50','feat_51','feat_52','feat_53','feat_54','feat_55','feat_56','feat_57','feat_58','feat_59','feat_60','feat_61','feat_62','feat_63','feat_64','feat_65','feat_66','feat_67','feat_68','feat_69','feat_70','feat_71','feat_72','feat_73','feat_74','feat_75','feat_76','feat_77','feat_78','feat_79','feat_80','feat_81','feat_82','feat_83','feat_84','feat_85','feat_86','feat_87','feat_88','feat_89','feat_90','feat_91','feat_92','feat_93']].as_matrix()

# Write out hyperparameters
learning_rate = float(sys.argv[4])
training_epochs = int(sys.argv[5])
display_step = 50
n_samples = inputY.size

# Creating the neural network
x = tf.placeholder(tf.float32, [None,93])

# Create weights
# Float matrix
W = tf.Variable(tf.zeros([93,9]))

# Add biases
b = tf.Variable(tf.zeros([9]))

# Multiply weights by inputs
y_values = tf.add(tf.matmul(x, W), b)

# Apply softmax to value
y = tf.nn.softmax(y_values)

# Feed in a matrix of labels
y_ = tf.placeholder(tf.float32, [61878, 9])

# Perform training
cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize variables and TensorFlow session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Training loop
for i in range(training_epochs):
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY})

    # Write out training logs    
    if (i) % display_step == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_ : inputY})
        print("Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc))

print('Optimization finished!')
#training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
# print(("Training cost:", training_cost, "W=", sess.run(W), "b=", sess.run(b), "\n"), file=outfile)

# Print inference results to output file
np.set_printoptions(threshold=np.nan)
print(((sess.run(y, feed_dict = {x: TestInputX}))), file=outfile)
