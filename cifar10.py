#import tensorflow as tf
import keras
#from tensorflow.keras.layers import Input
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
#import keras_contrib.applications
#from keras import applications
from keras_applications import densenet
from keras_contrib.applications import DenseNet
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--load", help="if load trial or not",action='store_true')
args = parser.parse_args()


#mnist = tf.keras.datasets.mnist

#(x_train, y_train),(x_test, y_test) = mnist.load_data()
(x_train, y_train),(x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
num_classes = 10
#print(y_train)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])'''



print('Default DenseNet:')
#model = keras_contrib.applications.DenseNet(input_shape=(32,32,3))
#model = keras_contrib.applications.DenseNetImageNet121(include_top=True, weights=None, input_tensor=None, input_shape=(32,32,3), pooling=None, classes=10


#model = densenet.DenseNet(blocks=[12, 24, 12, 24], input_shape=(32,32,3), include_top=True, weights=None, input_tensor=None, pooling=None, classes=10,backend=keras.backend, layers=keras.layers, models=keras.models)#, keras_utils= keras.utils )
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = 16
dropout_rate = 0.2
weight_decay = 1e-4

model = DenseNet(depth=depth, nb_dense_block=nb_dense_block, growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay, input_shape=(32,32,3), weights=None)

sgd = keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
#model.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state = 42)

if args.load:
	model.load_weights('./checkpoints/my_checkpoint_tmp_best_0')
for i in range(30):
	model.fit(x_train, y_train, epochs=10, batch_size=64)
	if i == 15:
		sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
		model.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['accuracy'])
	elif i ==25:
		sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
		model.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['accuracy'])
		
	weights_path = './checkpoints/my_checkpoint_tmp_best_' + str(i)
	model.save_weights(weights_path)
#model.evaluate(x_test, y_test)
	test_loss, test_acc = model.evaluate(x_test, y_test)
	val_loss, val_acc = model.evaluate(x_val, y_val)
	print('Val accuracy at ', i, ": ", val_acc)
	print('Test accuracy at ', i, ": ", test_acc)

