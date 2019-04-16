#import tensorflow as tf
import numpy as np
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
import ray
import ray.tune as tune
import argparse
from ray.tune.schedulers import AsyncHyperBandScheduler, PopulationBasedTraining
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

parser = argparse.ArgumentParser()
parser.add_argument("--load", help="if load trial or not",action='store_true')
args = parser.parse_args()



class DenseCifar10Model(tune.Trainable):
	def _read_data(self):
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
		(x_train, y_train),(x_test, y_test) = cifar10.load_data()
		x_train, x_test = x_train / 255.0, x_test / 255.0
		num_classes = 10
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)
		return (x_train, y_train), (x_test, y_test)

	def _build_model(self, input_shape):
		#blks = [self.config['block_0'], self.config['block_1'], self.config['block_2'], self.config['block_3']]
		#print('--------BLOCK:',blks)
		#model = densenet.DenseNet(blocks=[6,6,6,6], input_shape=(32,32,3), include_top=True, weights=None, input_tensor=None, pooling=None, classes=10, backend=keras.backend, layers=keras.layers, models=keras.models)#, keras_utils= keras.utils )
		#depth = 40
		depth = self.config['depth']*3 + 4
		nb_dense_block = 3
		growth_rate = self.config['growth_rate']
		nb_filter = 16 #TODO
		dropout_rate = self.config['dropout']
		weight_decay = self.config['weight_decay']
		model = DenseNet(depth=depth, nb_dense_block=nb_dense_block, growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, input_shape=(32,32,3), weights=None)
		#model = densenet.DenseNet(blocks=[12,24,12,24], input_shape=(32,32,3), include_top=True, weights=None, input_tensor=None, pooling=None, classes=10, backend=keras.backend, layers=keras.layers, models=keras.models)#, keras_utils= keras.utils )
		#model.summary()
		return model
	def _setup(self, config):
		self.train_data, self.test_data = self._read_data()
		x_train = self.train_data[0]
		model = self._build_model(x_train.shape[1:])
		#sgd = keras.optimizers.SGD(lr=self.config['lr'], momentum=self.config['momentum'], nesterov=True)
		sgd = keras.optimizers.SGD(lr=0.1, momentum=self.config['momentum'], nesterov=True)
		model.compile(optimizer=sgd,#'adam',
			loss='categorical_crossentropy',
			metrics=['accuracy'])
		self.model = model
		print('finish setup')
	def _train(self):
		m=1
		x_train, y_train = self.train_data
		x_test, y_test = self.test_data
		x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state = 42)
		def lr_schedule(epoch):
			lr = 0.1
			if epoch>150/m:
				lr = 0.01
			if epoch>250/m:
				lr = 0.001
			return lr
		lr_sched = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
		path = './'+str(self.config)+'/tboard'
		tboard = keras.callbacks.TensorBoard(log_dir=path)
		
		self.model.fit(x_train, y_train, epochs=300//m, validation_data=(x_val,y_val), batch_size=64, callbacks=[lr_sched, tboard])
		self.val_loss, self.val_acc = self.model.evaluate(x_val, y_val)
		#print('Val accuracy:', self.val_acc)
		test_loss, test_acc = self.model.evaluate(x_test, y_test)
		print('Val accuracy:', self.val_acc)
		print('Test accuracy:', test_acc)
		print('Model parameter num: ',self.model.count_params())
		return {"mean_accuracy": self.val_acc}

	def _save(self, checkpoint_dir):
		file_path = checkpoint_dir + "/model"
		self.model.save_weights(file_path)
		#file_path = checkpoint_dir + "/model.h5"
		#self.model.save(file_path)
		return file_path
	def _restore(self, path):
		#self.model=keras.models.load_model(path)
		self.model.load_weights(path)
		print('finish restore model')
	def _stop(self):
		pass

ray.init()	
space = {
	#'lr': hp.uniform('lr',0.0001,0.1),
	'momentum': hp.uniform('momentum',0.7,0.95),
	'dropout': hp.uniform('dropout',0.1,0.8),
	'weight_decay': hp.uniform('weight_decay',1e-5,1e-2),
	'depth': hp.quniform('depth',9,15,q=1),
	
}

start_param = [
	{
		#'lr':0.1,
		'momentum': 0.95,
		'dropout': 0.2,
		'weight_decay': 1e-4,
		'depth': 12
	}

]

hyperopt = HyperOptSearch(space, max_concurrent=1, reward_attr="mean_accuracy", points_to_evaluate=start_param)



'''pbt = PopulationBasedTraining(time_attr="training_iteration", reward_attr="mean_accuracy", perturbation_interval=10,
	hyperparam_mutations={
		#'block_0': lambda: np.random.randint(2, high=25),
		#'block_1': lambda: np.random.randint(2, high=25),
		#'block_2': lambda: np.random.randint(2, high=25),
		#'block_3': lambda: np.random.randint(2, high=25)
		'lr': lambda: np.random.uniform(0.0001, 0.1),#to be modified
		'momentum': lambda: np.random.uniform(0.7,0.95),
		'dropout': lambda: np.random.uniform(0.1,0.8),
		'weight_decay': lambda: np.random.uniform(1e-5,1e-2)
		#'growth_rate': lambda: np.random.randint(10, high=16)

	}
)'''

config={
	#'block': [ tune.sample_from(lambda spec: np.random.randint(2, high=25)), tune.sample_from(lambda spec: np.random.randint(2, high=25)), tune.sample_from(lambda spec: np.random.randint(2, high=25)), tune.sample_from(lambda spec: np.random.randint(2, high=25))]
	#'block_0': 6, 'block_1': 6, 'block_2': 6, 'block_3':6
	#'lr': tune.grid_search([0.1, 0.01, 0.001, tune.sample_from(lambda spec: np.random.uniform(0.0001,0.1)), tune.sample_from(lambda spec: np.random.uniform(0.0001,0.1))]),#sample_from(lambda spec: random.uniform(0.0001,0.1)),
	'lr':0.1,
	'momentum': 0.9,
	'dropout': 0.2,
	'weight_decay': 1e-4,#tune.grid_search([1e-3, 1e-4, 1e-5]),
	'growth_rate': 12#lambda spec: np.random.randint(10, high=16)

}
spec={ 
	"resources_per_trial":{"cpu":8, "gpu":1},
	"stop":{"training_iteration": 1},#'mean_accuracy':0.999},
	#"stop": {"timesteps_total": 100},
	"num_samples": 15,
	"config": config
}
#trials = tune.run(DenseCifar10Model, checkpoint_freq=5, name="my_pbt_Dense40_cifar10_0", scheduler=pbt, **spec)#, resume = True)
trials = tune.run(DenseCifar10Model, checkpoint_freq=5, name="my_hyperopt_Dense40_cifar10_0", search_alg=hyperopt, **spec)#, resume = True)
#trials = tune.run(DenseCifar10Model, checkpoint_freq=5, name="my_hyperopt_Dense40_cifar10_test0", **spec)#, resume = True)
#trials = tune.run(DenseCifar10Model, name="my_pbt_cifar10", scheduler=pbt, **spec)
print(trials)

exit(1)





################################################################################################################################
def train_cifar_tune(config, reporter):

	#mnist = tf.keras.datasets.mnist

	#(x_train, y_train),(x_test, y_test) = mnist.load_data()
	(x_train, y_train),(x_test, y_test) = cifar10.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	num_classes = 10
	#print(y_train)
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	
	x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1, random_state=41)
	'''model = tf.keras.models.Sequential([
	  tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
	  tf.keras.layers.Dense(512, activation=tf.nn.relu),
	  tf.keras.layers.Dropout(0.2),
	  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
	])'''


	print('Default DenseNet: ', config['block'])
	#model = keras_contrib.applications.DenseNet(input_shape=(32,32,3))
	#model = keras_contrib.applications.DenseNetImageNet121(include_top=True, weights=None, input_tensor=None, input_shape=(32,32,3), pooling=None, classes=10

	#NOTE: a workaround to use DenseNet of keras_applications, which is by specifing backend, layers, models.
	model = densenet.DenseNet(blocks=config['block'], input_shape=(32,32,3), include_top=True, weights=None, input_tensor=None, pooling=None, classes=10, backend=keras.backend, layers=keras.layers, models=keras.models)#, keras_utils= keras.utils )


	model.compile(optimizer='adam',
		      loss='categorical_crossentropy',
		      metrics=['accuracy'])
	
	#for i in range(0):
	model.fit(x_train, y_train, epochs=1, batch_size=64)
	val_loss, val_acc = model.evaluate(x_val, y_val)
	#print('Test loss:', test_loss)
	#print('Test accuracy:', test_acc)
	reporter(mean_accuracy=val_acc)



#model.fit(x_train, y_train, epochs=300)
#model.evaluate(x_test, y_test)

configuration = tune.Experiment(
	"pbt_tune_cifar10",
	run=train_cifar_tune,
	resources_per_trial={"cpu":8, "gpu":1},
	stop={'mean_accuracy':0.99},
	config={
		#'block': [ tune.sample_from(lambda spec: np.random.randint(2, high=25)), tune.sample_from(lambda spec: np.random.randint(2, high=25)), tune.sample_from(lambda spec: np.random.randint(2, high=25)), tune.sample_from(lambda spec: np.random.randint(2, high=25))]
		'bloack': [6,6,6,6]
	}	
)

pbt = PopulationBasedTraining(time_attr="training_iteration", reward_attr="mean_accuracy",
	hyperparam_mutations={
		'block': [ tune.sample_from(lambda spec: np.random.randint(2, high=25)), tune.sample_from(lambda spec: np.random.randint(2, high=25)), tune.sample_from(lambda spec: np.random.randint(2, high=25)), tune.sample_from(lambda spec: np.random.randint(2, high=25))]
	}
)

trials=tune.run_experiments(configuration, scheduler=pbt)
print(trials)
exit(1)



sched = AsyncHyperBandScheduler(
	time_attr="timesteps_total",
	reward_attr="mean_accuracy",
	max_t=400,
	grace_period=20)

cifar_spec = {
	'stop':{'mean_accuracy':0.99},
	'config':{
		'block': tune.grid_search([3, 6, 12, 24])
	}
}
#tune.run('train_cifar_tune', name="mytune", scheduler=sched, **cifar_spec)
tune.run('train_cifar_tune', name="mytune", scheduler=sched )
'''tune.run('train_cifar_tune', name="mytune", scheduler=sched, **{
	'stop':{'mean_accuracy':0.99},
	'config':{
		"lr": tune.sample_from(
		    lambda spec: np.random.uniform(0.001, 0.1))
		}
	}
)'''








#test_loss, test_acc = model.evaluate(x_test, y_test)
#print('Test loss:', test_loss)
#print('Test accuracy:', test_acc)

