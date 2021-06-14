import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import timeit

start_time = timeit.default_timer()

###################################

with open('dataset.txt', 'r', encoding='utf-8') as f:
    data = [ line.strip().split('\t') for line in f.read().strip().split('\n') ]
data_imgs = np.array([ [ [ float(px) for px in row ] for row in img.split('-') ] for (lbl, img) in data ], np.float32)
data_lbls = np.array([ int(lbl) for (lbl,img) in data ], np.int32)

train_imgs = data_imgs
train_lbls = data_lbls

with open('test.txt', 'r', encoding='utf-8') as f:
    data = [ line.strip().split('\t') for line in f.read().strip().split('\n') ]
test_imgs = np.array([ [ [ float(px) for px in row ] for row in img.split('-') ] for (lbl, img) in data ], np.float32)
test_lbls = np.array([ int(lbl) for (lbl,img) in data ], np.int32)

###################################

class Model(object):

    def __init__(self):
        image_width = 28
        image_height = 28
        output_size = 10
        #Set model hyperparameters here
        init_stddev = 1e-1
        state_size = 256 #RNN state vector size.
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.images = tf.placeholder(tf.float32, [None, image_height, image_width], 'images')
            self.targets = tf.placeholder(tf.int32, [None], 'targets')
            
            self.params = []

            batch_size = tf.shape(self.images)[0]

            with tf.variable_scope('hidden'):
                init_state = tf.get_variable('init_state', [state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev)) #Allow initial state to be a variable that is optimised with the other parameters
                batch_init = tf.tile(tf.reshape(init_state, [1, state_size]), [batch_size, 1]) #Replicate the initial state for every item in the batch
                
                cell = tf.nn.rnn_cell.BasicRNNCell(num_units=state_size)
                (_,self.states) = tf.nn.dynamic_rnn(cell, self.images, initial_state=batch_init)

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [state_size, output_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [output_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                
                logits = tf.matmul(self.states, W) + b
                self.probs = tf.nn.softmax(logits)
            
            self.error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=logits)) 
            self.optimiser_step = tf.train.AdamOptimizer().minimize(self.error)

            self.sensitivity = tf.abs(tf.gradients([ tf.reduce_max(self.probs[0]) ], [ self.images ])[0][0])
            self.prediction = tf.arg_max(self.probs[0], dimension=0)
            
            self.init = tf.global_variables_initializer()
            
            self.graph.finalize()

            self.sess = tf.Session()
    
    def initialise(self):
        self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def optimisation_step(self, images, targets):
        return self.sess.run([ self.optimiser_step ], { self.images: images, self.targets: targets })
    
    def get_params(self):
        return self.sess.run(self.params, { })
    
    def get_error(self, images, targets):
        return self.sess.run([ self.error ], { self.images: images, self.targets: targets })[0]
    
    def get_sensitivity(self, image):
        return self.sess.run([ self.prediction, self.sensitivity ], { self.images: [ image ] })
    
    def predict(self, images):
        return self.sess.run([ self.probs ], { self.images: images })[0]

###################################

#Set training hyperparameters here
max_epochs = 500

(fig, ax) = plt.subplots(1, 1)

[ train_error_plot ] = ax.plot([], [], color='red', linestyle='-', linewidth=1, label='train')
ax.set_xlim(0, max_epochs)
ax.set_xlabel('epoch')
ax.set_ylim(0.0, 2.5)
ax.set_ylabel('Error')
ax.grid(True)
ax.set_title('Error progress')
ax.legend()

fig.canvas.set_window_title('Training progress')
fig.tight_layout()
fig.show()

###################################

model = Model()
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs+1):
    train_error = model.get_error(train_imgs, train_lbls)
    train_errors.append(train_error)
    
    if epoch%100 == 0:
        print(epoch, train_error, sep='\t')

        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    #Optimisation per epoch here
    model.optimisation_step(train_imgs, train_lbls)
    
(fig, axs) = plt.subplots(2, 5)

digit = 0
for row in range(2):
    for col in range(5):
        img = test_imgs[test_lbls == digit][0]
        [ prediction, sensitivity ] = model.get_sensitivity(img)
        sensitivity = (sensitivity - np.min(sensitivity))/(np.max(sensitivity) - np.min(sensitivity))
        
        axs[row,col].set_axis_off()
        axs[row,col].matshow(np.reshape(1 - img, [28, 28]), vmin=0.0, vmax=1.0, cmap='gray')
        axs[row,col].matshow(np.reshape(sensitivity, [28, 28]), vmin=0.0, vmax=1.0, cmap='bwr', alpha=0.8)
        axs[row,col].text(1.0, 0.0, str(prediction), dict(fontsize=10, ha='right', va='top', transform=axs[row,col].transAxes), color='red' if prediction != digit else 'green')
        axs[row,col].set_title(str(digit))
        
        digit += 1

fig.canvas.set_window_title('Sensitivity analysis')
fig.tight_layout()
fig.show()

predictions = np.argmax(model.predict(test_imgs), axis=1)
confusion_matrix = [ [ np.sum(predictions[test_lbls == target] == output) for output in range(10) ] for target in range(10) ]
accuracy = np.sum(predictions == test_lbls)/len(test_lbls)
duration = round((timeit.default_timer() - start_time)/60, 1)
num_params = sum(p.size for p in model.get_params())

(fig, ax) = plt.subplots(1, 1)

ax.matshow(confusion_matrix, cmap='bwr')
ax.set_xlabel('output')
ax.set_ylabel('target')
ax.text(1.0, 0.5, 'Accuracy: {:.2%}\nDuration: {}min\nParams: {}'.format(accuracy, duration, num_params), dict(fontsize=10, ha='left', va='center', transform=ax.transAxes))

fig.canvas.set_window_title('Confusion matrix')
fig.tight_layout()
fig.show()

input()

model.close()