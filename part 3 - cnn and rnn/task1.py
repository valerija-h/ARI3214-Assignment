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
data_imgs = np.array([ [ [ [ float(px) ] for px in row ] for row in img.split('-') ] for (lbl, img) in data ], np.float32)
data_lbls = np.array([ int(lbl) for (lbl,img) in data ], np.int32)

train_imgs = data_imgs
train_lbls = data_lbls

with open('test.txt', 'r', encoding='utf-8') as f:
    data = [ line.strip().split('\t') for line in f.read().strip().split('\n') ]
test_imgs = np.array([ [ [ [ float(px) ] for px in row ] for row in img.split('-') ] for (lbl, img) in data ], np.float32)
test_lbls = np.array([ int(lbl) for (lbl,img) in data ], np.int32)

# test_imgs[test_imgs == 0] = -1
# data_imgs[data_imgs == 0] = -1

###################################

class Model(object):

    def __init__(self):
        image_width = 28
        image_height = 28
        output_size = 10
        #Set model hyperparameters here
        init_stddev = 1e-2

        kernel_width = 5
        kernel_height = 5
        kernel_size = 2
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.images = tf.placeholder(tf.float32, [None, image_height, image_width, 1], 'images')
            self.targets = tf.placeholder(tf.int32, [None], 'targets')
            
            self.params = []

            batch_size = tf.shape(self.images)[0]

            #Define model here
            with tf.variable_scope('hidden'):
                W = tf.get_variable('W', [kernel_height, kernel_width, 1, kernel_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [kernel_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                self.conv_hs = tf.sigmoid(tf.nn.conv2d(self.images, W, [1,1,1,1], 'VALID') + b)
                
                #The number of rows and columns resulting from the convolution need to be known prior to running the graph because they need to be used to set the weight matrix size in the output layer.
                num_conv_rows = image_height - kernel_height + 1
                num_conv_cols = image_width - kernel_width + 1
                vec_size_per_img = (num_conv_rows*num_conv_cols)*kernel_size
                self.flat_hs = tf.reshape(self.conv_hs, [batch_size, vec_size_per_img])

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [vec_size_per_img, output_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [output_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                
                logits = tf.matmul(self.flat_hs, W) + b
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
max_epochs = 1000

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