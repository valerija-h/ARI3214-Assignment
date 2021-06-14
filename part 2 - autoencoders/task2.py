import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import tsne
import timeit

start_time = timeit.default_timer()

###################################

with open('dataset.txt', 'r', encoding='utf-8') as f:
    data = [ line.strip().split('\t') for line in f.read().strip().split('\n') ]
data_imgs = np.array([ [ float(px) for px in img.replace('-', '') ] for (lbl, img) in data ], np.float32)
data_lbls = np.array([ int(lbl) for (lbl,img) in data ], np.int32)

train_imgs = data_imgs
train_lbls = data_lbls

with open('test.txt', 'r', encoding='utf-8') as f:
    data = [ line.strip().split('\t') for line in f.read().strip().split('\n') ]
test_imgs = np.array([ [ float(px) for px in img.replace('-', '') ] for (lbl, img) in data ], np.float32)
test_lbls = np.array([ int(lbl) for (lbl,img) in data ], np.int32)

###################################

class Model(object):

    def __init__(self):
        #Set model hyperparameters here
        init_stddev = 1e-2
        # learning_rate and momentum is commented out because the Adam Optimizer is used
        # learning_rate = 1.2
        # momentum = 0.9

        input_size = 28*28
        gen_output_size = input_size
        pred_output_size = 10
        thought_vector_size = 256
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.images = tf.placeholder(tf.float32, [None, input_size], 'images')
            self.targets = tf.placeholder(tf.int32, [None], 'targets')
            
            self.params = []

            #Define model here
            # hidden layer
            with tf.variable_scope('hidden'):
                W = tf.get_variable('W', [input_size, thought_vector_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [thought_vector_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                self.thought_vectors = tf.tanh(tf.matmul(self.images, W) + b)

            # generation layer
            with tf.variable_scope('gen'):
                W = tf.get_variable('W', [thought_vector_size, gen_output_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [gen_output_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                logits = tf.matmul(self.thought_vectors, W) + b
                self.out_images = tf.sigmoid(logits)
                self.error_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.images, logits=logits))
            
            # prediction layer
            with tf.variable_scope('pred'):
                W = tf.get_variable('W', [thought_vector_size, pred_output_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [pred_output_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                logits = tf.matmul(self.thought_vectors, W) + b    
                self.probs = tf.nn.softmax(logits)
                self.error_pred = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=logits))
                
            self.error = self.error_gen + (self.error_pred*0.6) # adding a weighting to prioritize error_gen
            self.optimiser_step = tf.compat.v1.train.AdamOptimizer().minimize(self.error) # Adam Optimizer
            
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
    
    def get_thoughtvectors(self, images):
        return self.sess.run([ self.thought_vectors ], { self.images: images })[0]
    
    def predict(self, images):
        return self.sess.run([ self.probs, self.out_images ], { self.images: images })

###################################

#Set training hyperparameters here
max_epochs = 2000

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
print()

(fig, axs) = plt.subplots(4, 5)

accuracy = np.sum(np.round(model.predict(test_imgs)[1]) == test_imgs)/test_imgs.size
duration = round((timeit.default_timer() - start_time)/60, 1)
num_params = sum(p.size for p in model.get_params())

digit = 0
row = 0
for _ in range(2):
    for col in range(5):
        img = test_imgs[test_lbls == digit][0]
        [ [ probs ], [ out_img ] ] = model.predict([ img ])
        prediction = np.argmax(probs)
        
        axs[row,col].set_axis_off()
        axs[row,col].matshow(np.reshape(img, [28, 28]), vmin=0.0, vmax=1.0, cmap='bwr')
        
        axs[row+1,col].set_axis_off()
        axs[row+1,col].matshow(np.reshape(out_img, [28, 28]), vmin=0.0, vmax=1.0, cmap='bwr')
        axs[row+1,col].text(1.0, 0.0, str(prediction), dict(fontsize=10, ha='right', va='top', transform=axs[row+1,col].transAxes), color='red' if prediction != digit else 'green')
        
        digit += 1
        if digit == 5:
            row += 2
axs[1,4].text(1.0, 0.5, 'Accuracy: {:.2%}\nDuration: {}min\nParams: {}'.format(accuracy, duration, num_params), dict(fontsize=10, ha='left', va='center', transform=axs[1,4].transAxes))

fig.canvas.set_window_title('Generated images')
fig.tight_layout()
fig.show()

(fig, ax) = plt.subplots(1, 1)

thought_vectors = model.get_thoughtvectors(test_imgs)
points_2d = tsne.tsne(thought_vectors)
for digit in range(0, 9+1):
    ax.plot(points_2d[test_lbls==digit, 0], points_2d[test_lbls==digit, 1], linestyle='', marker='o', markersize=5, label=str(digit))
ax.legend()
fig.tight_layout()
fig.show()

model.close()
input()