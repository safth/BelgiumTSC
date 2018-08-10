import numpy as np
import tensorflow as tf
import os
from skimage import io, transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


class Data():

    def __init__(self):
        self.i = 0
        self.nb_labels = 0
        self.shape= ()
        self.images = [None]
        self.labels = [None]
    # methode pour loader les fichiers.
    def load_data(self,data_directory):
        #Fait une liste des dossiers.
        directories = [d for d in os.listdir(data_directory)
                       if os.path.isdir(os.path.join(data_directory, d))]
        images = []
        labels = []
        for d in directories: # loop sur les dossiers (labels)
            label_directory = os.path.join(data_directory,d) #donne le path des labels
            # fait une lite des fichiers dans un dossier (labels)
            file_names = [os.path.join(label_directory, file) for file
                          in os.listdir(label_directory) if file.endswith(".ppm")]
            # la on a toutes les images d'un dossier, loop sur eux
            for file in file_names:
                images.append(io.imread(file))
                labels.append(int(d))
        self.images=images
        self.labels=labels
        self.nb_labels = len(set(Train_data.labels))
        print("images and labels loaded")

    # methode pour redimentionner les images
    def resize(self,dim=(28,28)):
        self.images = [transform.resize(image,(dim[0],dim[1])) for image in self.images]
        self.shape = dim

    # methode pour enlever le channel de couleur
    def togray(self):
        self.images = rgb2gray(np.array(self.images))

    def one_hot_encode(self):
        n=len(self.labels)
        out = np.zeros((n,self.nb_labels))
        out[range(n), self.labels]=1
        self.labels = out

        #methode qui donne des batch de datas
    def next_batch(self, batch_size=100):
        x = self.images[self.i:self.i+batch_size]
        y = self.labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.images)
        return x, y

#
# Helpers function pour le modèle
#
# INIT WEIGTH
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(init_random_dist)

    # INIT BIAS
def init_bias(shape):
    init_bias_vals = tf.constant(0.1,shape=shape) #constante=0.1
    return tf.Variable(init_bias_vals)

    # CONVOLUTION 2D
def conv2d(x,W):
    # x --> input tensor [Batch,Height,width,channels]
    # w --> kernel [filter height, filter width, channels In, channels Out]

    #SAME -> padding de zeros (met des zeros autour de l'image)
    #strides -> on se déplace de 1 pixel dans chaque direction
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# POOLING prend le max d'une grid->enleve des datas
def max_pool_2by2(x):
    # x --> input tensor [Batch,Height,width,channels]
    #on pool juste sur heigth & width [1,2,2,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# CONVOLUTIONNAL LAYER
def convolutional_layer(input_x,shape):
    W=init_weights(shape)
    b=init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W)+b)

    # NORMAL LAYER (FULLY CONNECTED)
def normal_full_layer(input_layer,size):
    input_size = int(input_layer.get_shape()[1])

    W = init_weights([input_size,size])
    b = init_bias([size])
    return tf.matmul(input_layer,W)+b


# les datas
Train_data = Data()
Test_data = Data()

Train_data.load_data("/Users/simonboivin/Documents/BelgiumTSC/Training")
Test_data.load_data("/Users/simonboivin/Documents/BelgiumTSC/Testing")

Train_data.resize()
Test_data.resize()

Train_data.togray()
Test_data.togray()

Train_data.one_hot_encode()
Test_data.one_hot_encode()

#
# modele
#
#

num_output = Train_data.nb_labels
x=tf.placeholder(tf.float32,shape=[None,28,28])
y_true = tf.placeholder(tf.float32,shape=[None,num_output])
x_image = tf.reshape(x,[-1,28,28,1])

convo_1 = convolutional_layer(x_image,shape=[5,5,1,32]) #input 1, output 32
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling,shape=[5,5,32,64]) #input 32, output 64
convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

#dropout
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

#dropout
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout,num_output)

# LOSS FUNCTION
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

# OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train =optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()


epoch = 400
batch_size=100

with tf.Session() as sess:
    sess.run(init)
    for e in range(epoch):
        for step in range(int(len(Train_data.images)/batch_size)):

            batch_x, batch_y = Train_data.next_batch(batch_size)

            sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})

        matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
        acc = tf.reduce_mean(tf.cast(matches,tf.float32))
        print("on epoch: {}".format(e))
        print("Accuracy:")
        print(sess.run(acc,feed_dict={x:Test_data.images,y_true:Test_data.labels,hold_prob:1.0}))
