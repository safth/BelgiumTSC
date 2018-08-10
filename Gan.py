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


# les datas
Train_data = Data()
Test_data = Data()

Train_data.load_data("Training")
Test_data.load_data("Testing")


Train_data.resize((28,28))
Test_data.resize((28,28))

Train_data.togray()
Test_data.togray()

Train_data.one_hot_encode()
Test_data.one_hot_encode()

def generator(z,reuse=None):
    with tf.variable_scope('gen',reuse=reuse):
        hidden1 = tf.layers.dense(inputs=z,units=128)
        # Leaky Relu
        alpha = 0.01
        hidden1 = tf.maximum(alpha*hidden1,hidden1)
        hidden2 = tf.layers.dense(inputs=hidden1,units=128)

        hidden2 = tf.maximum(alpha*hidden2,hidden2)
        output = tf.layers.dense(hidden2,units=784,activation=tf.nn.tanh)
        return output

def discriminator(X,reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
        hidden1 = tf.layers.dense(inputs=X,units=128)
        # Leaky Relu
        alpha = 0.01
        hidden1 = tf.maximum(alpha*hidden1,hidden1)

        hidden2 = tf.layers.dense(inputs=hidden1,units=128)
        hidden2 = tf.maximum(alpha*hidden2,hidden2)

        logits = tf.layers.dense(hidden2,units=1)
        output = tf.sigmoid(logits)

        return output, logits


real_images = tf.placeholder(tf.float32,shape=[None,784])
z = tf.placeholder(tf.float32,shape=[None,100])

G = generator(z)
D_output_real , D_logits_real = discriminator(real_images)
D_output_fake, D_logits_fake = discriminator(G,reuse=True)

# LOSSES
def loss_func(logits_in,labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))


#nessaie pas de savoir si c'est un 1 ou un 2, veut seulement
#savoir si c'Est une vrai image. on met tout les labels à 1
#... 0.9 en fit pour smoother unpeu
D_real_loss = loss_func(D_logits_real,tf.ones_like(D_logits_real)* (0.9))
D_fake_loss = loss_func(D_logits_fake,tf.zeros_like(D_logits_real))
D_loss = D_real_loss + D_fake_loss
G_loss = loss_func(D_logits_fake,tf.ones_like(D_logits_fake))

learning_rate = 0.001

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]


D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_vars)


init = tf.global_variables_initializer()

batch_size = 100
epochs = 5000
init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list=g_vars)
# Save a sample per epoch
samples = []




with tf.Session() as sess:

    sess.run(init)

    # Recall an epoch is an entire run through the training data
    for e in range(epochs):
        # // indicates classic division
        batch_size=100
        num_batch = int(len(Train_data.images)/batch_size)

        for i in range(num_batch):

            # Grab batch of images
            batch, _ = Train_data.next_batch(batch_size)
            batch_size = (batch.shape[0])

            # Get images, reshape and rescale to pass to D
            batch_images = batch.reshape((batch_size, 784))
            batch_images = batch_images*2 - 1

            # Z (random latent noise data for Generator)
            # -1 to 1 because of tanh activation
            batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))

            # Run optimizers, no need to save outputs, we won't use them
            _ = sess.run(D_trainer, feed_dict={real_images: batch_images, z: batch_z})
            _ = sess.run(G_trainer, feed_dict={z: batch_z})


        print("Currently on Epoch {} of {} total...".format(e+1, epochs))

        # Sample from generator as we're training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size=(1, 100))
        gen_sample = sess.run(generator(z ,reuse=True),feed_dict={z: sample_z})

        samples.append(gen_sample)

#         saver.save(sess, './models/500_epoch_model.ckpt')


plt.imshow(samples[49].reshape(28,28))
plt.show()
