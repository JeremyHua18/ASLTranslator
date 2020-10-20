import os
import tensorflow as tf
from tf.keras.optimizers import RMSprop
from tf.keras.preprocessing.image import ImageDataGenerator
from tf.keras import layers
from tf.keras import Model
from tf.keras.applications.inception_v3 import InceptionV3
from tf.keras.optimizers import RMSprop

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.959):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

# Transfer Learning using InceptionV3 Model

base_train_dir = '../dataset/train'
train_a_dir = os.path.join(base_train_dir, 'a')

base_test_dir = '../dataset/test'
test_a_dir = os.path.join(base_test_dir, 'a')

pre_trained_model = InceptionV3(input_shape = (200,200,3),  #Shape of images
                                include_top = False,         
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False

x = layers.Flatten()(pre_trained_model.output)
x = layers.Dense(1024, activation = 'relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation = 'sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer = RMSprop(lr = 0.0001), loss = 'binary_crossentropy', metrics = ['acc'])



# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(base_train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'binary', 
                                                    target_size = (200, 200))     

test_generator =  test_datagen.flow_from_directory( base_test_dir,
                                                          batch_size  = 20,
                                                          class_mode  = 'binary', 
                                                          target_size = (200, 200))


callbacks = myCallback()
history = model.fit_generator(
            train_generator,
            validation_data = test_generator,
            steps_per_epoch = 100,
            epochs = 100,
            validation_steps = 50,
            verbose = 2,
            callbacks=[callbacks])





