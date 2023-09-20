**Fashion MNIST GAN**

This repository contains a Generative Adversarial Network (GAN) implementation for the Fashion MNIST dataset.

**Fashion MNIST** is a dataset of 28x28 grayscale images of 70,000 fashion products from 10 categories, with 7,000 images per category. 

**GANs** are a type of machine learning model that can be used to generate new data, such as images, text, or audio. They work by training two competing models: a generator and a discriminator. The generator tries to generate new data that is indistinguishable from real data, while the discriminator tries to distinguish between real and fake data.

**To train the GAN:**
- number of epochs = 20 # reccommended > 1000
1. Download the Fashion MNIST dataset from tensorflow_datasets
2. Run the `FashionGan.ipynb` script.
3. Monitor the training progress by visualizing the generated images and the loss curves.
4. After training, upload the model weights for the generator model from 'generatormodel.h5'.
5. ```python
   generator.load_model('generatormodel.h5')

**Errors Faced**
- axes dont match array error occurred as the model architecture is changed after the model weights are saved. When the model weights are loaded back into the model, the axes don't match array error occurs.

To avoid this error, you can freeze the model's layers before loading the weights:
  
  ```python
  from keras.models import Model
  def freeze_layers(model):
    for i in model.layers:
        i.trainable = False
        if isinstance(i, Model):
            freeze_layers(i)
    return model

model_freezed = freeze_layers(generator)
model_freezed.save("/content/generatormodel.h5") 

