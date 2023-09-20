**Fashion MNIST GAN**

This repository contains a Generative Adversarial Network (GAN) implementation for the Fashion MNIST dataset.

**Fashion MNIST** is a dataset of 28x28 grayscale images of 70,000 fashion products from 10 categories, with 7,000 images per category. 

**GANs** are a type of machine learning model that can be used to generate new data, such as images, text, or audio. They work by training two competing models: a generator and a discriminator. The generator tries to generate new data that is indistinguishable from real data, while the discriminator tries to distinguish between real and fake data.

**To train the GAN:**
- number of epochs = 20 # reccommended > 1000
1. Import the neccessary dependancies - tensorflow, tensroflow_datasets
2. Upload the model weights for the generator model from 'generatormodel.h5'
   ```python
   generator.load_model('generatormodel.h5')
 
3. Run the `FashionGan.ipynb` script on colab.(A link to colab is provided)
4. Monitor the training progress by visualizing the loss curve for both generator and discriminator

 **Generating new images**
 ```python
imgs = generator.predict(tf.random.normal((16, 128, 1)))

# Plot the generated images
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(10, 10))
for r in range(4):
    for c in range(4):
        ax[r][c].imshow(imgs[(r + 1) * (c + 1) - 1])
```

**Errors Faced**
- Axes dont match array error occurred as the model architecture is changed after the model weights are saved. When the model weights are loaded back into the model, the axes don't match array error occurs.

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

