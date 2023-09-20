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
- axes dont match array error occurred due to change in architecture shape during training of the generator model with the 'generatormodel.h5' shape. So, I used the freeze_layers to freeze layers in keras. 

**Generating new images** 
```python

imgs = generator.predict(tf.random.normal((16, 128, 1)))
# Plot the generated images
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(10, 10))
for r in range(4):
    for c in range(4):
        ax[r][c].imshow(imgs[(r + 1) * (c + 1) - 1])



        
