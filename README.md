# Plant-and-Weed-detection-using-UNet-CNN

This repository shows the use of a UNet Architecture CNN for semantic image segmentation of agricultural crops and weeds.
This particular model was trained with images of young maize and sugar beet plants in order to differentiate four different classes:
- 0 - Background
- 1 - Weeds
- 2 - Sugar Beets
- 3 - Maize

This was a school project for my computational intelligence class in my bachelor studies of agricultural technologies and digital farming at the university of applied sciences Wiener Neustadt.
The code for the 'model_training.py' is largely based on [this](https://colab.research.google.com/github/MarkDaoust/models/blob/segmentation_blogpost/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb#scrollTo=7Plun_k1dAML) tutorial by Mark Daoust on Google Colab (the page was accessed in June 2022).

Some changes had to be made in order to get the code to work in Tensorflow 2.10.
Additionally, instead of using dice loss, a weighted cross entropy loss function was used in training.

## Training

The dataset used for training consist of images taken in the field and contain mostly images of maize and some of sugar beets.
The images were taken and hand segmented by me and my class colleagues. Feel free to use them for your own purposes.

The best results were achieved after 245 epochs with a learn rate of 0,0001. That particular model can be found in 'trained_models' folder.

## Results

The trained model can be tested using the 'full_image_testing.py'. It loads a specified model file and passes images located in the 'test_images' directory through the model. It then overlays the resulting segmentation mask with the original image and calculates the percentage of pixels for each class. The resulting graph is then output to the 'temp' directory.

Here is an example of that:

![example result of maize](/results/full_image_test4.png)
