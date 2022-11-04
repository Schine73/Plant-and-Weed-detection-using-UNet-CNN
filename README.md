# Plant-and-Weed-detection-using-UNet-CNN

This repository shows the use of a UNet Architecture CNN for semantic image segmentation of agricultural crops and weeds.
This particular model was trained with images of young maize and sugar beet plants in order to differentiate four different classes:
- 0 - Background
- 1 - Weeds
- 2 - Sugar Beets
- 3 - Maize

This was a school project for my computational intelligence class in my bachelor studies of agricultural technologies and digital farming at the university of applied sciences Wiener Neustadt.
The code for the 'model_training.py' is largely based on [this](https://colab.research.google.com/github/MarkDaoust/models/blob/segmentation_blogpost/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb#scrollTo=7Plun_k1dAML) tutorial by Mark Daoust on Google Colab (the page was accessed in June 2022).

Some changes had to be made in order to get the code to work on Tensorflow 2.10.
Additionally, instead of using dice loss, a weighted cross entropy loss function was used in training.

The training images were taken and segmented by me and my class colleagues. Feel free to use them for your own purposes.
