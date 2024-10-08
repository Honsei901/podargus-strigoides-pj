# Vechicle Image Classification Model

This project involces building a model to classify image of cars and motorcycles. The loss function used is Adam. During training, the accuracy starts to drop, likely due to overfitting caused by insufficient data. To address this issue, transfer learning using a pre-trained VGG16 model is introduced to improve the model's performace.

## Technologies Used

- TensorFlow / Keras
- Python 3.9.16
- CNN

## Objective

The goal is to build a model capable of classifying two types of images: cars and motorcycles.

## Structure

1. Initial Model Construction <br />
   A CNN is used to build the image classification model, with Adam as the loss function.
   <br />
   <br />
2. Identified Problem <br />
   The model initially achieves reasonable accuracy, but performace starts to degrade during training. This is likely due to overfitting caused by the small dataset.
   <br />
   <br />
3. Solution <br />
   Transfer learning with a pre-trained VGG16 model is introduced. By leveraging weights trained on a large dataset, the model can maintain higher accuracy even with a smaller dataset.
