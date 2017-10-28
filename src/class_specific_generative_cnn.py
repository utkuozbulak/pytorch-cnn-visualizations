"""
Created on Thu Oct 26 14:19:44 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import copy
import cv2
import numpy as np

from torch.optim import SGD
from torchvision import models

from misc_functions import preprocess_image


def recreate_image(im_as_var, reverse_mean, reverse_std):
    """
        Recreates images from a torch variable, sort of reverse preprocessing

    Args:
        im_as_var (torch variable): Image to recreate
        reverse_mean (list): Original mean list multiplied by -1
        reverse_std (list): Original std list to the power -1 (1/org)

    returns:
        recreated_im (numpy arr): Recreated image in array
    """

    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im *= 255

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im


class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class):
        self.mean = [-0.485, -0.456, -0.406]
        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.model = model
        self.target_class = target_class
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        cv2.imwrite('generated/initial_image.jpg', self.created_image)
        # Create the folder to export images if not exists
        if not os.path.exists('generated'):
            os.makedirs('generated')

    def generate(self):
        initial_learning_rate = 6
        for i in range(1, 200):
            # Process image, return variable
            self.processed_image = preprocess_image(self.created_image)
            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            class_loss = -output[0, self.target_class]
            print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.data.numpy()[0]))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image, self.mean, self.std)
            # Save image
            cv2.imwrite('generated/iteration_'+str(i)+'.jpg', self.created_image)
        return self.processed_image


if __name__ == '__main__':
    target_example = 130  # Flamingo
    pretrained_model = models.alexnet(pretrained=True)
    cig = ClassSpecificImageGeneration(pretrained_model, target_example)
    cig.generate()
