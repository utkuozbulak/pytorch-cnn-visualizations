"""
Created on Thu Oct 26 14:19:44 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

from torch.optim import SGD
from torchvision import models

from misc_functions import preprocess_image, recreate_image, save_image


class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class):
        self.mean = [-0.485, -0.456, -0.406]
        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.model = model
        self.model.eval()
        self.target_class = target_class
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        # Create the folder to export images if not exists
        if not os.path.exists(f'../generated/class_{self.target_class}'):
            os.makedirs(f'../generated/class_{self.target_class}')

    def generate(self, iterations=150, blur_freq=6, blur_rad=0.8, wd = 0.05):
        initial_learning_rate = 6
        for i in range(1, iterations):
            # Process image and return variable

            #implement gaussian blurring every ith iteration 
            #to improve output
             if i % blur_freq == 0:
                self.processed_image = preprocess_image(
                    self.created_image, False, blur_rad)
            else:
                self.processed_image = preprocess_image(self.created_image, False)

            # Define optimizer for the image - use weight decay to add regularization
            # in SGD, wd = 2 * L2 regularization (https://bbabenko.github.io/weight-decay/)
            optimizer = SGD([self.processed_image], lr=initial_learning_rate, weight_decay=wd)
            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            class_loss = -output[0, self.target_class]

            if i % 50 == 0: print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.data.numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            if i % 10 == 0:
                # Save image
                im_path = f'../generated/class_{self.target_class}/c_{self.target_class}_iter_{i}_loss_{class_loss.data.numpy()}.jpg'
                save_image(self.created_image, im_path)

            #save final image
        im_path = f'../generated/class_{self.target_class}/c_{self.target_class}_iter_{i}_loss_{class_loss.data.numpy()}.jpg'
        save_image(self.created_image, im_path)
        return self.processed_image


if __name__ == '__main__':
    target_class = 130  # Flamingo
    pretrained_model = models.alexnet(pretrained=True)
    csig = ClassSpecificImageGeneration(pretrained_model, target_class)
    csig.generate()
