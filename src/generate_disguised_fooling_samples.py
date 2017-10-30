"""
Created on Thu Oct 29 14:09:01 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import cv2

from torch.optim import SGD
from torchvision import models
from torch.nn import functional

from misc_functions import preprocess_image, recreate_image, get_params


class DisguisedFoolingSampleGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent, breaks as soon as
        the target prediction confidence is captured
    """
    def __init__(self, model, initial_image, target_class, minimum_confidence):
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.minimum_confidence = minimum_confidence
        # Generate a random image
        self.initial_image = initial_image
        # Create the folder to export images if not exists
        if not os.path.exists('generated'):
            os.makedirs('generated')

    def generate(self):
        for i in range(1, 500):
            # Process image and return variable
            self.processed_image = preprocess_image(self.initial_image)
            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=0.7)
            # Forward
            output = self.model(self.processed_image)
            # Get confidence from softmax
            target_confidence = functional.softmax(output)[0][self.target_class].data.numpy()[0]
            if target_confidence > self.minimum_confidence:
                # Reading the raw image and pushing it through model to see the prediction
                # this is needed because the format of preprocessed image is float and when
                # it is written back to file it is converted to uint8, so there is a chance that
                # there are some losses while writing
                confirmation_image = cv2.imread('generated/fooling_sample_class_' +
                                                str(self.target_class) + '.jpg', 1)
                # Preprocess image
                confirmation_processed_image = preprocess_image(confirmation_image)
                # Get prediction
                confirmation_output = self.model(confirmation_processed_image)
                # Get confidence
                softmax_confirmation = \
                    functional.softmax(confirmation_output)[0][self.target_class].data.numpy()[0]
                if softmax_confirmation > self.minimum_confidence:
                    print('Generated fooling image with', "{0:.2f}".format(softmax_confirmation),
                          'confidence at', str(i) + 'th iteration.')
                    break
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
            self.initial_image = recreate_image(self.processed_image)
            # Save image
            cv2.imwrite('generated/fooling_sample_class_' + str(self.target_class) + '.jpg',
                        self.initial_image)
        return confirmation_image


if __name__ == '__main__':
    target_example = 3  # Appple
    (original_image, prep_img, target_class, _, _) =\
        get_params(target_example)

    fooling_target_class = 398  # Abacus
    lowest_confidence = 0.99
    pretrained_model = models.alexnet(pretrained=True)
    fool = DisguisedFoolingSampleGeneration(pretrained_model,
                                            original_image,
                                            fooling_target_class,
                                            lowest_confidence)
    generated_image = fool.generate()
