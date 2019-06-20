"""
Created on Wed Jun 19 17:12:04 2019

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images)
from vanilla_backprop import VanillaBackprop
# from guided_backprop import GuidedBackprop  # To use with guided backprop
# from integrated_gradients import IntegratedGradients  # To use with integrated grads

if __name__ == '__main__':
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)

    grad_times_image = vanilla_grads[0] * prep_img.detach().numpy()[0]
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
    # Save grayscale gradients
    save_gradient_images(grayscale_vanilla_grads,
                         file_name_to_export + '_Vanilla_grad_times_image_gray')
    print('Grad times image completed.')
