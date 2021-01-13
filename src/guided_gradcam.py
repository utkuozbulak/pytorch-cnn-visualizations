"""
Created on Thu Oct 23 11:27:15 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np

from src.misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images)
from src.gradcam import GradCam
from src.guided_backprop import GuidedBackprop


def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb


if __name__ == '__main__':
    # Get params
    seed = 10
    np.random.seed(seed)
    target_example = 1  # Snake
    for seed in [0, 1]:
        for target_example in [0,1,2]:
            for pflag in [True, False]:
                for dflag in [True, False]:
                    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
                        get_example_params(target_example, pretrain=pflag, diffclass=dflag)

                    # Grad cam
                    gcv2 = GradCam(pretrained_model, target_layer=11)
                    # Generate cam mask
                    cam = gcv2.generate_cam(prep_img, target_class)
                    print('Grad cam completed')

                    # Guided backprop
                    GBP = GuidedBackprop(pretrained_model)
                    # Get gradients
                    guided_grads = GBP.generate_gradients(prep_img, target_class)
                    print('Guided backpropagation completed')

                    # Guided Grad cam
                    cam_gb = guided_grad_cam(cam, guided_grads)
                    save_gradient_images(cam_gb, file_name_to_export + '_seed%d'%seed)
                    grayscale_cam_gb = convert_to_grayscale(cam_gb)
                    res = save_gradient_images(grayscale_cam_gb, file_name_to_export + '_seed%d'%seed)
    # from PIL import Image
    # res = Image.fromarray(res[0]*255)
    # res.show()
    # print('Guided grad cam completed')
