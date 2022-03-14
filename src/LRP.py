# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:32:09 2022

@author: ut
"""
import copy
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from misc_functions import apply_heatmap, get_example_params


class LRP():
    """
        Layer-wise relevance propagation with gamma+epsilon rule

        This code is largely based on the code shared in: https://git.tu-berlin.de/gmontavon/lrp-tutorial
        Some stuff is removed, some stuff is cleaned, and some stuff is re-organized compared to that repository.
    """
    def __init__(self, model):
        self.model = model

    def LRP_forward(self, layer, input_tensor, gamma=None, epsilon=None):
        # This implementation uses both gamma and epsilon rule for all layers
        # The original paper argues that it might be beneficial to sometimes use
        # or not use gamma/epsilon rule depending on the layer location
        # Have a look a the paper and adjust the code according to your needs

        # LRP-Gamma rule
        if gamma is None:
            gamma = lambda value: value + 0.05 * copy.deepcopy(value.data.detach()).clamp(min=0)
        # LRP-Epsilon rule
        if epsilon is None:
            eps = 1e-9
            epsilon = lambda value: value + eps

        # Copy the layer to prevent breaking the graph
        layer = copy.deepcopy(layer)

        # Modify weight and bias with the gamma rule
        try:
            layer.weight = nn.Parameter(gamma(layer.weight))
        except AttributeError:
            pass
            # print('This layer has no weight')
        try:
            layer.bias = nn.Parameter(gamma(layer.bias))
        except AttributeError:
            pass
            # print('This layer has no bias')
        # Forward with gamma + epsilon rule
        return epsilon(layer(input_tensor))

    def LRP_step(self, forward_output, layer, LRP_next_layer):
        # Enable the gradient flow
        forward_output = forward_output.requires_grad_(True)
        # Get LRP forward out based on the LRP rules
        lrp_rule_forward_out = self.LRP_forward(layer, forward_output, None, None)
        # Perform element-wise division
        ele_div = (LRP_next_layer / lrp_rule_forward_out).data
        # Propagate
        (lrp_rule_forward_out * ele_div).sum().backward()
        # Get the visualization
        LRP_this_layer = (forward_output * forward_output.grad).data

        return LRP_this_layer

    def generate(self, input_image, target_class):
        layers_in_model = list(self.model._modules['features']) + list(self.model._modules['classifier'])
        number_of_layers = len(layers_in_model)
        # Needed to know where flattening happens
        features_to_classifier_loc = len(self.model._modules['features'])

        # Forward outputs start with the input image
        forward_output = [input_image]
        # Then we do forward pass with each layer
        for conv_layer in list(self.model._modules['features']):
            forward_output.append(conv_layer.forward(forward_output[-1].detach()))

        # To know the change in the dimensions between features and classifier
        feature_to_class_shape = forward_output[-1].shape
        # Flatten so we can continue doing forward passes at classifier layers
        forward_output[-1] = torch.flatten(forward_output[-1], 1)
        for index, classifier_layer in enumerate(list(self.model._modules['classifier'])):
            forward_output.append(classifier_layer.forward(forward_output[-1].detach()))

        # Target for backprop
        target_class_one_hot = torch.FloatTensor(1, 1000).zero_()
        target_class_one_hot[0][target_class] = 1

        # This is where we accumulate the LRP results
        LRP_per_layer = [None] * number_of_layers + [(forward_output[-1] * target_class_one_hot).data]

        for layer_index in range(1, number_of_layers)[::-1]:
            # This is where features to classifier change happens
            # Have to flatten the lrp of the next layer to match the dimensions
            if layer_index == features_to_classifier_loc-1:
                LRP_per_layer[layer_index+1] = LRP_per_layer[layer_index+1].reshape(feature_to_class_shape)

            if isinstance(layers_in_model[layer_index], (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MaxPool2d)):
                # In the paper implementation, they replace maxpool with avgpool because of certain properties
                # I didn't want to modify the model like the original implementation but
                # feel free to modify this part according to your need(s)
                lrp_this_layer = self.LRP_step(forward_output[layer_index], layers_in_model[layer_index], LRP_per_layer[layer_index+1])
                LRP_per_layer[layer_index] = lrp_this_layer
            else:
                LRP_per_layer[layer_index] = LRP_per_layer[layer_index+1]
        return LRP_per_layer


if __name__ == '__main__':
    # Get params
    target_example = 2  # Spider
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)

    # LRP
    layerwise_relevance = LRP(pretrained_model)

    # Generate visualization(s)
    LRP_per_layer = layerwise_relevance.generate(prep_img, target_class)

    # Convert the output nicely, selecting the first layer
    lrp_to_vis = np.array(LRP_per_layer[1][0]).sum(axis=0)
    lrp_to_vis = np.array(Image.fromarray(lrp_to_vis).resize((prep_img.shape[2],
                          prep_img.shape[3]), Image.ANTIALIAS))

    # Apply heatmap and save
    heatmap = apply_heatmap(lrp_to_vis, 4, 4)
    heatmap.figure.savefig('../results/LRP_out.png')
