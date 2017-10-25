import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2  # Only needed for GradCam heatmap
import numpy as np
from collections import OrderedDict
from torch.nn import ReLU
from PIL import Image
from scipy.misc import imresize
import os

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        print('Hooked on the feeling!')
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_index=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_index is None:
            target_index = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_index] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = imresize(cam, (224, 224))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam




class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation of the image
    """
    def __init__(self, model, processed_im, target_class):
        self.model = model
        self.input_image = processed_im
        self.target_class = target_class
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            print('Hooked on the feeling!')
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(pretrained_model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def run(self):
        # Forward
        model_output = self.model(self.input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][self.target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation of the given image
    """
    def __init__(self, model, processed_im, target_index):
        self.model = model
        self.input_image = processed_im
        self.target_index = target_index
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            print('Hooked on the feeling!')
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(pretrained_model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for pos, module in pretrained_model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def run(self):
        # Forward pass
        model_output = self.model(self.input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][self.target_index] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


def save_gradient_pictures(gradient, file_name, org_img):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if org_img:
        org_grad = gradient - gradient.min()
    else:
        org_grad = gradient + gradient.min()
    org_grad /= org_grad.max()
    org_grad = np.uint8(org_grad * 255).transpose(1, 2, 0)
    path_to_file = os.path.join('results', file_name + '.jpg')
    cv2.imwrite(path_to_file, org_grad)


def save_class_activation_on_image(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    # Grayscale activation map
    path_to_file = os.path.join('results', file_name+'_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    path_to_file = os.path.join('results', file_name+'_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('results', file_name+'_Cam_OnImage.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))


def preprocess_image(cv2im):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process

    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor

    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    im_as_arr = np.float32(cv2.resize(cv2im, (224, 224)))
    im_as_arr = im_as_arr[..., ::-1]  # Convert BGR to RGB
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


if __name__ == '__main__':
    img_path = 'examples/both.png'
    file_name = 'my_cat'
    image_class = 282
    # Read image
    cv2im = cv2.imread(img_path, 1)
    # Process image
    #prep_im = preprocess_image(cv2im)
    # Load model

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    im_as_arr = np.float64(cv2.resize(cv2im, (224, 224)))
    im_as_arr = im_as_arr[..., ::-1]  # Convert BGR to RGB
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels

    for channel, _ in enumerate(im_as_arr):
        print('a')
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]

    # Convert to float tensor

    PIL_img = Image.open(img_path)
    PIL_img = PIL_img.resize((224, 224), Image.ANTIALIAS)
    # Convert to np array
    im_as_arr2 = np.array(PIL_img, dtype=np.float64)
    # Transpose to obtain D-W-H
    im_as_arr2 = im_as_arr2.transpose(2, 0, 1)
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        print(channel)
        im_as_arr2[channel] /= 255
        im_as_arr2[channel] -= mean[channel]
        im_as_arr2[channel] /= std[channel]
    im_as_ten = torch.from_numpy(im_as_arr2).float()


    """

    pretrained_model=models.vgg19(pretrained=True)

    gcv2 = GradCam(pretrained_model, target_layer = 35)
    m = gcv2.generate_cam(prep_im, image_class)
    save_class_activation_on_image(cv2im, m, file_name)


    VBP = VanillaBackprop(pretrained_model, prep_im, image_class)
    vanilla_grads = VBP.run()
    save_gradient_pictures(vanilla_grads,file_name + '_org_Vanilla_BP', True)


    GBP = GuidedBackprop(pretrained_model, prep_im, image_class)
    guided_grads = GBP.run()
    save_gradient_pictures(guided_grads, file_name + '_org_Guided_BP', True)
    save_gradient_pictures(guided_grads, file_name + '_enhanced_Guided_BP', False)
    """


    """
    z = GBP(pretrained_model, prep_im, 999)
    all_grads = z.run()ü

    """
    """

    all_grads = all_grads.data.numpy()[0]
    all_grads -= all_grads.min()
    all_grads /= all_grads.max()
    #all_grads = np.maximum(all_grads, 0)
    # Normalize between 0 - 1
    #all_grads = (all_grads - np.min(all_grads)) / (np.max(all_grads) - np.min(all_grads))
    all_grads = all_grads.transpose(1,2,0)
    show_cam_on_image(img, all_grads)
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    """

    """
    gb_model = GuidedBackpropReLUModel(model=models.vgg19(pretrained=True),
                                       use_cuda=False)

    gb = gb_model(prep_im, index=282)
    utils.save_image(torch.from_numpy(gb), 'gb.jpg')
    """

    """
    cam_mask = np.zeros(gb.shape)
    for i in range(0, gb.shape[0]):
        cam_mask[i, :, :] = mask

    cam_gb = np.multiply(cam_mask, gb)
    utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
    """


"""
class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1),
            positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()

    # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        output = self.forward(input)

        if index == None:
            index = np.argmax(output.data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(gradient=one_hot, retain_variables=True)

        output = input.grad.data.numpy()
        output = output[0,:,:,:]

        return output
"""