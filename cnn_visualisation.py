import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import numpy as np
from collections import OrderedDict
from torch.nn import ReLU
from PIL import Image

class CamExtractor():
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        print(self.target_layer)
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x
        return conv_output, x

    def forward_pass(self, x):
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.model.classifier(x)
        return conv_output, x


class GradCam_v2:
    def __init__(self, model, target_layer):
        print(target_layer)
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_index=None):
        # Forward pass
        conv_outputs, model_output = self.extractor.forward_pass(input_image)
        if target_index is None:
            target_index = np.argmax(model_output.data.numpy())
        # Target for guided backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_index] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_outputs.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        # Normalize between 0 - 1
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        return cam

def PIL_process_image(PIL_img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    PIL_img = PIL_img.resize((224, 224), Image.ANTIALIAS)
    im_as_arr = np.array(PIL_img, dtype=np.float)
    im_as_arr = im_as_arr.transpose(2, 0, 1)
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    im_as_ten = torch.from_numpy(im_as_arr).float()
    im_as_ten.unsqueeze_(0)
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input


def show_cam_on_image(img, mask):
    cv2.imwrite("vanilla-1.jpg", np.uint8(255 * mask))
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_BONE)
    heatmap = np.float32(heatmap) / 255
    cv2.imwrite("vanilla.jpg", np.uint8(255 * heatmap))
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class Vanilla_Backprop():

    def __init__(self, model, processed_im, layer, target_index):
        self.model = model
        self.input_image = processed_im
        self.target_index = target_index
        self.selected_laye = layer
        self.gradients = []
        self.all_grads = None
        self.hook_gradients()


    def hook_gradients(self):
        def hook_function(module, grad_in, grad_out):
            print('Hooked on the feeling')
            self.all_grads = grad_in[0]

        list(pretrained_model.features._modules.items())[0][1].register_backward_hook(hook_function)

    def run(self):

        model_output = self.model(self.input_image)  # Forward
        self.model.zero_grad()
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][self.target_index] = 1
        model_output.backward(gradient=one_hot_output)
        return self.all_grads


class GBP():

    def __init__(self, model, processed_im, layer, target_index):
        self.model = model
        self.input_image = processed_im
        self.target_index = target_index
        self.selected_laye = layer
        self.gradients = []
        self.all_grads = None
        self.updated_relus()
        self.hook_gradients()


    def hook_gradients(self):
        def hook_function(module, grad_in, grad_out):
            print('Hooked on the feeling')
            self.all_grads = grad_in[0]

        list(pretrained_model.features._modules.items())[0][1].register_backward_hook(hook_function)

    def updated_relus(self):
        def relu_hook_function(module, grad_in, grad_out):
            if isinstance(module, ReLU):
                print('Hooked on the relu')
                return (torch.clamp(grad_in[0], min=0.0),)

        for pos, module in pretrained_model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)



    def run(self):

        model_output = self.model(self.input_image)  # Forward
        self.model.zero_grad()
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][self.target_index] = 1
        model_output.backward(gradient=one_hot_output)
        return self.all_grads




if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization.
    """
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    img = cv2.imread('examples/both.png')
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    prep_im = preprocess_image(img)

    imv2 = Image.open('examples/both.png')
    prep_imv2 = PIL_process_image(imv2)

    pretrained_model=models.vgg19(pretrained=True)

    """
    gcv2 = GradCam_v2(pretrained_model, target_layer = 35)
    m = gcv2.generate_cam(prep_im, 282)
    show_cam_on_image(img, m)
    """

    z = GBP(pretrained_model, prep_im, 0, 999)
    all_grads = z.run()
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