# Convolutional Neural Network Visualizations 

This repo contains following CNN operations implemented in Pytorch: 

* Gradient visualization with vanilla backpropagation
* Gradient visualization with guided backpropagation [1]
* Gradient visualization with saliency maps [4]
* Gradient-weighted [3] class activation mapping [2] 
* Guided gradient-weighted class activation mapping [3]
* CNN filter visualization [9]
* Deep Dream [10]
* Class specific image generation (A generated image that maximizes a certain class) [4]
* Fooling images (Unrecognizable images predicted as classes with high confidence) [7]
* Fooling images disguised as another image (Picture of ipod being predicted as horse) [7]

It will also include following operations in near future as well:


* Inverted Image Representations [5]
* Weakly supervised object segmentation [4]
* Semantic Segmentation with Deconvolutions [6]
* Smooth Grad [8]

The code uses pretrained VGG19, VGG16 and AlexNet in the model zoo. Some of the code assumes that the layers in the model are separated into two sections; **features**, which contains the convolutional layers and **classifier**, that contains the fully connected layer (after flatting out convolutions). If you want to port this code to use it on your model that does not have such separation, you just need to do some editing on parts where it calls *model.features* and *model.classifier*.

All images are pre-processed with mean and std of the ImageNet dataset before being fed to the model. None of the code uses GPU as these operations are quite fast (for a single image). You can make use of gpu with very little effort. The examples below include numbers in the brackets after the description, like *Mastiff (243)*, this number represents the class id in the ImageNet dataset.

I tried to comment on the code as much as possible, if you have any issues understanding it or porting it, don't hesitate to reach out. 

Below, are some sample results for each operation.


## Gradient Visualization and Segmentation
<table border=0 >
	<tbody>
    <tr>
			<td>  </td>
			<td align="center"> Target class: King Snake (56) </td>
			<td align="center"> Target class: Mastiff (243) </td>
			<td align="center"> Target class: Spider (72)</td>
		</tr>
		<tr>
			<td width="19%" align="center"> Original Image </td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/snake.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/cat_dog.png"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/spider.png"> </td>
		</tr>
		<tr>
			<td width="19%" align="center"> Colored Vanilla Backpropagation </td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/snake_Vanilla_BP_color.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/cat_dog_Vanilla_BP_color.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/spider_Vanilla_BP_color.jpg"> </td>
		</tr>
			<td width="19%" align="center"> Vanilla Backpropagation Saliency </td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/snake_Vanilla_BP_gray.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/cat_dog_Vanilla_BP_gray.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/spider_Vanilla_BP_gray.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center"> Colored Guided Backpropagation <br />  <br />  (GB)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/snake_Guided_BP_color.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/cat_dog_Guided_BP_color.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/spider_Guided_BP_color.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center">Guided Backpropagation Saliency<br />  <br /> (GB)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/snake_Guided_BP_gray.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/cat_dog_Guided_BP_gray.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/spider_Guided_BP_gray.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center">Guided Backpropagation Negative Saliency<br />  <br /> (GB)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/snake_neg_sal.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/cat_dog_neg_sal.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/spider_neg_sal.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center">Guided Backpropagation Positive Saliency<br />  <br /> (GB)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/snake_pos_sal.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/cat_dog_pos_sal.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/spider_pos_sal.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center"> Gradient-weighted Class Activation Map <br />  <br /> (Grad-CAM)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/snake_Cam_Grayscale.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/cat_dog_Cam_Grayscale.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/spider_Cam_Grayscale.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center"> Gradient-weighted Class Activation Heatmap <br />  <br /> (Grad-CAM)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/snake_Cam_Heatmap.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/cat_dog_Cam_Heatmap.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/spider_Cam_Heatmap.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center"> Gradient-weighted Class Activation Heatmap on Image <br />  <br /> (Grad-CAM)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/snake_Cam_On_Image.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/cat_dog_Cam_On_Image.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/spider_Cam_On_Image.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center"> Colored Guided Gradient-weighted Class Activation Map <br />  <br /> (Guided-Grad-CAM)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/snake_GGrad_Cam.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/cat_dog_GGrad_Cam.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/spider_GGrad_Cam.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center"> Guided Gradient-weighted Class Activation Map Saliency <br />  <br /> (Guided-Grad-CAM)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/snake_GGrad_Cam_gray.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/cat_dog_GGrad_Cam_gray.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/spider_GGrad_Cam_gray.jpg"> </td>
		</tr>
	</tbody>
</table>

## Convolutional Neural Network Filter Visualization
CNN filters can be visualized when we optimize the input image with respect to output of the specific convolution operation. For this example I used a pre-trained **VGG16**. Visualizations of layers start with basic color and direction filters at lower levels. As we approach towards the final layer the complexity of the filters also increases. If you employ techniques like blurring, gradient clipping etc. you will probably produce better images.

<table border=0 width="50px" >
	<tbody> 
		<tr>
			<td width="19%" align="center"> Layer 2 <br /> (Conv 1-2)</td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l2_f1.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l2_f21.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l2_f54.jpg"> </td>
		</tr>
		<tr>
			<td width="19%" align="center"> Layer 10 <br /> (Conv 2-1)</td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l10_f7.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l10_f10.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l10_f69.jpg"> </td>
		</tr>
		<tr>
			<td width="19%" align="center"> Layer 17 <br /> (Conv 3-1)</td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l17_f4.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l17_f8.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l17_f9.jpg"> </td>
		</tr>
		<tr>
			<td width="19%" align="center"> Layer 24 <br /> (Conv 4-1)</td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l24_f4.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l24_f17.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l24_f22.jpg"> </td>
		</tr>
	</tbody>
</table>


## Deep Dream
Deep dream is technically the same operation as layer visualization the only difference is that you don't start with a random image but use another picture. The samples below were created with VGG19, the produced result is entirely up to the filter so it is kind of hit or miss. The more complex models produce mode high level features, meaning that If you replace VGG19 with an Inception variant you will get more noticable shapes when you target higher conv layers. Like layer visualization, if you employ additional techniques like gradient clipping, blurring etc. you might get better visualizations.

<table border=0 width="50px" >
	<tbody>
		<tr>
			<td width="19%" align="center">Original Image</td>
			<td width="70%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/dd_tree.jpg"> </td>
		</tr>
		<tr>
			<td width="19%" align="center">VGG19 <br /> Layer: 34  <br /> (Final Conv. Layer) Filter: 94</td>
			<td width="70%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/dd_l34_f94_iter250.jpg"> </td>
		</tr>
		<tr>
			<td width="19%" align="center">VGG19 <br /> Layer: 34  <br /> (Final Conv. Layer) Filter: 103</td>
			<td width="70%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/dd_l34_f103_iter250.jpg"> </td>
		</tr>
	</tbody>
</table>


## Class Specific Image Generation
This operation produces different outputs based on the model and the applied regularization method. Below, are some samples produced with L2 regularization from VGG19. Note that these images are generated with regular CNNs with optimizing the input (rather than the model weights) and not with GANs.

<table border=0 width="50px" >
	<tbody>
    <tr>
			<td width="27%" align="center"> Target class: Worm Snake (52) - (VGG19) </td>
			<td width="27%" align="center"> Target class: Spider (72) - (VGG19) </td>
		</tr>
		<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/snake.gif"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/spider.gif"> </td>
		</tr>
	</tbody>
</table>

The samples below show the produced image with no regularization, l1 and l2 regularizations on target class: **flamingo** (130) to show the differences between regularization methods. These images are generated with a pretrained AlexNet. 

<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> No Regularization </td>
			<td width="27%" align="center"> L1 Regularization </td>
			<td width="27%" align="center"> L2 Regularization </td>
		</tr>
		<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/flamingo_no_norm.gif"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/flamingo_l1_norm.gif"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/flamingo_l2_norm.gif"> </td>
		</tr>
	</tbody>
</table>

Produced samples can further be optimized to resemble the desired target class, some of the operations you can incorporate to improve quality are; blurring, clipping gradients that are below a certain treshold, random color swaps on some parts, random cropping the image, forcing generated image to follow a path to force continuity.

## Fooling Image Generation
This operation is quite similar to generating class specific images, we start with a random image and continously update the image with targeted backpropagation (for a certain class) and stop when we achieve target confidence for that class. All of the below images are generated from pretrained AlexNet to fool it.


<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> Predicted as <strong>Zebra</strong> (340) <br/> Confidence: 0.94 </td>
			<td width="27%" align="center"> Predicted as <strong>Bow tie</strong> (457) <br/> Confidence: 0.95 </td>
			<td width="27%" align="center"> Predicted as <strong>Castle</strong> (483) <br/> Confidence: 0.99 </td>
		</tr>
		<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/fooling_sample_class_340.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/fooling_sample_class_457.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/fooling_sample_class_483.jpg"> </td>
		</tr>
	</tbody>
</table>


## Disguised Fooling Image Generation
For this operation we start with an image and perform gradient updates on the image for a specific class but with smaller learning rates so that the original image does not change too much. As it can be seen from samples, on some images it is almost impossible to recognize the difference between two images but on others it can clearly be observed that something is wrong. All of the examples below were created from and tested on AlexNet to fool it.


<table border=0 width="50px" >
	<tbody> 
		<tr>		<td width="27%" align="center"> Predicted as <strong>Eel</strong> (390) <br/> Confidence: 0.96 </td>
			<td width="27%" align="center"> Predicted as <strong>Apple</strong> (948) <br/> Confidence: 0.95 </td>
			<td width="27%" align="center"> Predicted as <strong>Snowbird</strong> (13) <br/> Confidence: 0.99 </td>
		</tr>
		<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/eel.JPEG"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/apple.JPEG"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/bird.JPEG"> </td>
		</tr>
		<tr>		<td width="27%" align="center"> Predicted as <strong>Banjo</strong> (420) <br/> Confidence: 0.99 </td>
			<td width="27%" align="center"> Predicted as <strong>Abacus</strong> (457) <br/> Confidence: 0.99 </td>
			<td width="27%" align="center"> Predicted as <strong>Dumbell</strong> (543) <br/> Confidence: 1 </td>
		</tr>
		<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/fooling_sample_class_420.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/fooling_sample_class_398.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/fooling_sample_class_543.jpg"> </td>
		</tr>
	</tbody>
</table>




## Requirements:
```
torch >= 0.2.0.post4
torchvision >= 0.1.9
numpy >= 1.13.0
opencv >= 3.1.0
```


## References:

[1] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. *Striving for Simplicity: The All Convolutional Net*, https://arxiv.org/abs/1412.6806

[2] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, A. Torralba. *Learning Deep Features for Discriminative Localization*, https://arxiv.org/abs/1512.04150

[3] R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, https://arxiv.org/abs/1610.02391

[4] K. Simonyan, A. Vedaldi, A. Zisserman. *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps*, https://arxiv.org/abs/1312.6034

[5] A. Mahendran, A. Vedaldi. *Understanding Deep Image Representations by Inverting Them*, https://arxiv.org/abs/1412.0035

[6] H. Noh, S. Hong, B. Han,  *Learning Deconvolution Network for Semantic Segmentation* https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Noh_Learning_Deconvolution_Network_ICCV_2015_paper.pdf

[7] A. Nguyen, J. Yosinski, J. Clune.  *Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable  Images* https://arxiv.org/abs/1412.1897

[8] D. Smilkov, N. Thorat, N. Kim, F. Viégas, M. Wattenberg. *SmoothGrad: removing noise by adding noise* https://arxiv.org/abs/1706.03825

[9] MD. Zeiler, R. Fergus. *Visualizing and Understanding Convolutional Networks* https://arxiv.org/abs/1311.2901

[10] A. Mordvintsev, C. Olah, M. Tyka. *Inceptionism: Going Deeper into Neural Networks* https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html
