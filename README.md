# Convolutional Neural Network Visualizations 

This repo contains following CNN operations implemented in Pytorch: 

* Vanilla Backpropagation
* Guided Backpropagation [1]
* Gradient-weighted [3] Class Activation Mapping [2] 
* Guided Gradient-weighted Class Activation Mapping [3]
* Class Specific Image Generation (A generated image that maximizes a certain class) [4]

It will also include following operations in near future as well:

* Inverted Image Representations [5]
* Weakly supervised object segmentation [4]
* Semantic Segmentation with Deconvolutions [6]

The code uses pretrained VGG19, VGG16 and AlexNet in the model zoo. Some of the code assumes that the layers in the model are separated into two sections; **features**, which contains the convolutional layers and **classifier**, that contains the fully connected layer (after flatting out convolutions). If you want to port this code to use it on your model that does not have such separation, you just need to do some editing on parts where it calls *model.features* and *model.classifier*.

All images are pre-processed with mean and std of the ImageNet dataset before being fed to the model.

I tried to comment on the code as much as possible, if you have any issues understanding it or porting it, don't hesitate to reach out. 

Below, are some sample results for each operation.

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

## Class Specific Image Generation
This operation produces different outputs based on the model and the applied regularization method. Below, are some samples produced with L2 regularization from VGG19.

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

To observe the differnece in output with the applied regularizatin method, the samples below show the produced image with no regularization, l2 and l2 on target class _flamigo_ (130).

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

Produced samples can further be optimized to resemble the desired target class, some of the operations you can incorporate are; blurring, clipping gradients that are below a treshold, random color swaps on some parts, random cropping the image.

## Requirements:
```
torch >= 0.2.0.post4
torchvision >= 0.1.9
numpy >= 1.13.0
opencv >= 3.1.0
```


[1] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. *Striving for Simplicity: The All Convolutional Net*, https://arxiv.org/abs/1412.6806

[2] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, A. Torralba. *Learning Deep Features for Discriminative Localization*, https://arxiv.org/abs/1512.04150

[3] R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, https://arxiv.org/abs/1610.02391

[4] K. Simonyan, A. Vedaldi, A. Zisserman. *Deep Inside Convolutional Networks: Visualisng Image Classification Models and Saliency Maps*, https://arxiv.org/abs/1312.6034

[5] A. Mahendran, A. Vedaldi. *Understanding Deep Image Representations by Inverting Them*, https://arxiv.org/abs/1412.0035

[6] H. Noh, S. Hong, B. Han,  *Learning Deconvolution Network for Semantic Segmentation* https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Noh_Learning_Deconvolution_Network_ICCV_2015_paper.pdf
