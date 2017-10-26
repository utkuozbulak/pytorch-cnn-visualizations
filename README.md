# Convolutional Neural Network Visualizations 

This repo contains following CNN operations implemented in Pytorch: 

* Vanilla Backpropagation
* Guided Backpropagation [1]
* Gradient-weighted [3] Class Activation Mapping [2] 
* Guided Gradient-weighted Class Activation Mapping [3]

It will also include following operations in near future as well:

* Image Specific Class Saliency Visualization (A generated image that maximizes a certain class) [4]
* Inverted Image Representations [5]
* Weakly supervised object segmentation [4]
* Semantic Segmentation with Deconvolutions [6]

The code uses pretrained VGG19 in the model zoo. It assumes that the layers in the model are separated into two sections; **features**, which contains the convolutional layers and **classifier**, that contains the fully connected layer (after flatting out convolutions). If you want to port this code to use it on your model that does not have such separation, you just need to do some editing on parts where it calls *model.features* and *model.classifier*.

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

[1] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. *Striving for Simplicity: The All Convolutional Net*, https://arxiv.org/abs/1412.6806

[2] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, A. Torralba. *Learning Deep Features for Discriminative Localization*, https://arxiv.org/abs/1512.04150

[3] R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, https://arxiv.org/abs/1610.02391

[4] K. Simonyan, A. Vedaldi, A. Zisserman. *Deep Inside Convolutional Networks: Visualisng Image Classification Models and Saliency Maps*, https://arxiv.org/abs/1312.6034

[5] A. Mahendran, A. Vedaldi. *Understanding Deep Image Representations by Inverting Them*, https://arxiv.org/abs/1412.0035

[6] H. Noh, S. Hong, B. Han,  *Learning Deconvolution Network for Semantic Segmentation* https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Noh_Learning_Deconvolution_Network_ICCV_2015_paper.pdf
