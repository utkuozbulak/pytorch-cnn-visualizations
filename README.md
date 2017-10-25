<p align="center"><img width="40%" src="https://raw.githubusercontent.com/utkuozbulak/pytorch-custom-dataset-examples/master/data/pytorch-logo-dark.png" /></p>

--------------------------------------------------------------------------------
# Convolutional Neural Network Visualizations 

This repo contains following CNN visualizations implemented in Pytorch: 

* Vanilla Backpropagation
* Guided Backpropagation
* Gradient-weighted Class Activation Mapping
* Guided Gradient-weighted Class Activation Mapping

The code uses VGG19 as the model. It assumes the model is logically separated into two sections; **features**, which contains the convolutional layers and **classifier**, which contains the fully connected layer (after flatting out convolutions). 

<table border=0 >
	<tbody>
    <tr>
			<td>  </td>
			<td align="center"> Target class: Shepherd (235) </td>
			<td align="center"> Target class: Mastiff (243) </td>
			<td align="center"> Target class: Spider (72)</td>
		</tr>
		<tr>
			<td width="19%" align="center"> Original Image </td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/examples/dog_car.png"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/examples/cat_dog.png"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/examples/spider.png"> </td>
		</tr>
		<tr>
			<td width="19%" align="center"> Colored Vanilla Backpropagation </td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/dog_car_Vanilla_BP_color.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/cat_dog_Vanilla_BP_color.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/spider_Vanilla_BP_color.jpg"> </td>
		</tr>
			<td width="19%" align="center"> Grayscale Vanilla Backpropagation </td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/dog_car_Vanilla_BP_gray.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/cat_dog_Vanilla_BP_gray.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/spider_Vanilla_BP_gray.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center"> Colored Vanilla Backpropagation <br />  <br />  (GB)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/dog_car_Guided_BP_color.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/cat_dog_Guided_BP_color.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/spider_Guided_BP_color.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center"> Grayscale Guided Backpropagation <br />  <br /> (GB)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/dog_car_Guided_BP_gray.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/cat_dog_Guided_BP_gray.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/spider_Guided_BP_gray.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center"> Gradient-weighted Class Activation Map <br />  <br /> (Grad-CAM)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/dog_car_Cam_Grayscale.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/cat_dog_Cam_Grayscale.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/spider_Cam_Grayscale.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center"> Gradient-weighted Class Activation Heatmap <br />  <br /> (Grad-CAM)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/dog_car_Cam_Heatmap.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/cat_dog_Cam_Heatmap.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/spider_Cam_Heatmap.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center"> Gradient-weighted Class Activation Heatmap on Image <br />  <br /> (Grad-CAM)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/dog_car_Cam_On_Image.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/cat_dog_Cam_On_Image.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/spider_Cam_On_Image.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center"> Colored Guided Gradient-weighted Class Activation Map <br />  <br /> (Guided-Grad-CAM)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/dog_car_GGrad_Cam.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/cat_dog_GGrad_Cam.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/spider_GGrad_Cam.jpg"> </td>
		</tr>
    <tr>
			<td width="19%" align="center"> Grayscale Guided Gradient-weighted Class Activation Map  <br />  <br /> (Guided-Grad-CAM)</td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/dog_car_GGrad_Cam_gray.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/cat_dog_GGrad_Cam_gray.jpg"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualization/master/results/spider_GGrad_Cam_gray.jpg"> </td>
		</tr>
	</tbody>
</table>

