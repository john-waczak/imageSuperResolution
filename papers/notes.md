IDEA: use seam carving (in reverse) to identify the *most important* seem. For each pixel in the seem, apply model to generate 2 pixels. Perform sequence of horizontal and vertical stretches until desired resolution is achieved.

-> still need to figure out how to set up training dataset for this

# Deep Learning for Image Super-Resolution: A Survey 
**Notes** 

## Existing methods 
- early Convolutional Neural Network based methods (e.g. SRCNN) 
- Generative Adversarial Networks (GANs) 

The major differences between methods are: 
- network architecture
- loss function 
- learning principles 

## Problem statement
Super resolution can be modeled as trying to recover a HR image $I_h$ from a low resolution image $I_l$. We model the degradation process of producing $I_l$ from $I_h$ as 
\begin{equation}
    I_l = \mathcal{D}(I_h)
\end{equation}
Where $\mathcal{D}$ represents the degradation operation. $\mathcal{D}$ can be viewed many different ways: There is bicubic downsampling, or we could be more complicated (realistic) and model the process as some convolution with a blurring filter $f_{blurr}$ before downsampling. We can also include a stochastic noise term, $\eta$, to model any noise phenomena that may occur at the image sensor. Thus, 
\begin{equation}
    I_l = \mathcal{D}(I_h * f_{blur})+\eta
\end{equation}
The task of super-resolution is therefore to perform the ill-posed task of inverting this downsampling process, i.e., we seek to find a model $\mathcal{F}$ such that 
\begin{equation} 
    \hat{I}_h = \mathcal{F}(I_l; \theta)
\end{equation}
where $\theta$ are the model parameters.

## Super-resolution Datasets
Some datasets provide both HR and LR images. Others only provide you with the HR data and it is your job to downsample by whatever method you like. A common practice is to use the `resize` function in MATLAB (i.e. bicubic interpolation with anti-aliasing). 

## Image Quality Assessment
Image quality refers to those visual attributes of images that focus on the perceptual assessments of viewers. IQA methods include subjective methods based on human perception (i.e. how *realistic* the image looks). It also includes objective computational methods. These two categories often disagree... 
### Peak Signal-to-Noise Ratio (PSNR) 
One of the most popular quality metrics. It is defined by the maximum pixel value $L$ and the MSE between images. If $I$ is the ground truth image with $N$ pixels and $\hat{I}$ is the reconstructed image, the PSNR is defined as 
\begin{equation}
    PSNR = 10 \cdot \log_{10}\left(  \frac{L^2}{\frac{1}{N}\sum_i(I(i)-\hat{I}(i))^2} \right)
\end{equation}
where $L$ is $255$ in general cases where 8-bit image representations are used. 

### Structural Similiarity Index (SSIM)
The idea of this method is to compare *structural similarity* of images via luminance, contrast, and structures. For an image $I$ with $N$ pixels, the luminance $\mu_I$ and contrast $\sigma_I$ are estimated as the mean and standard deviation of the image intensity. The comparisons on luminance and contrast are denoted $\mathcal{C}_l(I, \hat{I})$ and $\mathcal{C}_c(I,\hat{I})$ respectively, where: 
\begin{equation}
    \mathcal{C}_l(I,\hat{I}) = \frac{2\mu_I\mu_{\hat{I}} + C_1}{\mu_I^2+\mu_{\hat{I}}^2 + C_1} 
\end{equation}
\begin{equation}
    \mathcal{C}_c(I,\hat{I}) = \frac{2\sigma_I\sigma_{\hat{I}} + C_2}{\sigma_I^2+\sigma_{\hat{I}}^2 + C_2} 
\end{equation}
\begin{equation}
    C_1 = (k_1L)^2 \qquad C_2 = (k_2L)^2 \qquad k_1,k_2 << 1
\end{equation}
where $C_1, C_2$ are small constants to avoid numerical instabilities.
Finally, the structure comparison, $\mathcal{C}_s$ is defined via the correlation between the $I$ and $\hat{I}$ images. I.e. 
\begin{equation}
    \sigma_{I\hat{I}} = \frac{1}{N-1}\sum_i\left(I(i)-\mu_I\right)\left(\hat{I}(i)-\mu_{\hat{I}}\right)
\end{equation}
\begin{equation}
    \mathcal{C}_s(I,\hat{I}) = \frac{\sigma_{I\hat{I}}+C_3}{\sigma_I\sigma_{\hat{I}}+C_3}
\end{equation}

Finally, we put all three of these together to compute the SSIM: 
\begin{equation} 
    SSIM(I,\hat{I}) = \left[\mathcal{C}_l\right]^{\alpha}\left[\mathcal{C}_c\right]^{\beta}\left[\mathcal{C}_s\right]^{\gamma}
\end{equation}
where $\alpha, \beta,\gamma$ are parameters to adjust the relative importance of each term.

### Mean opinion score 
This uses human raters to score the reconstructed images.

## Operating Channels 
RGB color space is most commonly used in SR. The YCbCr color space is also used where the channels denote illuminance, blue-difference, and red difference. This is worth noting because the ./

## Super-resolution Challenges 
There are some popular super-resolution community challenges: 
- **NTIRE**: The New Trends in Image Restoration and Enhancement. Built on the DIV2K dataset consisting of bicubic downscaling and blind tracks with unknown degradation. 
- **PIRM**: The Perceptual Image Restoration and Manipulation challenges is another popular challenge that focuses on perceptual quality of SR images. 


# Supervised Super-Resolution 
Most models consist of the same sets of components: model framework, upsampling method, network design, and learning strategy. 
## Super-resolution frameworks 
Super-resolution is a fundamentally ill-posed problem... we seek to get more information from less. This means that the question of how to perform *upsampling* is the key problem. Architectures vary widely, but there are four basic frameworks based on the upsampling model and their location (depth) in the network.

### Pre-upsampling Super-Resolution 
In this strategy, one first upsamples the image to the desired size using traditional methods and then refines the result to match the HR training image. An example of this strategy is SRCNN which seeks to learn the mapping from interpolated LR images to HR images using bicubic interpolation for the upsampling. This is one of the most popular strategies as once the upscaled image has been formed, the model is essentially just a standard CNN. The problem is that the upscaling process can lead to side effects such as amplified noise and blurring. 

### Post-upsampling super-resolution 
This method seeks to improve the computational efficiency by replacing the upscaling process with learnable layers integrated at the end of the models. 

### Progressive Upsampling Super-Resolution 
(this is probably what my seamcarving idea would be good for... We could do it for pre or progressive) 
The problem with the post-upsampling strategy is that because upsampling is performed in the final step, the learning difficulty is greatly increased. One way to ameliorate the situation is to use a progressing framework as in *Laplacian pyramid SR* which are based on a cascade of CNN's that progressively reconstruct the image. Another model using this idea is the *MS-LapSRN*. 

### Iterative Up-and-Down Sampling Super-Resolution 
This class of models incorporates a notion of **back projection**. In this model framework, the reconstruction error is computed and then sent back to the LR image to tune the final HR image. One such model is called DBPN.

## Upsampling Methods 
How to perform the upsampling is just as, if not more important than where the upsampling occurs in the model. There are various traditional methods as well as models designed to learn the end-to-end upsampling. 

### Interpolation-based Upsampling 
Image interpolation refers to digitally resizing images. Traditional methods include nearest naeighbor interpolation, bilinear, and bicubic interpolations, Sinc and Laczos resampling, etc... 

**Nearest-neighbor** Interpolation: The nearest-neighbor interpolation is a simple algorithm. It slects the value of the nearest pixel for each position to be interpolated regardless of any other pixels. This is very fast but results in low quality, blocky images. 

**Bilinear interpolation**: This performs a fast linear interpolationm on one axis of the image and then does the same to the other. The result is a quadratic interpolation that is still performant but tends to do better than nearest neighbor. 

**Bicubic Interpolation**: This is the same idea as bilinear but instead performing a cubic interpolation on each axis. This results in smoother images with fewer artifacts but at a lower speed.

The current trend is to replace interpolation by learnable upsampling layers. 

### Learning-based Upsampling 
There are two layers introduce to the DL framework to allow learnable upsampling: Transposed Convolutional Layer, and the Sub-pixel Layer. 

**Transposed  Convolutional Layer**: (aka *Deconvolution* layer) tries to perform a transformation opposite to a normal convolution, i.e. predicting the possible input based on feature maps sized like convolution output. These specifically increase the size of an image by inserting zeros around each pixel and then performing a convolution. 
**(IDEA: why not pad with 1's around each pixel? Is there a reason 0's are preferred?)** 
These layers can lead to a checkerboard like effect as the magnitudes of each sub region are not varied uniformly. 

**Sub-pixel layer**: This layer performs upsampling by generating multiple channels via a convolution and then reshaping them into the new pixels. First a convolution is performed that maps 
\begin{equation}
    h\times w\times c \to h\times w\times s^2c
\end{equation}
where $s$ is the scaling factor. After that, a rescaling operation (a shuffling) occurs which maps 
\begin{equation}
    h\times w\times s^2c \to sh \times sw \times c
\end{equation}
This layer has a larger **receptive field** than the TCL which means more contextual information is provided to the layer.

## Network Design
### Residual Learning 
ResNet is a commonly used architecture for learning residuals. Residual learning networks can be subdivided into two subcategories: **Global residual learning**, and **local residual learning**. For the global case, only the residuals between the SR and target image are learned. This avoids learning complicated transforms from one image to another so that only a high-frequency detail map need be learned. The local residual learning structure is used to address the degradation problem caused by increasing network depths. 

The above methods are applied in practice by **shortcut connections**. 

### Recursive Learning 
In recursive learning, the same modules are applied multiple times in a recursive manner (i.e. successive layers are coppies with shared weights). The idea is that this will reduce the number of learnable parameters. (This is probably ideal for the seam-carving idea. i.e. freeze the layers and generate copies that stretch horizontal and vertical axes until desired scaling is reached). Some examples of this strategy are: 
- 16-recursive DRCN
- DRRN
- MemNet (Tai et al)
- CARN (cascading residual network) 
- DSRN (dual-state recurrent network)

### Multi-path Learning 
This method refers to passing features through different paths which each perform different operations and then fusing them back together. This category can be subdivided into **global**, **local**, and **scale-specific**. 

### Dense Connections 

### Attention Mechanisms 

### Advanced Convolution
**Dilated Convolutions**: the goal of this type of convolution is to increase the receptive field in order to increase the contextual awareness. 

**Group Convolution**: The idea of group convolution is to reduce the number of parameters and operations.

**Depthwise Separable Convolution**: This includes factorizing convolutions and point-wise convolutions to improve efficiency

## Loss Functions 
- Pixel-wise mse 
- 
