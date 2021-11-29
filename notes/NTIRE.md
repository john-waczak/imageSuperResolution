
# NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study
authors: *Eirikur Agustsson*, *Radu Timofte*
[paper link](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Agustsson_NTIRE_2017_Challenge_CVPR_2017_paper.pdf)

## Jargon
- **SR**: super-resolution
- **LR**: low-resolution 
- **HR**: high-resolution
- **Perceptual image super-resolution**
- **Residual Network**, i.e. *ResNet*
- **Haar Wavelet Residual Network**
- **BM3D**
- **Weiner Filter**


## Datasets
- DIVerse 2k (DIV2k) 
- Set5
- Set14
- B100 
- Urban100 
- Train91 (Yang et al)  

## Notes
- Most common downscaling method are *bicubic downscaling* via `imresize()` in MATLAB and so called *unkown downscaling operators*
- Authors propose new dataset, **DIV2k**, with 1000 training images
- Most common magnification factors today: $\times 2$, $\times 3$, and $\times 4$
- "In practice, however, often the ground truth is not available and, therefore, plausible and perceptually qualitative super-resolved images are desirable as long as the information from the LR image is preserved""


## Image Metrics
- Image entropy 
- bit per pixel (bpp) PNG compression rates
- CORNIA (Codebook Representation for No-Reference Image Assessment)
- PSNR (peak signal to noise ratio)
- MSE (mean square error)
- SSIM (structural similarity index)
- IFC (information fidelity criterion)


## SR Architectures 
- **SRCNN** (Dong et al)
- **SSResNet***
- **SNU_CVLab1** (Lim et al)
- **SNU_CVLab2** 
- **HelloSR** winner of NTIRE 2017 based on stacked residual-refined network
- **Lab402** (Bae et al) 41 layers of Haar Wavelet Residual network
- **WSDSR** self-similarity method based on BM3D and Weiner filter (Cruz et al)
- **A+** or Adjusted ANR of Timofte et al 


## Data Augmentation / Ensemble Prediction
- The top ranked entries SNU CVLab1, SNU CVLab2, HelloSR, UIUC-IFP and ‘I hate mosaic’ in the NTIRE 2017 SR Challenge use ensembles of HR predictions.
-  flipping and/or rotating by $90^\circ$ the LR input image then process them to achieve HR corresponding results and align these results back for the final HR average result
- Learn the degredation operator 
