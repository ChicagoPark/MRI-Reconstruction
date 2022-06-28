# CIGProject

##### Project Topic:  Deep `Supervised Learning` for `MRI Reconstruction`

##### Project Supervisor: `Weijie Gan`

----

## Study
1. Fourier Transform Shift
2. Fourier Inverse Transform Shift
3. Complex training
4. Read more review paper of MRI Reconstruction 

----
## Priority

1. Utilize Complex Value

2. Visualize the trained result

3. Normalize the input of PSNR and SSIM

----

## Problem Analysis

#### [1] `Terminology Definition`
* P: Sampling operator
* F: Fourier transform
* x: input image
* e: noise vector



#### [2] `Dataset`

> Use trnOrg and trnMask for training only. Divide tstOrg and tstMast into the Validation set and Test set.

|      `Training set`  |         `Validation set` |      `Test set`  |
| :---:         |     :---:      |    :---:  |
| 360 frames    | 105 frames     | 59 frames |
| 69 %          | 20 %           | 11 %      |


#### [3] Data Preprocessing

> (1) In supervised learning-based reconstruction task, it is necessary to `pair input and output dataset`.
> 
> (2) As we have `ground-truth` MRI, we will match zero-filled images to the ground-truth. We will call that input data `X hat`.


#### [4] Model
> This project utilizes CNN with Residual structure for the deep network and overcomes vanishing gradient problems.



| Detail implementation  | Loss function module|
|          :---:         |          :---:      |
|       Project.ipynb    |         test.py     |


----


## Step-by-Step Instructions

### [1] Visualize x hat

> <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/175949666-dd277fc1-9e96-4508-932c-935fb9f77cb0.png">

<!--
> <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/175498260-77506634-dc1c-4194-b86b-f4d9671f3bfb.png">
-->

### [2] Performance evaluation by PSNR and SSIM

> <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/176088132-40298521-2ac6-4d56-aaf8-074adcdaaacd.png">

### [3] Visualizing test result

> <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/176088171-faab6e31-fbaa-45dd-b8e3-e25a53409e40.png">

> <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/176088176-55728779-2b8c-4ceb-bcfc-cf8c070d2596.png">

> <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/176088180-c8b486c0-e030-47d5-a804-11e50b429c38.png">


## Review

#### 1. Study (inverse) Fourier Transform and shift

#### 2. Analyze MRI Data such as K-space and so on.




<!--
How to get zero-filled images

Zero-Fill
Zero-Filling is the process of adding data points to the end of the FID before Fourier Transform.

Zero-Filling can improve data quality by increasing the amount of points per ppm in the processed data to better define the resonance. The added data points have 0 amplitude so the only change of the processed data is more discreet data points.

The Zero-Fill menu is located under the Processing tab then Zero-Fill/LP. LP is Linear Prediction which is discussed here:

[Things to do]
1. Define Loss function (Done)
2. Construct overall model architecture (including normalization)
3. Study how to implement ResNet from scratch
4. Plot grid
5. Separate dataset (Train / Validation / Test)


### Encountered Error - 1

RuntimeError: expected scalar type Double but found Float

Solution: use `.float()` when we transfer from numpy to torch tensor

```python
noisy_torch = torch.from_numpy(xHat).float()
```

conv2d() received an invalid combination of arguments

Solution: padding was floating point by mistake.



-->




