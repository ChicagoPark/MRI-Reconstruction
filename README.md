# CIGProject

##### Project Topic:  Deep `Supervised Learning` for `MRI Reconstruction`

##### Project Supervisor: `Weijie Gan`

----

## Problem Analysis

#### [1] `Terminology Definition`
* P: Sampling operator
* F: Fourier transform
* x: input image
* e: noise vector



#### [2] `Dataset`

> Use trnOrg and trnMask for training only. Divide tstOrg and tstMast into the Validation set and Test set.

| Training set  | Validation set | Test set  |
| :---:         |     :---:      |    :---:  |
| 360 frames    | 105 frames     | 59 frames |
| 69 %          | 20 %           | 11 %      |


#### [3] Data Preprocessing

> (1) In supervised learning-based reconstruction task, it is necessary to `pair input and output dataset`.
> 
> (2) As we have `ground-truth` MRI, we will give deformation through Fourier Transform and Inverse Fourier Transform to the ground-truth for matched output data. We will call that data `X hat`.


#### [4] Model
> This project utilizes CNN with Residual structure for the deep network and overcomes vanishing gradient problems.



| Detail implementation  | Loss function module|
|          :---:         |          :---:      |
|       Project.ipynb    |         test.py     |



<!--
How to get zero-filled images

Zero-Fill
Zero-Filling is the process of adding data points to the end of the FID before Fourier Transform.

Zero-Filling can improve data quality by increasing the amount of points per ppm in the processed data to better define the resonance. The added data points have 0 amplitude so the only change of the processed data is more discreet data points.

The Zero-Fill menu is located under the Processing tab then Zero-Fill/LP. LP is Linear Prediction which is discussed here:

-->

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


## Realization


