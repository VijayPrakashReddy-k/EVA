from albumentations import (
	Compose,
    HorizontalFlip,
    Normalize,
    Cutout,
    CoarseDropout,
    Rotate,
    GaussianBlur,
    HueSaturationValue
)
from albumentations.pytorch import ToTensor
import numpy as np	

def albumentations_transforms(p=1.0, is_train=False):
	# Mean and standard deviation of train dataset
	mean = np.array([0.4914, 0.4822, 0.4465])
	std = np.array([0.2023, 0.1994, 0.2010])
	transforms_list = []
	# Use data aug only for train data
	if is_train:
		transforms_list.extend([
			HueSaturationValue(p=0.25),
			HorizontalFlip(p=0.5),
			Rotate(limit=15),
			CoarseDropout(max_holes=1, max_height=16, max_width=16, min_height=4,
						min_width=4, fill_value=mean*255.0, p=0.75),
			Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=0, always_apply=False, p=0.5),
		    Normalize(
			mean=mean,
			std=std,
			max_pixel_value=255.0,
			p=1.0),
		    ToTensor()
		])
	transforms = Compose(transforms_list, p=p)
	return lambda img:transforms(image=np.array(img))["image"]