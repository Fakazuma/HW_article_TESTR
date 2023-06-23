import albumentations as A

# TODO: place in config
MAX_SIZE = 1024
BBOX_FORMAT = 'coco'
KEYPOINT_FORMAT = 'xy'

preproc_transforms = [A.LongestMaxSize(MAX_SIZE),
                      ]

main_transforms = [A.InvertImg(p=0.1),
                   A.Perspective(p=0.1, scale=(0.01, 0.05)),
                   A.OneOf([
                       A.RandomBrightnessContrast(p=1),
                       A.RGBShift(p=1),
                       A.ChannelShuffle(p=1),
                   ], p=0.2),
                   A.HueSaturationValue(p=0.2),
                   A.OneOf([
                       A.GaussNoise(p=1),
                       A.ISONoise(p=1),
                   ], p=0.2),
                   A.RandomRotate90(p=0.4),
                   # A.ShiftScaleRotate(), # TODO check is it OK (after make A.LongestMaxSize)
                   A.ToGray(p=0.2),
                   ]

post_transforms = [A.PadIfNeeded(min_height=MAX_SIZE, min_width=MAX_SIZE),
                   A.Normalize(mean=0, std=1, max_pixel_value=255),
                   A.ToTensorV2()
                   ]

preproc_aug = A.Compose(preproc_transforms,
                        bbox_params=A.BboxParams(format=BBOX_FORMAT, label_fields=['class_labels']),
                        keypoint_params=A.KeypointParams(format=KEYPOINT_FORMAT)
                        )

main_aug = A.Compose(main_transforms,
                     bbox_params=A.BboxParams(format=BBOX_FORMAT, label_fields=['class_labels']),
                     keypoint_params=A.KeypointParams(format=KEYPOINT_FORMAT)
                     )

post_aug = A.Compose(post_transforms,
                     bbox_params=A.BboxParams(format=BBOX_FORMAT, label_fields=['class_labels']),
                     keypoint_params=A.KeypointParams(format=KEYPOINT_FORMAT)
                     )
