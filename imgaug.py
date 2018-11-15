from imgaug import augmenters as iaa
import cv2

# data augmentation
def customizedImgAug(input_img):
    rarely = lambda aug: iaa.Sometimes(0.1, aug)
    sometimes = lambda aug: iaa.Sometimes(0.25, aug)
    often = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        often(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.12, 0)},
            rotate=(-10, 10),
            shear=(-8, 8),
            order=[0, 1],
            cval=(0, 255),
        )),
        iaa.SomeOf((0, 4), [
            rarely(
                iaa.Superpixels(
                    p_replace=(0, 0.3),
                    n_segments=(20, 200)
                )
            ),
            iaa.OneOf([
                iaa.GaussianBlur((0, 2.0)),
                iaa.AverageBlur(k=(2, 4)),
                iaa.MedianBlur(k=(3, 5)),
            ]),
            iaa.Sharpen(alpha=(0, 0.3), lightness=(0.75, 1.5)),
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.5)),
            rarely(iaa.OneOf([
                iaa.EdgeDetect(alpha=(0, 0.3)),
                iaa.DirectedEdgeDetect(
                    alpha=(0, 0.7), direction=(0.0, 1.0)
                ),
            ])),
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
            ),
            iaa.OneOf([
                iaa.Dropout((0.0, 0.05), per_channel=0.5),
                iaa.CoarseDropout(
                    (0.03, 0.05), size_percent=(0.01, 0.05),
                    per_channel=0.2
                ),
            ]),
            rarely(iaa.Invert(0.05, per_channel=True)),
            often(iaa.Add((-40, 40), per_channel=0.5)),
            iaa.Multiply((0.7, 1.3), per_channel=0.5),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))),
            sometimes(
                iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)
            ),

        ], random_order=True),
        iaa.Fliplr(0.5),
        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
    ], random_order=True)  # apply augmenters in random order

    output_img = seq.augment_image(input_img)
    return output_img