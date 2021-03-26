from imgaug import augmenters as iaa
# from keras.preprocessing.image import array_to_img, img_to_array, load_img
from scipy import misc
import imageio
import numpy as np
from PIL import Image, ImageDraw
import os
from os.path import join
if __name__ == '__main__':
    outFolder = 'TestDeterministic'
    os.makedirs(outFolder, exist_ok=True)

    seq_img = iaa.Sequential([
        iaa.Affine(rotate=(-45, 45), order=1, mode="constant", cval=0, name="MyAffine")
    ])

    seq_masks = iaa.Sequential([
        iaa.Affine(rotate=(-45, 45), order=0, mode="constant", cval=0, name="MyAffine")
    ])

    img = Image.new('L', [400, 400])
    d = ImageDraw.Draw(img)
    d.rectangle([(150, 150), (250, 250)], fill=255)

    x = np.array(img)
    images = x.reshape((1,) + x.shape)

    masks = images

    for i in range(10):

        seq_img = seq_img.localize_random_state()

        seq_img_i = seq_img.to_deterministic()
        seq_masks_i = seq_masks.to_deterministic()

        seq_masks_i = seq_masks_i.copy_random_state(seq_img_i, matching="name")

        imgs_aug = seq_img_i.augment_images(images)
        masks_aug = seq_masks_i.augment_images(masks)

        imageio.imwrite(join(outFolder, "img"+ str(i)+".bmp"), np.squeeze(imgs_aug))
        imageio.imwrite(join(outFolder, "msk"+ str(i)+".bmp"), np.squeeze(masks_aug))
