from PIL import Image, ImageDraw
import os
from glob import glob
import json
import numpy as np
import keras
import cv2


def get_mask(img, mask_data):
    '''
    :param img: PIL.Image
    :param mask_data: data from json file
    :return: mask
    '''
    x = mask_data.get('regions')[0].get('shape_attributes').get('all_points_x')
    y = mask_data.get('regions')[0].get('shape_attributes').get('all_points_y')
    mask_img = Image.fromarray(np.zeros(list(reversed(img.size)), dtype=np.uint8))
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(list(zip(x, y)), fill="white")

    return mask_img


def get_resized_data(path_out, imgs, masks_json, size):

    '''
    Function generates resized images and masks for further modeling

    :param path_out: path to allocate resized images/masks
    :param imgs: the list of images, full paths
    :param masks_json: json file with masks coordinates
    :param size: the final size of images
    :return: None. Two additional directories in path_out
    '''

    cmd = 'rm -rf {0}'.format(os.path.join(path_out, 'images_' + str(size)))
    print(cmd)
    os.system(cmd)
    cmd = 'mkdir {0}'.format(os.path.join(path_out, 'images_' + str(size)))
    print(cmd)
    os.system(cmd)

    cmd = 'rm -rf {0}'.format(os.path.join(path_out, 'masks_' + str(size)))
    os.system(cmd)
    cmd = 'mkdir {0}'.format(os.path.join(path_out, 'masks_' + str(size)))
    os.system(cmd)


    for img_path in imgs:
        img_name = img_path.split('/')[-1]
        mask_data = masks_json.get('_via_img_metadata').get(img_name)
        img = Image.open(img_path)
        mask = get_mask(img, mask_data)

        img = img.resize((size, size))
        mask = mask.resize((size, size))

        img.save(os.path.join(os.path.join(path_out, 'images_' + str(size) + '/' + img_name)))
        mask.save(os.path.join(os.path.join(path_out, 'masks_' + str(size) + '/' + img_name)))


class DataGenerator(keras.utils.Sequence):
    '''
    Keras data generator
    '''

    def __init__(self, imgs_path, masks_path,
                 augmentations=None, batch_size=16, img_size=512, n_channels=3, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.train_im_paths = glob(imgs_path + '/*')

        self.train_im_path = imgs_path
        self.train_mask_path = masks_path

        self.img_size = img_size

        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.train_im_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:min((index + 1) * self.batch_size, len(self.train_im_paths))]

        # Find list of IDs
        list_IDs_im = [self.train_im_paths[k] for k in indexes]

        # Generate data
        X, y = self.data_generation(list_IDs_im)

        if self.augment is None:
            return X, np.array(y) / 255
        else:
            im, mask = [], []
            for x, y in zip(X, y):
                augmented = self.augment(image=x, mask=y)
                im.append(augmented['image'])
                mask.append(augmented['mask'])
            return np.array(im), np.array(mask) / 255

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.train_im_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_im):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(list_IDs_im), self.img_size, self.img_size, self.n_channels))
        y = np.empty((len(list_IDs_im), self.img_size, self.img_size, 1))

        # Generate data
        for i, im_path in enumerate(list_IDs_im):

            im = np.array(Image.open(im_path))
            mask_path = im_path.replace(self.train_im_path, self.train_mask_path)

            mask = np.array(Image.open(mask_path))

            if len(im.shape) == 2:
                im = np.repeat(im[..., None], 3, 2)

            #             # Resize sample
            X[i,] = cv2.resize(im, (self.img_size, self.img_size))

            # Store class
            y[i,] = cv2.resize(mask, (self.img_size, self.img_size))[..., np.newaxis]
            y[y > 0] = 255

        return np.uint8(X), np.uint8(y)


if __name__ == '__main__':

    # ---------- Testing images resizing and masks preparation
    # input_path = '/home/aziz/Documents/Cars_Proj/input/data/'
    # path_out = '/home/aziz/Documents/Cars_Proj/input/resized/train'
    # size = 512
    #
    # train = sorted(glob(os.path.join(input_path, 'train/*.jpeg')))
    # val = sorted(glob(os.path.join(input_path, 'val/*.jpeg')))
    #
    # print(len(train), len(val))
    #
    # json_file = open(os.path.join(input_path, 'train.json'))
    # train_masks_json = json.load(json_file)
    #
    # json_file = open(os.path.join(input_path, 'val.json'))
    # val_masks_json = json.load(json_file)
    #
    # get_resized_data(path_out, train, train_masks_json, size)
    # -------------------------------------------------------------

    # Testing data generator
    train_im_path = '/home/aziz/Documents/Cars_Proj/input/resized/train/images_512'
    train_mask_path = '/home/aziz/Documents/Cars_Proj/input/resized/train/masks_512'

    a = DataGenerator(imgs_path=train_im_path, masks_path=train_mask_path, augmentations=None,
                      batch_size=16, img_size=512, n_channels=3, shuffle=False)
    images, masks = a.__getitem__(1)
    print(images.shape, masks.shape)
    # -------------------------------------------------------------