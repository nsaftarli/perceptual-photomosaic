import sys
sys.path.append('layers/')
sys.path.append('utils/')
import os
import numpy as np
from PIL import Image
from utils.constants import Constants
import itertools


###############################################
from skimage import feature
from scipy import ndimage as ndi
##############################################

const = Constants()

img_data_dir = const.img_data_dir
ascii_data_dir = const.ascii_data_dir
flipped_data_dir = const.ascii_data_dir_flip
val_data_dir = const.val_data_dir
char_array = const.char_array
char_dict = const.char_dict
train_set_size = const.train_set_size
coco_dir = const.coco_dir
video_dir = const.video_dir



def load_data(num_batches=100,
              batch_size=6,
              img_rows=224,
              img_cols=224,
              txt_rows=28,
              txt_cols=28,
              flipped=False,
              validation=False,
              test=False):

    ind = 0
    x = np.zeros((batch_size, img_rows, img_cols, 3), dtype='uint8')

    while True:
        if ind == num_batches:
            ind = 0

        if flipped:
            count = batch_size / 2
        else:
            count = batch_size

        for i in range(count):
            if validation:
                imgpath = img_data_dir + 'in_' + str(train_set_size + (ind * count) + i) + '.jpg'
            else:
                imgpath = img_data_dir + 'in_' + str(ind * count + i) + '.jpg'

            img = Image.open(imgpath)
            x[i] = np.asarray(img, dtype='uint8')

            if flipped:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                x[i + count] = np.asarray(img, dtype='uint8')

        ind += 1

        if test:
            break

        yield (x, i)


def load_data_gen(num_batches=100,
                  batch_size=6,
                  img_rows=512,
                  img_cols=512):
    ind = 0
    n = 0

    directory = os.listdir(coco_dir)
    while True:
        x = np.zeros((batch_size, img_rows, img_cols, 3), dtype='uint8')

        for i in itertools.count(ind, 1):
            if ind == 12000:
                ind = 0
                n = 0
                break
            # imgpath = img_data_dir + 'in_' + str(i) + '.jpg'
            # imgpath = coco_dir + 'COCO_train2014' + str(i) + '.jpg'
            imgpath = coco_dir + directory[n]
            n += 1

            img = Image.open(imgpath)
            x[i % batch_size] = np.asarray(img, dtype='uint8')
            if i == ind + batch_size:
                ind += batch_size
                yield x, i


def load_val_data_gen(num_batches=100,
                      batch_size=6,
                      img_rows=512,
                      img_cols=512):

    ind = 15000

    n = 0
    directory = os.listdir(coco_dir)
    while True:
        x = np.zeros((batch_size, img_rows, img_cols, 3), dtype='uint8')

        for i in itertools.count(ind, 1):
            if ind == 15060:
                ind = 15000
                break
            # imgpath = img_data_dir + 'in_' + str(i) + '.jpg'
            imgpath = coco_dir + directory[n]
            n += 1
            img = Image.open(imgpath)
            x[i % batch_size] = np.asarray(img, dtype='uint8')
            if i == ind + batch_size:
                ind += batch_size
                yield x, ind

def load_vid_data_gen(num_batches=100,
                      batch_size=100,
                      img_rows=376,
                      img_cols=512):

    ind = 0
    n = 0

    directory = sorted(os.listdir(video_dir))
    while True:
        x = np.zeros((batch_size, img_rows, img_cols, 3), dtype='uint8')

        for i in itertools.count(ind, 1):
            if i == 5654:
                ind = 0
                n = 0
                break
            # imgpath = img_data_dir + 'in_' + str(i) + '.jpg'
            # imgpath = coco_dir + 'COCO_train2014' + str(i) + '.jpg'
            # imgpath = coco_dir + directory[n]
            # imgpath = video_dir + str(i+1) + '.jpg'
            imgpath = video_dir + directory[n]
            n += 1
            # print(imgpath)
            if n == len(directory):
                n = 0

            img = Image.open(imgpath)
            x[i % batch_size] = np.asarray(img, dtype='uint8')
            if i == ind + batch_size:
                ind += batch_size
                yield x,i+1

def get_templates(path='./assets/char_set/', temp_size=8, num_temps=16, rgb=True):

    if not rgb:
        images = np.zeros((1, temp_size, temp_size, num_temps))
        for j in range(num_temps):
            im = Image.open(path + str(j) + '.png')
            im = np.asarray(im, dtype='uint8')
            images[0, :, :, j] = np.mean(im, axis=-1)
    else:
        images = np.zeros((1, temp_size, temp_size, 3, num_temps))
        for j in range(num_temps):
            if j == 11:
                continue
            im = Image.open(path + str(j) + '.png')
            im = np.asarray(im, dtype='uint8')
            images[0, :, :, :, j] = im
    return images

def get_emoji_templates(path='./assets/emoji_temps/', num_temps=16, rgb=True):
    directory = os.listdir(path)
    images = np.zeros((1, 16, 16, 3, num_temps))
    for j in range(num_temps):
        x = np.random.randint(0,573)
        im = Image.open(path + directory[x]).resize((16, 16))
        im = np.asarray(im, dtype='uint8')
        images[0, ..., j] = im[..., :3]
    return images

def get_other_templates(path='./assets/char_set_coloured/', patch_size=8):
    directory = os.listdir(path)
    num_temps = len(directory)
    images = np.zeros((1, patch_size, patch_size, 3, num_temps))
    for j in range(num_temps):
        im = Image.open(path + directory[j])
        im = np.asarray(im, dtype='uint8')
        images[0, ..., j] = im[..., :3]
        # print(images.shape)
    return images

def process_emojis(num_temps=62):
    path_in = './assets/emoji_temps/'
    path_out = './assets/emoji_temps_full_16/'
    directory = os.listdir(path_in)

    # im = Image.open(path + directory[1])
    x = 0
    for j in range(753):
        # print(j)
        im = Image.open(path_in + directory[j]).resize((16, 16))
        im_arr = np.asarray(im, dtype='uint8')
        print(im_arr.shape)
        if len(im_arr.shape) < 3 or im_arr.shape[2] <= 2:
            continue
        if im_arr.shape[2] == 4:
            im = pure_pil_alpha_to_color_v2(im)
        im_arr = np.asarray(im, dtype='uint8')
        print(im_arr.shape[2])
        im.save(path_out + str(x) + '.png')
        x += 1


def rgba_to_rgb(path):
    directory = os.listdir(path)
    print(directory)
    print(len(directory))
    for i in range(len(directory)):
        im = Image.open(path + directory[i])
        if np.asarray(im).shape[2] == 4:
            print(im)
            im = pure_pil_alpha_to_color_v2(im)
            im.save(path + directory[i])


def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background


def get_pebbles(path='./pebbles.jpg'):
    edges = np.zeros((224, 224, 3), dtype='uint8')
    img = Image.open(path).resize((224, 224))
    img_arr = np.asarray(img)

    return img_arr.reshape((-1, 224, 224, 3))



def create_char_img(path='./assets/char_set/0.png'):
    tiles = np.zeros((224, 224, 3))
    im = Image.open(path)
    for i in range(28):
        for j in range(28):
            tiles[i:2*i-1, j:2*j-1, :] = im

def make_template_ims(path='./assets/temp_pics/',temp_size=8, num_temps=62):
    imgpath = path + 'a.jpg'
    # print(Image.)
    im = np.asarray(Image.open(imgpath).resize((287, 224)))
    template = np.zeros((8, 8, 3))
    print(im.shape)
    h = im.shape[0]
    w = im.shape[1]
    n = 0

    for i in range(num_temps):
        y = np.random.randint(0, 196)
        x = np.random.randint(0, 264)

        template[:,:,:] = im[y:y+8, x:x+8,:]
        out = Image.fromarray(template.astype('uint8'))
        out.save('./assets/cam_templates/' + str(n) + '.png', 'PNG')
        n += 1


def turn_im_into_templates(path='./assets/temp_pics/', temp_size=8):
    imgpath = path + 'picasso-femmes-d-alger.jpg'
    im = np.asarray(Image.open(imgpath).resize((288, 224)))
    x = 0
    for i in range(im.shape[0] // temp_size):
        for j in range(im.shape[1] // temp_size):
            window = im[i*temp_size:i*temp_size+temp_size, j*temp_size:j*temp_size+temp_size, :]
            out = Image.fromarray(window.astype('uint8'))
            out.save('./assets/cam_templates_2/' + str(x) + '.png', 'PNG')
            x += 1

def make_face_templates(path='./assets/rgb_in/img_celeba/', num_temps=62):
    for i in range(num_temps):
        n = np.random.randint(0,202598)
        imgpath = path + 'in_' + str(n) + '.jpg'
        im = np.asarray(Image.open(imgpath).resize((8, 8)))
        out = Image.fromarray(im.astype('uint8'))
        out.save('./assets/face_templates/' + str(i), 'JPEG')


def clean_dataset(path='../data/coco-resized-512/'):
    path = '/home/nsaftarl/ASCIIArtNN/assets/coco-set/train2014/'
    for im in os.listdir(path):
        if len(np.asarray(Image.open(path + im)).shape) != 3:
            os.remove(path + im)


def resize_dataset(in_path = '/home/nsaftarl/ASCIIArtNN/assets/coco-set/train2014/',
                   out_path = '/home/nsaftarl/ASCIIArtNN/assets/coco-resized-512/'):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for im in os.listdir('/home/nsaftarl/ASCIIArtNN/assets/coco-set/train2014/'):
        img = np.asarray(Image.open(in_path + im).resize((512, 512), resample=Image.BILINEAR), dtype='uint8')
        img = Image.fromarray(img)
        img.save(out_path + im)



def resize_movie(in_path = '/home/nsaftarl/ASCIIArtNN/assets/mv/',
                 out_path = '/home/nsaftarl/ASCIIArtNN/assets/mv-resized-512/'):

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for im in os.listdir(video_dir):
        print(im)
        img = np.asarray(Image.open(in_path + im).resize((512, 512), resample=Image.BILINEAR), dtype='uint8')
        img = Image.fromarray(img)
        img.save(out_path + im)

if __name__ == '__main__':
    # create_char_img()
    # overlay_img()
    # make_template_ims()
    # turn_im_into_templates()
    # make_face_templates()
    # resize_coco()
    # resize_movie()
    # load_vid_data_gen()
    process_emojis()