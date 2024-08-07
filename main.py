import skimage as ski
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d


darkness = 0
saturation = 16383

    
def linearize(img, darkness, saturation):
    img_linearized = (img - darkness) / (saturation - darkness)
    img_linearized = np.clip(img_linearized, 0, 1)
    return img_linearized


def downsample(img):
    width = img.shape[1]
    height = img.shape[0]

    channel_r = np.zeros((int(height / 2), int(width / 2)))
    channel_g = np.zeros((int(height / 2), int(width / 2)))
    channel_b = np.zeros((int(height / 2), int(width / 2)))

    for i in range(0, height, 2):
        for j in range(0, width, 2):
            channel_r[int(i / 2), int(j / 2)] = img[i + 1, j + 1]
            channel_g[int(i / 2), int(j / 2)] = img[i, j + 1]
            channel_b[int(i / 2), int(j / 2)] = img[i, j]

    img_downsampled = np.dstack((channel_r, channel_g, channel_b))
    return img_downsampled

# Gray world white balance         
def gray_world_white_balance(img):
    downsampled = downsample(img)
    avg_r = np.mean(downsampled[:, :, 0])
    avg_g = np.mean(downsampled[:, :, 1])
    avg_b = np.mean(downsampled[:, :, 2])

    avg_gray = (avg_r + avg_g + avg_b) / 3

    for i in range(0, img.shape[0], 2):
        for j in range(0, img.shape[1], 2):
            img[i, j] = img[i, j] * avg_gray / avg_b
            img[i, j + 1] = img[i, j + 1] * avg_gray / avg_g
            img[i + 1, j] = img[i + 1, j] * avg_gray / avg_g
            img[i + 1, j + 1] = img[i + 1, j + 1] * avg_gray / avg_r
    
    img = np.clip(img, 0, 1).astype(np.double)
    return img

def white_world_white_balance(img):  
    downsampled = downsample(img)
    max_r = np.max(downsampled[:, :, 0])
    max_g = np.max(downsampled[:, :, 1])
    max_b = np.max(downsampled[:, :, 2])

    for i in range(0, img.shape[0], 2):
        for j in range(0, img.shape[1], 2):
            img[i, j] = img[i, j] / max_b
            img[i, j + 1] = img[i, j + 1] / max_g
            img[i + 1, j] = img[i + 1, j] / max_g
            img[i + 1, j + 1] = img[i + 1, j + 1] / max_r

    img = np.clip(img, 0, 1).astype(np.double)

    return img

def preset_white_balance(img, r_scale, g_scale, b_scale):
    for i in range(0, img.shape[0], 2):
        for j in range(0, img.shape[1], 2):
            img[i, j] = img[i, j] * b_scale
            img[i, j + 1] = img[i, j + 1] * g_scale
            img[i + 1, j] = img[i + 1, j] * g_scale
            img[i + 1, j + 1] = img[i + 1, j + 1] * r_scale
    
    img = np.clip(img, 0, 1).astype(np.double)

    return img

def demosaic(img):
    r_channel = np.zeros_like(img)
    g_channel = np.zeros_like(img)
    b_channel = np.zeros_like(img)

    r_channel[0::2, 0::2] = img[0::2, 0::2]
    g_channel[0::2, 1::2] = img[0::2, 1::2]
    g_channel[1::2, 0::2] = img[1::2, 0::2]
    b_channel[1::2, 1::2] = img[1::2, 1::2]

    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])

    interp_r = interp2d(x[0::2], y[0::2], r_channel[0::2, 0::2], kind='linear')    
    interp_g = interp2d(x, y[0::2], g_channel[0::2], kind='linear')
    interp_b = interp2d(x[1::2], y[1::2], b_channel[1::2, 1::2], kind='linear')

    red = interp_r(x, y)
    green = interp_g(x, y)
    blue = interp_b(x, y)

    img_demosaiced = np.dstack((red, green, blue))
    img_demosaiced = np.clip(img_demosaiced, 0, 1).astype(np.double)
    return img_demosaiced

def color_correction(img):
    # adobe_coeff referenced from dcraw code for Nikon D3400     
    adobe_coeff = np.array([[6988,-1384,-714],[-5631,13410,2447], [-1485,2204,7318]])
    srgb = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])

    transform = np.matmul(adobe_coeff, srgb)
    row_sums = transform.sum(axis=1, keepdims=True)
    transform = transform / row_sums

    corrected_img = img.reshape(-1,3).dot(transform.T)
    corrected_img = corrected_img.reshape(img.shape)
    corrected_img = np.clip(corrected_img, 0, 1).astype(np.double)

    return corrected_img

def brighten(img):
    grayscale = ski.color.rgb2gray(img)
    target = 0.35   #parameter
    ratio = target / np.mean(grayscale)
    img_brightened = img * ratio
    img_brightened = np.clip(img_brightened, 0, 1).astype(np.double)
    return img_brightened

def gamma_encode(img):    #np.where 条件判断  第一个是成立的
    img_gamma_encoded = np.where(img <= 0.0031308,
                    12.92 * img,
                    1.055 * np.power(img, 1/2.4) - 0.055)
    
    return img_gamma_encoded

def automatic(img):
    # Generate White Balanced Images
    linearized_img = linearize(img, darkness, saturation)
    img_gray_whitebalanced = gray_world_white_balance(linearized_img)
    img_white_whitebalanced = white_world_white_balance(linearized_img)
    img_preset_whitebalanced = preset_white_balance(linearized_img, 1.628906, 1.0, 1.386719)

    img_gray_demosaiced = demosaic(img_gray_whitebalanced)
    img_white_demosaiced = demosaic(img_white_whitebalanced)
    img_preset_demosaiced = demosaic(img_preset_whitebalanced)

    img_gray_corrected = color_correction(img_gray_demosaiced)
    img_white_corrected = color_correction(img_white_demosaiced)
    img_preset_corrected = color_correction(img_preset_demosaiced)

    img_gray_brightened = brighten(img_gray_corrected)
    img_white_brightened = brighten(img_white_corrected)
    img_preset_brightened = brighten(img_preset_corrected)

    img_gray_gamma_encoded = gamma_encode(img_gray_brightened)
    img_white_gamma_encoded = gamma_encode(img_white_brightened)
    img_preset_gamma_encoded = gamma_encode(img_preset_brightened)

    # Display the image
    
    fig = plt.figure(figsize=(10, 10))

    fig.add_subplot(6, 3, 1)
    plt.ylabel('Original', rotation=0, size='large', labelpad=60)
    plt.imshow(img)
    plt.title('Gray World')

    fig.add_subplot(6, 3, 2)
    plt.imshow(img)
    plt.title('White World')

    fig.add_subplot(6, 3, 3)
    plt.imshow(img)
    plt.title('Preset')

    fig.add_subplot(6, 3, 4)
    plt.ylabel('White Balanced', rotation=0, size='large', labelpad=60)
    plt.imshow(img_gray_whitebalanced)

    fig.add_subplot(6, 3, 5)
    plt.imshow(img_white_whitebalanced)

    fig.add_subplot(6, 3, 6)
    plt.imshow(img_preset_whitebalanced)

    fig.add_subplot(6, 3, 7)
    plt.ylabel('Demosaiced', rotation=0, size='large', labelpad=60)
    plt.imshow(img_gray_demosaiced)

    fig.add_subplot(6, 3, 8)
    plt.imshow(img_white_demosaiced)

    fig.add_subplot(6, 3, 9)
    plt.imshow(img_preset_demosaiced)

    fig.add_subplot(6, 3, 10)
    plt.imshow(img_gray_corrected)
    plt.ylabel('Color Corrected', rotation=0, size='large', labelpad=60)

    fig.add_subplot(6, 3, 11)
    plt.imshow(img_white_corrected)


    fig.add_subplot(6, 3, 12)
    plt.imshow(img_preset_corrected)


    fig.add_subplot(6, 3, 13)
    plt.imshow(img_gray_brightened)
    plt.ylabel('Brightened', rotation=0, size='large', labelpad=60)

    fig.add_subplot(6, 3, 14)
    plt.imshow(img_white_brightened)


    fig.add_subplot(6, 3, 15)
    plt.imshow(img_preset_brightened)


    fig.add_subplot(6, 3, 16)
    plt.imshow(img_gray_gamma_encoded)
    plt.ylabel('Gamma Encoded', rotation=0, size='large', labelpad=60)

    fig.add_subplot(6, 3, 17)
    plt.imshow(img_white_gamma_encoded)

    fig.add_subplot(6, 3, 18)
    plt.imshow(img_preset_gamma_encoded)

    plt.tight_layout()
    plt.show()

    # Save the images
    ski.io.imsave('baby_gray.png', (img_gray_gamma_encoded*255).astype(np.uint8))
    ski.io.imsave('baby_white.png', (img_white_gamma_encoded*255).astype(np.uint8))
    ski.io.imsave('baby_preset.png', (img_preset_gamma_encoded*255).astype(np.uint8))

    ski.io.imsave('baby_gray.jpeg', (img_gray_gamma_encoded*255).astype(np.uint8), quality=95)
    ski.io.imsave('baby_white.jpeg', (img_white_gamma_encoded*255).astype(np.uint8), quality=95)
    ski.io.imsave('baby_preset.jpeg', (img_preset_gamma_encoded*255).astype(np.uint8), quality=95)
   


def manual(img):
    img = linearize(img, darkness, saturation)

    plt.imshow(img)
    plt.title("Click on a point that should be white")
    coords = plt.ginput(1)  # Get one point, this is a list of (x, y) tuples
    plt.close()
    x, y = int(coords[0][0]), int(coords[0][1])  # Convert float coords to int
    
    reference_rgb = img[y, x] 
    print(reference_rgb)
    scale = reference_rgb / np.max(reference_rgb)
    print(scale)
    # Apply the scaling factors to each channel
    img_wb = img / scale

    img_manual_whitebalanced = np.clip(img_wb, 0, 1).astype(np.double)
    
    img_manual_demosaiced = demosaic(img_manual_whitebalanced)
    img_manual_corrected = color_correction(img_manual_demosaiced)
    img_manual_brightened = brighten(img_manual_corrected)
    img_manual_gamma_encoded = gamma_encode(img_manual_brightened)

    fig = plt.figure(figsize=(10, 10))

    fig.add_subplot(6, 1, 1)
    plt.imshow(img)
    plt.ylabel('Original', rotation=0, size='large', labelpad=60)

    fig.add_subplot(6, 1, 2)
    plt.imshow(img_manual_whitebalanced)
    plt.ylabel('White Balanced', rotation=0, size='large', labelpad=60)

    fig.add_subplot(6, 1, 3)
    plt.imshow(img_manual_demosaiced)
    plt.ylabel('Demosaiced', rotation=0, size='large', labelpad=60)

    fig.add_subplot(6, 1, 4)
    plt.imshow(img_manual_corrected)
    plt.ylabel('Color Corrected', rotation=0, size='large', labelpad=60)

    fig.add_subplot(6, 1, 5)
    plt.imshow(img_manual_brightened)
    plt.ylabel('Brightened', rotation=0, size='large', labelpad=60)

    fig.add_subplot(6, 1, 6)
    plt.imshow(img_manual_gamma_encoded)
    plt.ylabel('Gamma Encoded', rotation=0, size='large', labelpad=60)

    plt.tight_layout()
    plt.show()


    ski.io.imsave('baby_white_manual.png', (img_manual_gamma_encoded*255).astype(np.uint8))

   
    

if __name__ == '__main__':

    img = ski.io.imread('baby.tiff')

    print("Automatic or Manual? (a/m)")
    choice = input()
    if choice == 'a':
        automatic(img)
    else:
        manual(img) 