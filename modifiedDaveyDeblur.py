import numpy as np
import math as m

import os, shutil
from PIL import Image, ImageFont, ImageDraw

import writeMatrix as wm # Debugging
import pdb               # Debugging


SNAPSHOT_DIR = './snapshots'

# Perform blind image deconvolution using the modified Davey et al algorithm
def bid_davey(original_img, blurred_img, (psfWidth, psfHeight), iterations, save_snapshots=False):

    snapshot_filenames = []
    if(save_snapshots) :
        if os.path.exists(SNAPSHOT_DIR):
            shutil.rmtree(SNAPSHOT_DIR)
        os.mkdir(SNAPSHOT_DIR)


    result = 0                                    # Optimal restored image

    original_img = original_img.astype(float)
    blurred_img = blurred_img.astype(float)

    supportWidth = np.shape(original_img)[1]
    supportHeight = np.shape(original_img)[0]
    channels = np.shape(original_img)[2]

    entireWidth = np.shape(blurred_img)[1]
    entireHeight = np.shape(blurred_img)[0]



    # Embed original image within frame (and make original image the frame)
    frame = np.zeros((entireHeight, entireWidth, channels))
    origin_x = (np.ceil((entireWidth-supportWidth)/2)+1).astype(int)
    origin_y = (np.ceil((entireHeight-supportHeight)/2)+1).astype(int)
    frame[origin_y:(origin_y+supportHeight), origin_x: (origin_x+supportWidth), 0:channels] = original_img
    original_img = frame

    # Storage for tracking MSE vs iteration count
    image_mse = np.zeros((iterations, 2))                 # Initialise image MSE array
    #psf_mse = np.zeros((iterations, 2))                   # Initialise PSF MSE array

    min_mse = 0                                          # Minimum MSE
    min_mse_iterations = 0                               # Number of iterations to achieve minimum MSE


    ##########################
    # Set up initial estimates

    f = blurred_img
    # initial_estimate = np.round(255*np.random.rand(entireHeight, entireWidth, channels))

    g = blurred_img
    # g  = np.round(255*np.random.rand(entireHeight, entireWidth, channels))

    ##########################

    # Message display
    print "Performing optimal deblur. Progress: [",

    # Used for progress bar display
    countMAX = np.round(iterations * 0.05) # 5% increments
    counter = 0

    G = np.fft.fft2(g, axes=(0,1))

    for iteration in range (0, iterations):
        counter += 1
        if (counter == countMAX):
            print ".",            # Progress bar display
            counter = 0
            if (save_snapshots):
                generateSnapshot(f, channels, iteration, SNAPSHOT_DIR, snapshot_filenames)

        F = np.fft.fft2(f, axes=(0,1))
        H = wiener(G, F)
        h = np.fft.fftshift(np.fft.ifft2(H, axes=(0,1)), axes=(0,1))
        h = psfConstraint(h, (psfWidth, psfHeight), (supportWidth, supportHeight))
        H = np.fft.fft2(h, axes=(0,1))
        F = wiener(G, H)
        f = np.fft.fftshift(np.fft.ifft2(F, axes=(0,1)), axes=(0,1))
        f = imageConstraint(f, (supportWidth, supportHeight))

        if (iteration == iterations - 1):
            print "] (complete)"

        # Calculate and store Mean Squared Error for iteration
        mse = computeMSE(f, original_img)
        image_mse[iteration, 0] = iteration
        image_mse[iteration, 1] = mse

        if (iteration == 0):
            min_mse = mse
            min_mse_iterations = 0
            result = f

        if (mse < min_mse):
            min_mse = mse
            min_mse_iterations = iteration
            result = f

    result = result[origin_y:origin_y+supportHeight, origin_x:origin_x+supportWidth, 0:channels]

    for ch in range (0, channels):
        result[:, :, ch] = np.multiply(np.divide(result[:, :, ch], np.max(result[:, :, ch])), 255)
    result = np.abs(result).astype(np.uint8)

    if(save_snapshots):
        images = []
        for filename in snapshot_filenames:
            images.append(imageio.imread(filename))
            imageio.mimsave(SNAPSHOT_DIR + '/snapshots.gif', images, format='GIF', duration=1)

    return (result, image_mse, min_mse, min_mse_iterations)


def psfConstraint(data, psf_dims, support):

    supportWidth = support[0]
    supportHeight = support[1]
    entireWidth = data.shape[1]
    entireHeight = data.shape[0]
    channels = data.shape[2]

    psfHeight  = psf_dims[1]
    psfWidth = psf_dims[0]

    origin_x = int(m.ceil((float(entireWidth - psfWidth) / 2) + 1))
    origin_y = int(m.ceil((float(entireHeight - psfHeight) / 2) + 1))

    # Enforce positivity constraint
    frame = np.zeros((entireHeight, entireWidth, channels))
    frame[origin_y-1 : origin_y+psfHeight-1, origin_x-1 : origin_x+psfWidth-1, 0:channels] = np.ones((psfHeight, psfWidth, channels))
    data = np.multiply(data, frame).clip(min=0)

    # TODO check whether whole thing should be set to zero if <0 -- or just individual real/complex parts
    # TODO need to re-add energy? from non-negative pixels?

    return data


def imageConstraint(data, support):

    entireWidth = data.shape[1]
    entireHeight = data.shape[0]
    channels = data.shape[2]

    supportWidth = support[0]
    supportHeight = support[1]

    origin_x = int(m.ceil((float(entireWidth - supportWidth) / 2) + 1))
    origin_y = int(m.ceil((float(entireHeight - supportHeight) / 2) + 1))

    # Enforce positivity constraint
    frame = np.zeros((entireHeight, entireWidth, channels))
    frame[origin_y-1 : origin_y+supportHeight-1, origin_x-1 : origin_x+supportWidth-1, 0:channels] = np.ones((supportHeight, supportWidth, channels))
    data = np.multiply(data, frame).clip(min=0)

    # TODO need to re-add energy? from non-negative pixels?

    return data



def wiener (num, den):
    beta = 0.5
    W = np.divide(np.conj(den), np.square(np.abs(den)) + np.divide(beta, np.square(np.abs(den)) ) )
    return np.multiply(W, num)


# Calculate the Mean Squared Error (MSE) between two images
def computeMSE(img_estimate, src_image):
    mse = 0

    # Normalise image estimate
    img_estimate = np.multiply(np.divide(img_estimate, np.max(img_estimate)), 255)
    if (img_estimate.shape != src_image.shape):
        raise Exception('ERROR: images are not the same dimensions [computeMSE()].')
    else:
        mse = np.abs(np.square(np.subtract(img_estimate, src_image)).mean())

    return mse

def generateSnapshot(f, channels, iteration, SNAPSHOT_DIR, snapshot_filenames):
    for ch in range (0, channels):
        snapshot = f
        snapshot[:, :, ch] = np.multiply(np.divide(snapshot[:, :, ch], np.max(snapshot[:, :, ch])), 255)
    snapshot = np.abs(snapshot).astype(np.uint8)
    filename = SNAPSHOT_DIR + '/snap_' + '{:06d}'.format(iteration) + '.bmp'

    img = Image.fromarray(snapshot[:,:,0],mode='L')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((10, snapshot.shape[0]-10), "Snap-" + str(iteration), font=font, fill=255)
    snapshot_filenames.append(filename)
    img.save(filename)
