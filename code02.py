# import libraries
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
import skimage.transform as sktr 
import skimage.filters as skfl 
import skimage.color as skcol
import os

# this class is for creating a list of image (jpeg, jpg) files,
# within a specific directory
class Image_directory:
    def __init__(self, directory='data'):
        self.directory = directory
    
    # function to create list of image files (will support jpeg or jpg files)
    # directory: 'data' (default), directory to be iterated
    def get_file_list(self):
        file_list = []
        # first check if 'data' directory is in current directory or in
        # ../ directory
        if os.path.exists(self.directory):
            dir = self.directory
        else:
            dir = '../'+self.directory
        for filename in os.listdir(dir):
            f = os.path.join(dir, filename)
            # checking if it is a file
            if os.path.isfile(f):
                if f.lower().endswith(('.jpg', '.jpeg')):
                    file_list.append(f)
        return file_list

# this class performs all functions regarding processing 
# the image
class Image_process:
    def __init__(self, image_in):
        """
        Constructor
        """
        self.image_in = image_in

    # function to display image
    # img: image to be displayed
    # title: title of image
    # grayscale: 0 (default), 1 for grayscale
    # newFig: True (default), False for not invoking plt.figure
    # immediateShow: True (default), False for not invoking plt.show()
    def show_im(self, img, title, grayscale=0, newFig=True, immediateShow=True):
        cmap_val = 'gray' if grayscale==1 else None
        if newFig:
            plt.figure()
        plt.imshow(img, cmap=cmap_val, vmin=0, vmax=1)
        plt.colorbar()
        plt.clim = (0,1)
        plt.title(title)
        if immediateShow:
            plt.show()

    # function to display histogram
    # img_in: input image
    # title: title of histogram
    # xName: x axis label
    # yName: y axis label
    # numBins: 20 (default), number of bins
    # histRange: 0,1 (default), range of values
    def show_hist(self, img, title, xName, yName, numBins=20, histRange=(0,1), newFig=True, immediateShow=True):
        if newFig:
            plt.figure()
        plt.hist(img, bins=numBins, range=histRange)
        plt.title(title)
        plt.xlabel(xName)
        plt.ylabel(yName)
        if immediateShow:
            plt.show()

    # function to perform mask detection and histogram stretching
    def mask_det_hist_str(self):
        # read image into array
        im_resz = self.load_image()
        # convert to grayscale
        imG = self.convrgb2gray(im_resz)
        # find otsu threshold
        thresh = self.findOtsuThresh(imG)
        # create mask
        bool_mask = self.boolMask(imG, thresh)
        # extract mask
        img_mask = self.extractBoolMask(bool_mask, imG)
        # histogram stretching on mask
        img_mask_stretch_flat = self.histStr(img_mask, 0, 1)
        # get extracted pixels ready for display
        img2 = self.extractStrPix(imG, img_mask_stretch_flat, bool_mask)
        return (imG, img2)

    # load image file and resize to 20%. Return resized image
    def load_image(self):
        # check if 'data' is in current directory or in 
        # ../ directory
        try: 
            im_orig = skio.imread(self.image_in)
        except:
            im_orig = skio.imread('../'+self.image_in)

        finally:
            im_resz = sktr.resize(im_orig, (im_orig.shape[0] // 5, im_orig.shape[1] // 5),
                       anti_aliasing=True)
            return im_resz

    # convert rgb to grayscale and return converted image
    def convrgb2gray(self, img):
        imG = skcol.rgb2gray(img)
        return imG
            
    # brightest area on image
    def brtAreaImg(self, img):
        ind_row, ind_col = np.unravel_index(np.argmax(img), img.shape)
        return ind_row, ind_col
    
    # create window for brightest area and return windowed image
    def windDim(self, img, width, ind_row, ind_col):
        wind_dim = int(img.shape[0]*width//2)
        imG_wind = img[ind_row-wind_dim:ind_row+wind_dim, ind_col-wind_dim:ind_col+wind_dim]
        return imG_wind

    # flatten image
    def flatImg(self, img):
        return img.flatten()

    # histogram stretching
    def histStr(self, img, vmin, vmax):
        stretch_a, stretch_b = (vmin, vmax)
        stretch_c, stretch_d = (img.min(), img.max())
        img_stretch_flat = ((img-stretch_c)*((stretch_b-stretch_a)/(stretch_d-stretch_c)))+stretch_a
        img_stretch_flat[img_stretch_flat < vmin] = 0
        img_stretch_flat[img_stretch_flat > vmax] = 1
        return img_stretch_flat

    # find otsu threshold
    def findOtsuThresh(self, img):
        thresh = skfl.threshold_otsu(img)
        return thresh
    
    # create boolean mask based on otsu threshold value
    def boolMask(self, img, thresh):
        bool_mask = img > thresh
        return bool_mask

    # extract pixels inside mask. This will be a 1D vector because 
    # once extraction of mask does not maintain original image
    # dimension
    def extractBoolMask(self, bool_mask, img):
        img_mask = img[bool_mask]
        return img_mask

    # consruct 2D array with masked pixels. This will be used for
    # displaying image after histogram stretching is performed on
    # the mask pixels
    def extractStrPix(self, img, img_mask_stretch_flat, bool_mask):
        img2 = img.copy()
        img2[np.invert(bool_mask)] = 0
        img2[bool_mask] = img_mask_stretch_flat
        return img2
