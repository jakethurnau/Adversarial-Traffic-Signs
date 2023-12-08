import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.image as mpimg

image_string = input("Please choose an image to add noise to: \n")
image = cv2.imread("unperterbed_signs/" + image_string)

img_arr = np.array(image)

#noise_type = input(
 #   "Will you be adding Gaussian noise, salt-and-pepper noise, poisson noise, speckle noise, or random noise " +
  #  "speckle noise? Type 1 for Gaussian, 2 for salt-and-pepper, 3 for poisson, 4 for speckle, 5 for random:\n")

#if noise_type == "1":	#Gaussian Noise
row,col,ch = img_arr.shape
mean = 0
var = 25	#edit this to change amt of noise
sigma = var#**0.5
gauss = np.random.normal(mean,sigma,(row,col,ch))
noisy_img = image + gauss
cv2.imwrite("gauss_signs/" + image_string, noisy_img)
	
	
#elif noise_type == "2":	#Salt and Pepper Noise
row,col,ch = img_arr.shape
s_vs_p = 0.5
amount = 0.1	#edit this to change amt of noise
out = np.copy(img_arr)
# Salt mode
num_salt = np.ceil(amount * img_arr.size * s_vs_p)
coords = [np.random.randint(0, i - 1, int(num_salt))
    for i in img_arr.shape]
out[coords] = 1
# Pepper mode
num_pepper = np.ceil(amount* img_arr.size * (1. - s_vs_p))
coords = [np.random.randint(0, i - 1, int(num_pepper))
    for i in img_arr.shape]
out[coords] = 0
noisy_img = out
cv2.imwrite("salt_and_pep_signs/" + image_string, noisy_img)

#elif noise_type == "3":	#Poisson Noise
vals = len(np.unique(img_arr))
vals = 2 ** np.ceil(np.log2(vals))
noisy = np.random.poisson(img_arr * vals) / float(vals) #edit this to change noise level
#noisy = np.random.poisson(150, image.shape) #edit this to change noise level
noisy_img = noisy + image
cv2.imwrite("poisson_signs/" + image_string, noisy_img)
	
#elif noise_type =="4":	#Speckle Noise
row,col,ch = img_arr.shape
noise = np.random.randn(row,col,ch)
noisy_img = image + image*noise
cv2.imwrite("speckle_signs/" + image_string, noisy_img)
	
#elif noise_type == "5":	#Random distributed noise
noise = np.random.randint(50, size = img_arr.shape, dtype = 'uint8')
noisy_img = image + noise
cv2.imwrite("random_noise_signs/" + image_string, noisy_img)
	
	
	
	
	