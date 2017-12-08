import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
from helper import fit_polynomial, fit_polynomial_consecutive, measure_curve, draw_lines
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Read the saved object points and image points
dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel>=thresh[0]) & (scaled_sobel<=thresh[1])] = 1
    return binary_output

def color_thresh(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    s_binary = np.zeros_like(S)
    s_binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    r_binary = img[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(s_binary == 1) | (r_binary == 1)] = 1
    return binary_output

def smooth_fits(left_fit, right_fit, n=3):
    left_fit_g.append(left_fit)
    right_fit_g.append(right_fit)

    left_fit_np = np.array(left_fit_g)
    right_fit_np = np.array(right_fit_g)

    if len(left_fit_g) > n:
        left_fit = np.mean(left_fit_np[-n:, :], axis=0)
    if len(right_fit_g) > n:
        right_fit = np.mean(right_fit_np[-n:, :], axis=0)
        
    return left_fit, right_fit

def sanity_check(left_fit, right_fit):
    # Calculate slope of left and right lanes at midpoint of y (i.e. 360)
    L_0 = 2*left_fit[0]*360+left_fit[1]
    R_0 = 2*right_fit[0]*360+right_fit[1]
    delta_slope_mid =  np.abs(L_0-R_0)

    #Check if lines are parallel at the middle
    if delta_slope_mid<=0.1:
        status = True
    else:
        status = False

    return status

def process_image(img):
    img = cv2.undistort(img, mtx, dist, None, mtx)
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=13, thresh=(25, 255))
    color_binary = color_thresh(img, thresh=(170, 255))
    binary_img = np.zeros_like(img[:,:,0])
    binary_img[(gradx == 1) | (color_binary == 1)] = 255
        
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[293, 668], [587, 458], [703, 458], [1028, 668]])
    dst = np.float32([[310, img_size[1]], [310, 0],[950, 0], [950, img_size[1]]])

    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(binary_img, M, img_size, flags=cv2.INTER_LINEAR)
    
    left_fit, right_fit, out_img = fit_polynomial(warped)
    left_curverad, right_curverad, center = measure_curve(left_fit, right_fit, warped)
    result = draw_lines(img, warped, left_fit, right_fit, left_curverad, right_curverad, center, Minv)
    
    return binary_img, warped, out_img, result

def process_video(img):
    global counter
    global left_fit
    global right_fit
    global last_left 
    global last_right
    
    img = cv2.undistort(img, mtx, dist, None, mtx)
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=13, thresh=(25, 255))
    color_binary = color_thresh(img, thresh=(90, 255))
    binary_img = np.zeros_like(img[:,:,0])
    binary_img[(gradx == 1) | (color_binary == 1)] = 255
        
    img_size = (img.shape[1], img.shape[0])
    
    src = np.float32([[293, 668], [587, 458], [703, 458], [1028, 668]])
    dst = np.float32([[310, img_size[1]], [310, 0],[950, 0], [950, img_size[1]]])

    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(binary_img, M, img_size, flags=cv2.INTER_LINEAR)
    
    if(counter == 0):
        left_fit, right_fit, out_img = fit_polynomial(warped)
        # counter = counter + 1
    else:
        left_fit, right_fit = fit_polynomial_consecutive(warped, left_fit, right_fit)
    
    status = sanity_check(left_fit, right_fit)
    left_fit, right_fit = smooth_fits(left_fit, right_fit)
    
    if (status == True):        
        #Save as last reliable fit
        last_left, last_right = left_fit, right_fit        
        counter = counter + 1
    else:        
        #Use the last realible fit
        left_fit, right_fit = last_left, last_right
        # counter = 0
    
    left_curverad, right_curverad, center = measure_curve(left_fit, right_fit, warped)
    result = draw_lines(img, warped, left_fit, right_fit, left_curverad, right_curverad, center, Minv)
    
    return result


# images = glob.glob("test_images/*.jpg")

# for idx, fname in enumerate(images):
#     img = mpimg.imread(fname)
#     binary_img, warped, out_img, result = process_image(img)
#     cv2.imwrite("output_images/binary_images/binary"+str(idx)+".jpg", binary_img)
#     cv2.imwrite("output_images/warped_images/warped"+str(idx)+".jpg", warped)
#     cv2.imwrite("output_images/window_centroid/window"+str(idx)+".jpg", out_img)
#     cv2.imwrite("output_images/marker/marked"+str(idx)+".jpg", result)

counter = 0
left_fit_g, right_fit_g = [], []

output_video = "output_video.mp4"
input_video = "project_video.mp4"

clip1 = VideoFileClip(input_video)
video_clip = clip1.fl_image(process_video)
video_clip.write_videofile(output_video, audio=False)
