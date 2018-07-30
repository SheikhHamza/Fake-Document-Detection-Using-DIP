import numpy as np
import cv2
from matplotlib import pyplot as plt


def compare(img1):
	treshold = 55
	# Initiate SIFT detector
	orb = cv2.ORB_create()
	kp1, des1 = orb.detectAndCompute(img1,None)

	c = 0
	sum_of_matches = 0


	# Best_originals contain best original samples of FAST's transcript 
	# loop through the best samples and compare each with the test image
	for j in range(1,12):
		img2  = cv2.imread("Best_Originals/h"+str(j)+".pgm",0) # trainImage

		c = c + 1

		# Initiate SIFT detector
		orb = cv2.ORB_create()
		# find the keypoints and descriptors with SIFT
		# detectAndCompute will give both keypoints and descriptor
		kp2, des2 = orb.detectAndCompute(img2,None)

		# number of keypoints (number of rows of descriptor)
		r = len(des1)

		# create BFMatcher object (BF = brute force)
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		# Match descriptors.
		# give matches
		matches = bf.match(des1,des2)

		# Sort them in the order of their distance.
		matches = sorted(matches, key = lambda x:x.distance)

		number_of_keypoints_match = len(matches)

		number_of_matches = (number_of_keypoints_match/r * 100)

		sum_of_matches = sum_of_matches + number_of_matches

		 
	average = sum_of_matches/c
	print("overall average:")
	print(average)

	if (average > treshold):
		 print("True")
		 return True
	else:
		 print("False")
		 return False




