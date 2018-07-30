import numpy as np
import cv2
from matplotlib import pyplot as plt

averages = 0
total_count = 312

for i in range(1,312):

	img1 = cv2.imread("newH/h"+str(i)+".jpg",0) # queryImage

	c = 0
	sum_of_matches = 0

	for j in range(13,16):
		if (j!=i):
			img2  = cv2.imread("fake/h"+str(j)+".pgm",0) # trainImage

			c = c + 1

			# Initiate SIFT detector
			orb = cv2.ORB_create()

			# find the keypoints and descriptors with SIFT
			kp1, des1 = orb.detectAndCompute(img1,None)
			kp2, des2 = orb.detectAndCompute(img2,None)

			r = len(des1)

			# create BFMatcher object
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

			# Match descriptors.
			matches = bf.match(des1,des2)

			# Sort them in the order of their distance.
			matches = sorted(matches, key = lambda x:x.distance)

			number_of_keypoints_match = len(matches)

			# Draw first 10 matches.
			img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)

			number_of_matches = (number_of_keypoints_match/r * 100)

			sum_of_matches = sum_of_matches + number_of_matches

			# if (number_of_matches > 70):
			# 	print("True")
			# else:
			# 	print("False")

	average = sum_of_matches/c
	print(average)

	averages = averages + average

overall_average = averages / total_count

print("Overall Average:")
print(overall_average)