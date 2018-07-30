from sift_comparison import compare
from Cnn_classification import classify_as_forged_or_original 
import cv2

def forgery_check(img):

	if(compare(img) | classify_as_forged_or_original(img)):
		return True
	else:
		return False


forgery_check(cv2.imread("test/h1.jpg"))
