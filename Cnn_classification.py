import cv2
import glob
import numpy as np
from keras.applications.resnet50 import ResNet50
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
import pickle


#import images from the file 
def import_images(path):
	X_data = []
	files = glob.glob (path)
	for myFile in files:
		print(myFile)
		image = cv2.imread (myFile)
		X_data.append(image)
	return X_data


# this function will return feature matrix for images
def extract_resnet(X):  
    [n,h,w,c] = np.array(X).shape
    resnet_model = ResNet50(input_shape=(h, w, c), weights='imagenet',pooling='max' ,include_top=False)  # Since top layer is the fc layer used for predictions
    features_array = resnet_model.predict(np.array(X),verbose=1)
    print(features_array.shape)
    return features_array


def classify_as_forged_or_original(image):

	# features for training images are already saves in TrainingFeaturesMatrix.npy
	X_train = np.load('TraningFeaturesMatrix.npy')
	#X_test = np.load('TestFeaturesMatrix.npy')


	X_test = []
	X_test.append(image)


	#extracting features from resnet for test image
	X_test = extract_resnet(X_test)

	# Apply standard scaler to output from resnet50
	ss = StandardScaler()
	ss.fit(X_train)
	X_train = ss.transform(X_train)
	X_test = ss.transform(X_test)

	# Take PCA to reduce feature space dimensionality
	pca = PCA(n_components=512, whiten=True)
	pca = pca.fit(X_train)
	print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)

	
	# getting object for isolation forest
	if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search

	# fitting the trainig data to isolation forest
	if_clf.fit(X_train)

	# predicting resutls from isolation forest
	if_preds = if_clf.predict(X_test)

	#print(oc_svm_clf)
	print(if_preds)

	if (if_preds[0]==1):
		return True
	else:
		return False
		
	


