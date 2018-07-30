import cv2
import glob
import numpy as np
from keras.applications.resnet50 import ResNet50
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
import pickle


def import_images(path):
	X_data = []
	files = glob.glob (path)
	for myFile in files:
		print(myFile)
		image = cv2.imread (myFile)
		X_data.append(image)
	return X_data

def extract_resnet(X):  
    [n,h,w,c] = np.array(X).shape
    resnet_model = ResNet50(input_shape=(h, w, c), weights='imagenet',pooling='max' ,include_top=False)  # Since top layer is the fc layer used for predictions
    features_array = resnet_model.predict(np.array(X),verbose=1)
    print(features_array.shape)
    return features_array


training_images_path = "newH/*.jpg"

X_train_images = import_images(training_images_path)

print(np.array(X_train_images).shape)

X_train = extract_resnet(X_train_images)
np.save('TraningFeaturesMatrix.npy',X_train)


ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)

# Take PCA to reduce feature space dimensionality
pca = PCA(n_components=512, whiten=True)
pca = pca.fit(X_train)
print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
X_train = pca.transform(X_train)

if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search
if_clf.fit(X_train)

# save the model to disk
filename = 'IsolationForest.sav'
pickle.dump(if_clf, open(filename, 'wb'))


