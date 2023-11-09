import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import warnings

from FeatureExtractor_GLCM import GLCMFeatureExtractor
warnings.filterwarnings("ignore")

class ImageClassifier:
    def __init__(self, dataset_dir, model_dir, feature_dir, feature_type):
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        self.feature_dir = feature_dir
        self.feature_type = feature_type
        self.data = []
        self.labels = []
        self.feature_extractors = {
            "histogram": self.extract_histogram,
            "glcm": self.extract_glcm
        }
        self.classifiers = {
            "mlp": self.train_mlp,
            "naive_bayes": self.train_naive_bayes
        }

    def extract_histogram(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        # Flatten and reshape histogram to 1-dimensional array
        hist = hist.reshape(1, -1)
        return hist

    def extract_glcm(self, image):
        feature_extractor = GLCMFeatureExtractor()
        glcm_features = feature_extractor.compute_glcm_features(image)
        return glcm_features

    def load_data(self):
        for folder in os.listdir(self.dataset_dir):
            folder_path = os.path.join(self.dataset_dir, folder)
            print(f"Folder path: {folder_path}")
            
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    print(f"File path: {file_path}")
                    if file.lower().endswith(('.jpg', '.jpeg')):
                        image = cv2.imread(file_path)
                        features = self.feature_extractors[self.feature_type](image)

                        self.data.append(features)
                        self.labels.append(folder)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
    def train_mlp(self):
        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
        mlp.fit(self.data.reshape(len(self.data), -1), self.labels)
        return mlp

    def train_naive_bayes(self):
        nb = GaussianNB()
        nb.fit(self.data.reshape(len(self.data), -1), self.labels)
        return nb
        
    def train_classifier(self, classifier_type):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

        classifier = self.classifiers[classifier_type]()
        classifier.fit(X_train.reshape(len(X_train), -1), y_train)

        y_pred = classifier.predict(X_test.reshape(len(X_test), -1))
        print(classification_report(y_test, y_pred))

        self.classifier = classifier

    def save_classifier(self, classifier_type):
        np.save(os.path.join(self.feature_dir, 'data.npy'), self.data)
        np.save(os.path.join(self.feature_dir, 'labels.npy'), self.labels)

        classifier = self.classifier

        with open(os.path.join(self.model_dir, f'{classifier_type}_model.pkl'), 'wb') as f:
            pickle.dump(classifier, f)


if __name__ == "__main__":
    DATASET_DIR = 'E:/kuliahh/AI/tugas/backend_classification/dataset/face_recognition'
    MODEL_DIR = 'E:/kuliahh/AI/tugas/backend_classification/model'
    FEATURE_DIR = 'E:/kuliahh/AI/tugas/backend_classification/fitur'
    FEATURE_TYPE = 'histogram'  # choose from 'histogram', 'glcm', or 'histogram_glcm'
    CLASSIFIER_TYPE = "mlp " # "mlp", "naive_bayes"

    # Create an instance of ImageClassifier and train the chosen classifier
    classifier = ImageClassifier(DATASET_DIR, MODEL_DIR, FEATURE_DIR, FEATURE_TYPE)
    classifier.load_data()
    classifier.train_classifier(CLASSIFIER_TYPE)
    classifier.save_classifier(CLASSIFIER_TYPE)
