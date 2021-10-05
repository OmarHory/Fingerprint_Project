from minutiae_src import extract_minutiae_features
import cv2

img = cv2.imread('/home/omar/Desktop/F1_enhanced/F1_F1_L1.bmp', 0)
FeaturesTerminations, FeaturesBifurcations = extract_minutiae_features(img, showResult=False)

print(FeaturesTerminations[0].__dict__)
print(FeaturesBifurcations[0].__dict__)