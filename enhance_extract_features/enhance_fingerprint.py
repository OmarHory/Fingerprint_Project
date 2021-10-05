import fingerprint_enhancer	
import cv2
import argparse
import os

#usage: python enhance_fingerprint.py --dir DIRECTORY_OF_IAMGES --out_dir OUTPUT_DIRECTORY

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help='dir of images')
parser.add_argument('--out_dir', type=str, required=True, help='dir of enhanced images')


args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

images = [path for path in os.listdir(args.dir)]
for path in images:
    path = os.path.join(args.dir, path)
    img = cv2.imread(path, 0)
    out = fingerprint_enhancer.enhance_Fingerprint(img)
    cv2.imwrite(os.path.join(args.out_dir, os.path.basename(path)), out)