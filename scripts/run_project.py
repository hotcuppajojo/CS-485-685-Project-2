import argparse
import sys
from pathlib import Path

# Add the parent directory of the script to the system path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))

import cv2
from image_processing.image_io import load_img, display_img
from image_processing.corner_detection import moravec_detector, harris_detector
from image_processing.feature_extraction import extract_LBP, extract_HOG
from feature_matching.matching import feature_matching
from visualization.plotting import plot_keypoints, plot_matches

def main(args):
    # Load and display the first image
    img1 = load_img(args.image1, grayscale=True)
    display_img(img1)
    
    # Detect and plot keypoints for the first image
    detector = moravec_detector if args.detector == 'Moravec' else harris_detector
    keypoints1 = detector(img1)
    img1_keypoints = plot_keypoints(img1, keypoints1)
    display_img(img1_keypoints)
    
    if args.image2:
        # Load and display the second image
        img2 = load_img(args.image2, grayscale=True)
        display_img(img2)
        
        # Detect and plot keypoints for the second image
        keypoints2 = detector(img2)
        img2_keypoints = plot_keypoints(img2, keypoints2)
        display_img(img2_keypoints)
        
        # Extract features and match them
        extractor = extract_LBP if args.extractor == 'LBP' else extract_HOG
        features1 = [extractor(img1, kp) for kp in keypoints1]
        features2 = [extractor(img2, kp) for kp in keypoints2]
        
        matches = feature_matching(img1, img2, args.detector, args.extractor)
        
        # Plot matches
        matched_image = plot_matches(img1, img2, matches)
        display_img(matched_image)
    else:
        print("Only one image provided. Feature matching requires two images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Project2 Image Processing Tasks")
    parser.add_argument('image1', type=str, help="Path to the first image")
    parser.add_argument('--image2', type=str, help="Path to the second image (optional)")
    parser.add_argument('--detector', type=str, choices=['Moravec', 'Harris'], default='Moravec', help="Feature detector to use")
    parser.add_argument('--extractor', type=str, choices=['LBP', 'HOG'], default='LBP', help="Feature extractor to use")
    
    args = parser.parse_args()
    main(args)