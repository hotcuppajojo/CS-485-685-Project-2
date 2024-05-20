import os

# Set environment variables for Matplotlib and Gradio
os.environ['MPLCONFIGDIR'] = '/tmp/'
os.environ['FLAGGING_DIR'] = '/tmp/flagged'

import gradio as gr
from image_processing.image_io import load_img
from image_processing.corner_detection import moravec_detector, harris_detector
from image_processing.feature_extraction import extract_LBP, extract_HOG
from feature_matching.matching import feature_matching
from visualization.plotting import plot_keypoints, plot_matches

def process_image(image1, image2, detector, extractor):
    try:
        # Load the first image
        img1 = load_img(image1, grayscale=True)
        print("Image 1 loaded successfully")
        
        # Detect keypoints in the first image
        if detector == 'Moravec':
            keypoints1 = moravec_detector(img1)
        else:
            keypoints1 = harris_detector(img1)
        print(f"Detected {len(keypoints1)} keypoints in Image 1 using {detector} detector")
        
        # Plot keypoints in the first image
        img1_with_keypoints = plot_keypoints(img1, keypoints1)
        
        if image2:
            # Load the second image
            img2 = load_img(image2, grayscale=True)
            print("Image 2 loaded successfully")
            
            # Detect keypoints in the second image
            if detector == 'Moravec':
                keypoints2 = moravec_detector(img2)
            else:
                keypoints2 = harris_detector(img2)
            print(f"Detected {len(keypoints2)} keypoints in Image 2 using {detector} detector")
            
            # Extract features from keypoints
            if extractor == 'LBP':
                features1 = [extract_LBP(img1, kp) for kp in keypoints1]
                features2 = [extract_LBP(img2, kp) for kp in keypoints2]
            else:
                features1 = [extract_HOG(img1, kp) for kp in keypoints1]
                features2 = [extract_HOG(img2, kp) for kp in keypoints2]
            print(f"Extracted features using {extractor} extractor")
            
            # Match features
            matches = feature_matching(img1, img2, detector, extractor)
            print(f"Found {len(matches)} matches")
            
            # Plot matches
            img_with_matches = plot_matches(img1, img2, matches)
            return img1_with_keypoints, img_with_matches

        return img1_with_keypoints, None
    
    except Exception as e:
        print(f"Error: {e}")
        return None, None

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="filepath", label="Image 1"),
        gr.Image(type="filepath", label="Image 2 (optional)"),
        gr.Radio(["Moravec", "Harris"], label="Detector"),
        gr.Radio(["LBP", "HOG"], label="Extractor")
    ],
    outputs=[
        gr.Image(type="numpy", label="Image with Keypoints"),
        gr.Image(type="numpy", label="Matched Image")
    ],
    title="Project2 Image Processing",
    description="Upload an image to detect and display keypoints. Optionally, upload a second image for feature matching.",
    flagging_dir=os.getenv('FLAGGING_DIR')  # Set the flagging directory to /tmp/flagged
)

iface.launch(server_name="0.0.0.0", server_port=7860)