import gradio as gr
from image_processing.image_io import load_img
from image_processing.corner_detection import moravec_detector, harris_detector
from image_processing.feature_extraction import extract_LBP, extract_HOG
from feature_matching.matching import feature_matching
from visualization.plotting import plot_keypoints, plot_matches

def process_image(image1, image2, detector, extractor):
    img1 = load_img(image1.name, grayscale=True)
    keypoints1 = moravec_detector(img1) if detector == 'Moravec' else harris_detector(img1)
    img1_with_keypoints = plot_keypoints(img1, keypoints1)

    if image2:
        img2 = load_img(image2.name, grayscale=True)
        matches = feature_matching(img1, img2, detector, extractor)
        img_with_matches = plot_matches(img1, img2, matches)
        return img1_with_keypoints, img_with_matches

    return img1_with_keypoints, None

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="file", label="Image 1"),
        gr.Image(type="file", label="Image 2 (optional)"),
        gr.Radio(["Moravec", "Harris"], label="Detector"),
        gr.Radio(["LBP", "HOG"], label="Extractor")
    ],
    outputs=[
        gr.Image(type="numpy", label="Image with Keypoints"),
        gr.Image(type="numpy", label="Matched Image")
    ],
    title="Project2 Image Processing",
    description="Upload an image to detect and display keypoints. Optionally, upload a second image for feature matching."
)

iface.launch(server_name="0.0.0.0", server_port=8000)