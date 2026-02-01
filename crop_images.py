
import cv2
import os

def crop_image(image_path, output_path, crop_factor=2):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    h, w, _ = img.shape
    new_h, new_w = h // crop_factor, w // crop_factor
    
    # Crop from the center
    start_h = (h - new_h) // 2
    start_w = (w - new_w) // 2
    
    cropped_img = img[start_h:start_h + new_h, start_w:start_w + new_w]
    
    cv2.imwrite(output_path, cropped_img)
    print(f"Cropped {image_path} and saved to {output_path}")

def main():
    data_dir = 'data'
    output_dir = 'data/cropped'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_files = ['heightmap.png', 'industries.png', 'map.png', 'promising_routes.png']
    
    for file_name in image_files:
        image_path = os.path.join(data_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        if os.path.exists(image_path):
            crop_image(image_path, output_path)

if __name__ == '__main__':
    main()
