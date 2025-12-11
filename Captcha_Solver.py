"""
CAPTCHA Solver using Exact Template Matching
"""
import numpy as np
import os
import matplotlib.pyplot as plt

class Captcha(object):
    def __init__(self, training_dir=None, threshold=100):
        """
        Initialize CAPTCHA solver
        
        Args:
            training_dir: Path to directory with training input/output files
            threshold: Binarization threshold (default: 100)
        """
        self.threshold = threshold
        self.templates = {}  # Dictionary: flattened_image_tuple -> character
        
        if training_dir:
            self._build_templates(training_dir)
    
    def _load_image(self, path):
        """
        Load image from text file format
        
        Args:
            path: Path to .txt file
            
        Returns:
            numpy array (height, width) grayscale
        """
        try:
            with open(path, 'r') as f:
                h, w = map(int, f.readline().strip().split())
                
                img = np.zeros((h, w), dtype=np.uint8)
                for r in range(h):
                    row = f.readline().strip().split()
                    for c, pixel in enumerate(row):
                        img[r, c] = int(pixel.split(',')[0])  # Use R channel as grayscale
            return img
        except Exception as e:
            raise ValueError(f"Error loading image from {path}: {e}")
    
    def _preprocess(self, img):
        """
        Convert to binary image (ink=1, background=0)
        
        Args:
            img: Grayscale image array
            
        Returns:
            Binary image (ink=1, background=0)
        """
        return (img < self.threshold).astype(np.uint8)
    
    def _segment_characters(self, binary_img):
        """
        Segment binary image into individual characters
        
        Args:
            binary_img: Binary image (30x60)
            
        Returns:
            List of character images
        """
        try:
            # Column-wise sum to find character boundaries
            col_sum = binary_img.sum(axis=0)
            
            chars = []
            inside_char = False
            start_col = 0
            
            for col in range(len(col_sum)):
                if col_sum[col] > 0 and not inside_char:
                    # Start of character
                    inside_char = True
                    start_col = col
                elif col_sum[col] == 0 and inside_char:
                    # End of character
                    inside_char = False
                    chars.append(binary_img[:, start_col:col])
            
            # Handle last character if it extends to edge
            if inside_char:
                chars.append(binary_img[:, start_col:])
            
            return chars
        except Exception as e:
            raise ValueError(f"Error segmenting characters: {e}")
    
    def show_segments(self, image_path):
        """
        Display segmented characters for debugging
        
        Args:
            binary_img: Binary image (30x60)
        """
        img = self._load_image(image_path)
        binary_img = self._preprocess(img)
        segments = self._segment_characters(binary_img)
        
        n = len(segments)
        plt.figure(figsize=(2*n, 2))
        for i, seg in enumerate(segments):
            plt.subplot(1, n, i+1)
            plt.imshow(seg, cmap="gray")
            plt.title(f"char {i}")
            plt.axis("off")
        plt.show()
    
    def _image_to_key(self, char_img):
        """
        Convert character image to hashable key for dictionary lookup
        
        Args:
            char_img: Character image array
            
        Returns:
            Tuple (hashable key)
        """
        return tuple(char_img.flatten())
    
    def _build_templates(self, training_dir):
        """
        Build template dictionary from training data
        
        Args:
            training_dir: Directory containing input*.txt and output*.txt files
        """
        input_files = sorted([
            f for f in os.listdir(training_dir) 
            if f.startswith('input') and f.endswith('.txt')
        ])
        for input_file in input_files:
            output_file = input_file.replace('input', 'output')
            output_path = os.path.join(training_dir, output_file)
            
            if not os.path.exists(output_path):
                continue
            
            try:
                # Load label
                with open(output_path, 'r') as f:
                    label = f.readline().strip()
                
                # Process image
                input_path = os.path.join(training_dir, input_file)
                img = self._load_image(input_path)
                binary = self._preprocess(img)
                chars = self._segment_characters(binary)
                
                # Store each character as template
                if len(chars) == len(label):
                    for char_img, char_label in zip(chars, label):
                        key = self._image_to_key(char_img)
                        self.templates[key] = char_label
                        
            except Exception as e:
                print(f"Warning: Failed to process {input_file}: {e}")
        
        print(f"Built {len(self.templates)} templates from {len(input_files)} files")
    
    def _recognize(self, char_img):
        """
        Recognize a single character using template matching
        
        Args:
            char_img: Binary character image
            
        Returns:
            Recognized character or '?'
        """
        key = self._image_to_key(char_img)
        return self.templates.get(key, '?')
    
    def __call__(self, im_path, save_path):
        """
        Solve CAPTCHA and save result
        
        Args:
            im_path: Path to input .txt file
            save_path: Path to save output result
            
        Returns:
            Recognized string
        """
        # Load and preprocess
        img = self._load_image(im_path)
        binary = self._preprocess(img)
        
        # Segment into characters
        chars = self._segment_characters(binary)
        
        # Recognize each character
        result = ''.join([self._recognize(char) for char in chars])

        print(f"Successfully recognised CAPTCHA:", result)
        
        # Save result
        with open(save_path, 'w') as f:
            f.write(result)
        
        return result


# Usage example
if __name__ == "__main__":
    # Train the solver
    solver = Captcha(training_dir='sampleCaptchas/training')
    
    # Test on a file
    result = solver(
        im_path='./sampleCaptchas/input/input06.txt',
        save_path='./output06.txt'
    )
