import os
import cv2
import json
import pickle
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import argparse
from pathlib import Path
import random


class ImageAttacker:
    """Class to apply various attacks on images and generate JSON and PKL metadata."""
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.processed_images = []  
        self.query_images = []  
        self.attack_images = []  # 存储所有攻击图像 (jpeg, crop, blur, noise, hybrid等)
        
        # [新增] 内部弱攻击的构建模块 (用于混合攻击)
        self._weak_attack_functions = [
            self._apply_blur,
            self._apply_noise,
            self._apply_brightness,
            self._apply_contrast,
            self._apply_paint
        ]

    # --- 1. 基础弱攻击：JPEG ---
    def apply_jpeg_compression(self, image_path, quality_factors=None):
        if quality_factors is None:
            quality_factors = [75, 50, 30, 10]
        
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]     #SCID数据集中的base就是SCI
        output_paths = []  #初始化空列表
        jpeg_indices = []  
        img = Image.open(image_path)
        
        for qf in quality_factors:
            if self.output_dir:
                output_path = os.path.join(self.output_dir, f"jpegqual_{qf}_{base_name}.jpg")
            else:
                output_path = os.path.join(os.path.dirname(image_path), f"jpegqual_{qf}_{base_name}.jpg")
                
            img.save(output_path, "JPEG", quality=qf)
            output_paths.append(output_path)
            
            output_filename = os.path.splitext(os.path.basename(output_path))[0]
            self.attack_images.append(output_filename) #从完整路径中提取文件 basename（包含扩展名），然后去除扩展名，得到纯文件名 # 例如："/path/to/jpegqual_75_image.jpg" → "jpegqual_75_image",将其添加进入攻击图像列表
            idx = len(self.attack_images) - 1  #索引从0开始
            jpeg_indices.append(idx)    #列表添加
            
            print(f"Created JPEG compressed image with quality {qf}: {output_path}")
        
        return output_paths, jpeg_indices
    
    # --- 2. 基础弱攻击：Cropping ---
    def apply_cropping(self, image_path, crop_percentages=None):
        if crop_percentages is None:
            crop_percentages = [10, 30, 50, 70]
        
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]
        output_paths = []
        crop_indices = []   #裁剪指数
        img = Image.open(image_path)
        width, height = img.size  #img.size: 这是 PIL.Image 类的一个属性，返回一个包含图像宽度和高度的元组 (width, height)
        
        for percentage in crop_percentages:
            crop_ratio = 1.0 - (percentage / 100.0)  #裁剪比例
            crop_width = int(width * crop_ratio)
            crop_height = int(height * crop_ratio)
            left = np.random.randint(0, width - crop_width + 1)
            top = np.random.randint(0, height - crop_height + 1)
            right = left + crop_width
            bottom = top + crop_height
            cropped_img = img.crop((left, top, right, bottom))
            
            if self.output_dir:
                output_path = os.path.join(self.output_dir, f"crops_{percentage}_{base_name}.jpg")  # 如果有指定输出目录，则保存到该目录下
            else:
                output_path = os.path.join(os.path.dirname(image_path), f"crops_{percentage}_{base_name}.jpg")   
                
            cropped_img.save(output_path, "JPEG", quality=95)  # 以JPEG格式保存裁剪后的图像，质量设置为95
            output_paths.append(output_path)  
            
            output_filename = os.path.splitext(os.path.basename(output_path))[0]
            self.attack_images.append(output_filename)
            idx = len(self.attack_images) - 1
            crop_indices.append(idx)
            
            print(f"Created cropped image with {percentage}% removed: {output_path}")
        
        return output_paths, crop_indices

    # --- 3. [新增] 弱攻击：Blur ---
    def apply_blur(self, image_path, radii=None):
        if radii is None:
            radii = [2, 3, 5]  #半径
        
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]
        output_paths = []
        blur_indices = []  
        img = Image.open(image_path)
        
        for r in radii:
            output_path = os.path.join(self.output_dir or os.path.dirname(image_path), f"blur_{r}_{base_name}.jpg")
            attacked_img = self._apply_blur(img, radius=r)
            attacked_img.save(output_path, "JPEG", quality=95)
            output_paths.append(output_path)
            
            output_filename = os.path.splitext(os.path.basename(output_path))[0]
            self.attack_images.append(output_filename)
            idx = len(self.attack_images) - 1
            blur_indices.append(idx)
            print(f"Created blur image with radius {r}: {output_path}")
        
        return output_paths, blur_indices

    # --- 4. [新增] 弱攻击：Noise ---
    def apply_noise(self, image_path, levels=None):
        if levels is None:
            levels = [5, 10, 15] # 噪声百分比
        
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]
        output_paths = []
        noise_indices = []
        img = Image.open(image_path)
        
#    当指定了 output_dir 时使用绝对路径
#    output_path = os.path.join(self.output_dir, f"filename.jpg")
   
#    当未指定 output_dir 时使用相对路径（相对于原图像目录）
#    output_path = os.path.join(os.path.dirname(image_path), f"filename.jpg")
   

        for level in levels:
            output_path = os.path.join(self.output_dir or os.path.dirname(image_path), f"noise_{level}_{base_name}.jpg")
            attacked_img = self._apply_noise(img, level_percent=level)
            attacked_img.save(output_path, "JPEG", quality=95)
            output_paths.append(output_path)

            
            output_filename = os.path.splitext(os.path.basename(output_path))[0]
            # 只提取文件名，不包含路径信息
            self.attack_images.append(output_filename)
            idx = len(self.attack_images) - 1
            noise_indices.append(idx)
            print(f"Created noise image with level {level}%: {output_path}")
        
        return output_paths, noise_indices
    
    # --- 5. [新增] 弱攻击：Brightness ---
    def apply_brightness(self, image_path, factors=None):
        if factors is None:
            factors = [0.5, 0.7, 1.3, 1.5]   #因素
        
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]
        output_paths = []
        brightness_indices = []
        img = Image.open(image_path)
        
        for f in factors:
            output_path = os.path.join(self.output_dir or os.path.dirname(image_path), f"brightness_{f:.1f}_{base_name}.jpg")
            attacked_img = self._apply_brightness(img, factor=f)
            attacked_img.save(output_path, "JPEG", quality=95)
            output_paths.append(output_path)
            
            output_filename = os.path.splitext(os.path.basename(output_path))[0]
            self.attack_images.append(output_filename)
            idx = len(self.attack_images) - 1
            brightness_indices.append(idx)
            print(f"Created brightness image with factor {f}: {output_path}")
        
        return output_paths, brightness_indices

    # --- 6. [新增] 弱攻击：Contrast ---
    def apply_contrast(self, image_path, factors=None):
        if factors is None:
            factors = [0.5, 0.7, 1.3, 1.5]
        
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]
        output_paths = []
        contrast_indices = []
        img = Image.open(image_path)
        
        for f in factors:
            output_path = os.path.join(self.output_dir or os.path.dirname(image_path), f"contrast_{f:.1f}_{base_name}.jpg")
            attacked_img = self._apply_contrast(img, factor=f)
            attacked_img.save(output_path, "JPEG", quality=95)
            output_paths.append(output_path)
            
            output_filename = os.path.splitext(os.path.basename(output_path))[0]
            self.attack_images.append(output_filename)
            idx = len(self.attack_images) - 1
            contrast_indices.append(idx)
            print(f"Created contrast image with factor {f}: {output_path}")
        
        return output_paths, contrast_indices

    # --- 7. [新增] 弱攻击：Paint ---
    def apply_paint(self, image_path, strokes_list=None):
        if strokes_list is None:
            strokes_list = [(5, 50), (10, 70)] # (stroke_size, num_strokes)
        
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]
        output_paths = []
        paint_indices = []
        img = Image.open(image_path)
        
        for i, (size, num) in enumerate(strokes_list):
            output_path = os.path.join(self.output_dir or os.path.dirname(image_path), f"paint_{i+1}_{base_name}.jpg")
            attacked_img = self._apply_paint(img, stroke_size=size, num_strokes=num)
            attacked_img.save(output_path, "JPEG", quality=95)
            output_paths.append(output_path)
            
            output_filename = os.path.splitext(os.path.basename(output_path))[0]
            self.attack_images.append(output_filename)
            idx = len(self.attack_images) - 1
            paint_indices.append(idx)
            print(f"Created paint image (level {i+1}): {output_path}")
        
        return output_paths, paint_indices

    # --- 8. [重写] 强/混合攻击：Hybrid ---
    def apply_hybrid_attacks(self, image_path, num_attacks=10):
        """[重写]
        Apply various HYBRID attacks by COMBINING weak attacks.
        """
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]
        output_paths = []
        hybrid_indices = []
        img = Image.open(image_path)
        
        for i in range(num_attacks):
            attacked_img = img.copy()
            
            # 随机选择 2 或 3 个弱攻击进行组合
            num_to_combine = random.randint(2, 3)
            attacks_to_combine = random.sample(self._weak_attack_functions, num_to_combine)
            
            # 随机应用这些攻击
            for attack_func in attacks_to_combine:
                # 使用 *随机参数* 调用内部辅助函数
                if attack_func == self._apply_blur:
                    attacked_img = self._apply_blur(attacked_img, radius=random.choice([2, 3, 5]))
                elif attack_func == self._apply_noise:
                    attacked_img = self._apply_noise(attacked_img, level_percent=random.randint(5, 15))
                elif attack_func == self._apply_brightness:
                    attacked_img = self._apply_brightness(attacked_img, factor=random.choice([0.5, 0.7, 1.3, 1.5]))
                elif attack_func == self._apply_contrast:
                    attacked_img = self._apply_contrast(attacked_img, factor=random.choice([0.5, 0.7, 1.3, 1.5]))
                elif attack_func == self._apply_paint:
                    attacked_img = self._apply_paint(attacked_img, stroke_size=random.choice([3, 5]), num_strokes=random.randint(30, 70))
            
            output_path = os.path.join(self.output_dir or os.path.dirname(image_path), f"hybrid_{i+1}_{base_name}.jpg")
            attacked_img.save(output_path, "JPEG", quality=95) # 以高质量保存混合结果
            output_paths.append(output_path)
            
            output_filename = os.path.splitext(os.path.basename(output_path))[0]
            self.attack_images.append(output_filename)
            idx = len(self.attack_images) - 1
            hybrid_indices.append(idx)
            
            print(f"Created hybrid attack image {i+1}: {output_path}")
        
        return output_paths, hybrid_indices
    
    # --- 内部辅助函数 (现在接受参数) ---
    
    def _apply_blur(self, img, radius=3):
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _apply_noise(self, img, level_percent=10):
        img_array = np.array(img).astype(np.int32)
        h, w, c = img_array.shape
        num_pixels = int((h * w * level_percent) / 100)

        for _ in range(num_pixels):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            for i in range(c):
                value = int(img_array[y, x, i]) + np.random.randint(-50, 50)
                img_array[y, x, i] = max(0, min(255, value))

        return Image.fromarray(img_array.astype(np.uint8))
    
    def _apply_brightness(self, img, factor=1.3):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def _apply_contrast(self, img, factor=1.3):
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def _apply_paint(self, img, stroke_size=5, num_strokes=50):
        painted_img = img.copy()
        draw = ImageDraw.Draw(painted_img)
        width, height = img.size
        
        for _ in range(num_strokes):
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(0, height)
            x2 = min(width, x1 + np.random.randint(-50, 50))
            y2 = min(height, y1 + np.random.randint(-50, 50))
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            draw.line((x1, y1, x2, y2), fill=color, width=stroke_size)
        
        return painted_img
    
    # --- [重写] process_image ---
    def process_image(self, image_path):
        """
        [重写]
        Process a single image with all *new* weak and hybrid attacks.
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        self.query_images.append(base_name)
        
        print(f"\n--- Processing Query Image: {base_name} ---")
        
        # Apply all weak attacks
        _, jpeg_indices = self.apply_jpeg_compression(image_path)
        _, crop_indices = self.apply_cropping(image_path)
        _, blur_indices = self.apply_blur(image_path)
        _, noise_indices = self.apply_noise(image_path)
        _, brightness_indices = self.apply_brightness(image_path)
        _, contrast_indices = self.apply_contrast(image_path)
        _, paint_indices = self.apply_paint(image_path)
        
        # Apply hybrid attacks
        _, hybrid_indices = self.apply_hybrid_attacks(image_path)
        
        # Store results for metadata generation (using indices)
        variant_results = {
            "jpegqual": jpeg_indices,
            "crops": crop_indices,
            "blur": blur_indices,
            "noise": noise_indices,
            "brightness": brightness_indices,
            "contrast": contrast_indices,
            "paint": paint_indices,
            "hybrid": hybrid_indices # Renamed from "strong"
        }
        
        self.processed_images.append(variant_results)
        return variant_results
    
    # --- [不变] generate_metadata ---
    def generate_metadata(self, json_path, pkl_path=None):
        if pkl_path is None:
            pkl_path = os.path.splitext(json_path)[0] + '.pkl'
        
        data = {
            "gnd": self.processed_images,  # Ground-truth (list of dicts)
            "imlist": self.attack_images,  # Database images (all attacks)
            "qimlist": self.query_images   # Query images (originals)
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nGenerated JSON metadata file: {json_path}")
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Generated PKL metadata file: {pkl_path}")
        print(f"- Query images: {len(self.query_images)}")
        print(f"- Database images: {len(self.attack_images)}")


def main():
    parser = argparse.ArgumentParser(description="Apply various attacks to images and generate metadata")
    parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    parser.add_argument("--output", "-o", default=None, help="Output directory (if not specified, save in same directory as source)")
    parser.add_argument("--json", "-j", default="dataset_metadata.json", help="Output JSON file path")
    parser.add_argument("--pkl", "-p", default=None, help="Output PKL file path (if not specified, derive from JSON path)")
    args = parser.parse_args()
    
    attacker = ImageAttacker(output_dir=args.output)
    
    input_path = Path(args.input)
    if input_path.is_file():
        attacker.process_image(str(input_path))
    elif input_path.is_dir():
        image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            image_files.extend(list(input_path.glob(ext)))
        
        print(f"Found {len(image_files)} images in {input_path}")
        for file_path in image_files:
            attacker.process_image(str(file_path))
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return
    
    attacker.generate_metadata(args.json, args.pkl)

if __name__ == "__main__":
    main()