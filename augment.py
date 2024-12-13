import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import argparse

# 변형 설정
transform = A.Compose([
    A.FancyPCA(alpha=0.1, p=0.5),  # 색상 변형
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),  # 밝기/대비
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=15, p=0.5),  # 색상 변형
    A.OneOf([
        A.GaussNoise(var_limit=(1, 7), p=0.5),  # 가우시안 잡음 추가
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
        A.JpegCompression(quality_lower=90, quality_upper=100, p=0.5)
    ], p=0.7),
    A.OneOf([
        A.Blur(blur_limit=(1, 2), p=0.5),  # 블러
        A.CLAHE(p=0.5)  # 히스토그램 균등화
    ], p=0.25)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 데이터 증강 함수
def apply_transformations(image, bboxes, class_labels):
    """
    Albumentations 변형을 적용하여 증강된 이미지를 반환합니다.
    """
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return transformed['image'], transformed['bboxes'], transformed['class_labels']

# 증강된 데이터를 저장하는 함수
def save_augmented_data(image_file, augmented_image, augmented_bboxes, augmented_class_labels, output_dir):
    """
    증강된 이미지와 라벨을 YOLO 형식으로 저장합니다.
    """
    # 이미지 저장
    cv2.imwrite(os.path.join(output_dir, image_file), augmented_image)
    # 라벨 파일 저장
    label_file = image_file.replace('.jpg', '.txt')
    with open(os.path.join(output_dir, label_file), 'w') as f:
        for i in range(len(augmented_bboxes)):
            f.write(f"{augmented_class_labels[i]} {' '.join(map(str, augmented_bboxes[i]))}\n")

# 메인 함수
def main(input_dir, output_dir, n):
    """
    입력 디렉토리에서 이미지를 읽어 증강하고 출력 디렉토리에 저장합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 입력 이미지 리스트
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    
    for image_file in tqdm(image_files, desc="Processing Images"):
        # 이미지 읽기
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        
        # 라벨 파일 읽기
        label_file = image_file.replace('.jpg', '.txt')
        with open(os.path.join(input_dir, label_file), 'r') as f:
            lines = f.readlines()
        
        # 바운딩 박스와 클래스 레이블 추출
        bboxes = np.array([line.split()[1:5] for line in lines], dtype=float)
        class_labels = np.array([line.split()[0] for line in lines], dtype=int)

        # 이미지 증강 및 저장
        for i in range(n):
            try:
                augmented_image, augmented_bboxes, augmented_class_labels = apply_transformations(image, bboxes, class_labels)
                output_image_file = f"aug_{i}_{image_file}"
                save_augmented_data(output_image_file, augmented_image, augmented_bboxes, augmented_class_labels, output_dir)
            except Exception as e:
                print(f"에러: {e}, 파일 이름: {image_file}")

# 스크립트 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Augmentation Script')
    parser.add_argument('--input', required=True, help='Path to the input directory')
    parser.add_argument('--output', required=True, help='Path to the output directory')
    parser.add_argument('--n', required=True, type=int, help='Number of augmentations per image')

    args = parser.parse_args()
    main(args.input, args.output, args.n)
