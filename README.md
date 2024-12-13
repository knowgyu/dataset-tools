# Data Augmentation Using Albumentations

이미지 데이터셋에 다양한 변형을 적용하고 **YOLO 형식의 라벨**과 함께 데이터 증강을 수행하는 스크립트입니다.

---

## 설치 방법

필수 라이브러리를 설치합니다.

```bash
pip install albumentations opencv-python numpy tqdm
```

---

## 사용법

### 실행 명령어

```bash
python augment.py --input <input-folder> --output <output-folder> --n <augmentations-per-image>
```

### 파라미터 설명
- `--input`: 입력 폴더 경로 (이미지와 YOLO 라벨)
- `--output`: 출력 폴더 경로
- `--n`: 이미지당 생성할 증강 개수

---

## 예시

### 입력 폴더 구조
```plaintext
input_data/
    ├── image1.jpg
    ├── image1.txt
    ├── image2.jpg
    └── image2.txt
```

### 실행 예시
```bash
python augment.py --input ./input_data --output ./output_data --n 5
```

### 출력 폴더 구조
```plaintext
output_data/
    ├── aug_0_image1.jpg
    ├── aug_0_image1.txt
    ├── aug_1_image1.jpg
    └── aug_1_image1.txt
```

---

## 참고자료

- [Albumentations GitHub](https://github.com/albumentations-team/albumentations)