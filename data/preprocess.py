import os
import shutil

def preprocess_tiny_imagenet():
    val_annotations = './dataset/tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt'
    val_images_path = './dataset/tiny-imagenet/tiny-imagenet-200/val/images'
    
    with open(val_annotations) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(f'./dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)
            shutil.copyfile(f'{val_images_path}/{fn}', f'./dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')
    
    shutil.rmtree(val_images_path)
