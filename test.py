import os
import shutil
import random
import yaml
from ultralytics import YOLO

def split_dataset(datapath, train_pct=0.9, output_base="C:\\Users\\Admin\\Downloads"):
    images_dir = os.path.join(datapath, 'images')
    labels_dir = os.path.join(datapath, 'labels')

    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_images.sort()
    random.shuffle(all_images)

    split_idx = int(len(all_images) * train_pct)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    def copy_files(image_list, split_name):
        image_out_dir = os.path.join(output_base, split_name, 'images')
        label_out_dir = os.path.join(output_base, split_name, 'labels')
        os.makedirs(image_out_dir, exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)

        for img in image_list:
            img_path = os.path.join(images_dir, img)
            label_name = os.path.splitext(img)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)

            shutil.copy(img_path, os.path.join(image_out_dir, img))
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(label_out_dir, label_name))
            else:
                print(f"⚠️ Warning: Label for {img} not found!")

    copy_files(train_images, 'train')
    copy_files(val_images, 'val')

    print(f"✅ Dataset split done! {len(train_images)} training and {len(val_images)} validation images.")


def create_data_yaml(path_to_classes_txt, path_to_data_yaml, base_path="C:\\Users\\Admin\\Downloads"):
    if not os.path.exists(path_to_classes_txt):
        print(f'❌ classes.txt not found at {path_to_classes_txt}')
        return

    with open(path_to_classes_txt, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    number_of_classes = len(classes)

    data = {
        'train': os.path.join(base_path, 'train', 'images'),
        'val': os.path.join(base_path, 'val', 'images'),
        'nc': number_of_classes,
        'names': classes
    }

    with open(path_to_data_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f'✅ Created data.yaml at {path_to_data_yaml}')
    print('\n📄 data.yaml content:\n')
    print(yaml.dump(data, sort_keys=False))


def main():
    dataset_path = "C:\\Users\\Admin\\Downloads\\archive\\yolo_output1"
    split_dataset(dataset_path)

    path_to_classes_txt = os.path.join(dataset_path, "classes.txt")
    path_to_data_yaml = os.path.join(dataset_path, "data.yaml")
    create_data_yaml(path_to_classes_txt, path_to_data_yaml)

    # ✅ Replace 'yolo11n.pt' with a valid model
    model = YOLO("yolov8n.pt")  # use yolov8n, yolov8s, etc.

    # ✅ Train with GPU
    results = model.train(
        data=path_to_data_yaml,
        epochs=20,
        imgsz=640,
        device='0'  # force to use GPU 0
    )


if __name__ == "__main__":
    main()
