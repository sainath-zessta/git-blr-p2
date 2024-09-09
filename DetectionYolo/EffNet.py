import os
import torch
import timm
import cv2
import numpy as np
import shutil
import json
from torchvision import transforms
from scipy.sparse import csr_matrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def vehicle_reid_process(detect_folder, output_dir, threshold=0.7):
    logging.info("Starting vehicle re-identification process")

    # Initialize model
    model = timm.create_model('efficientnet_b0', pretrained=True)
    model.eval().to('cuda')
    logging.info("Model initialized and set to evaluation mode")

    # Class mapping
    class_map = {
        'Bicycle': 0,
        'Bus': 1,
        'Car': 2,
        'LCV': 3,
        'Three-Wheeler': 4,
        'Truck': 5,
        'Two-Wheeler': 6
    }
    reverse_class_map = {v: k for k, v in class_map.items()}

    # Transformation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def load_and_preprocess_image(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img)
        return img

    def extract_features_from_folder(folder_path):
        features = []
        image_paths = []
        image_classes = []

        logging.info(f"Extracting features from folder: {folder_path}")
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                class_id = class_map.get(dir_name, -1)
                if class_id == -1:
                    continue

                dir_path = os.path.join(root, dir_name)
                for file in os.listdir(dir_path):
                    if file.endswith(".jpg") or file.endswith(".png"):
                        image_path = os.path.join(dir_path, file)
                        image = load_and_preprocess_image(image_path).to('cuda')
                        with torch.no_grad():
                            feature = model.forward_features(image.unsqueeze(0))
                            feature = feature.view(-1).cpu().numpy()

                            features.append(feature)
                            image_paths.append(image_path)
                            image_classes.append(class_id)

        return np.vstack(features), image_paths, image_classes

    def sparse_multiplication(all_features_sparse, all_paths, all_classes):
        logging.info("Calculating distances between features")
        distances = all_features_sparse.dot(all_features_sparse.T).toarray()
        matching_pairs = []
        for i in range(len(all_paths)):
            for j in range(i + 1, len(all_paths)):
                if distances[i, j] < threshold:
                    matching_pairs.append((all_paths[i], all_paths[j], all_classes[i]))
        logging.info(f"Found {len(matching_pairs)} matching pairs")
        return matching_pairs

    def save_matched_images(matches):
        images_folder = os.path.join(output_dir, "Images")
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
        
        class_match_ids = {cls_name: 1 for cls_name in reverse_class_map.values()}
        vehicle_id_map = {}

        logging.info("Saving matched images")
        for i, (img1_path, img2_path, class_id) in enumerate(matches):
            class_name = reverse_class_map.get(class_id, "Unknown")
            if class_name == "Unknown":
                continue
            
            match_id = class_match_ids[class_name]
            class_match_ids[class_name] += 1

            img1_folder = os.path.basename(os.path.dirname(img1_path))
            img1_name = os.path.splitext(os.path.basename(img1_path))[0]
            img2_folder = os.path.basename(os.path.dirname(img2_path))
            img2_name = os.path.splitext(os.path.basename(img2_path))[0]

            img1_output_name = f"{class_name}_{img1_folder}_{img1_name}_{match_id}.jpg"
            img2_output_name = f"{class_name}_{img2_folder}_{img2_name}_{match_id}.jpg"

            match_folder = os.path.join(images_folder, f"match_{match_id}")
            os.makedirs(match_folder, exist_ok=True)

            img1_output_path = os.path.join(match_folder, img1_output_name)
            img2_output_path = os.path.join(match_folder, img2_output_name)

            shutil.copy(img1_path, img1_output_path)
            shutil.copy(img2_path, img2_output_path)

            if img1_path not in vehicle_id_map:
                vehicle_id_map[img1_path] = f"{class_name}_{img1_folder}_{img1_name}_{match_id}"
            if img2_path not in vehicle_id_map:
                vehicle_id_map[img2_path] = f"{class_name}_{img2_folder}_{img2_name}_{match_id}"

        return vehicle_id_map

    def save_class_counts(image_paths, image_classes):
        matrices_folder = os.path.join(output_dir, "Matrices")
        if not os.path.exists(matrices_folder):
            os.makedirs(matrices_folder)

        logging.info("Calculating class counts and camera pair matrices")
        class_counts = {cls_id: [] for cls_id in class_map.values()}
        camera_folders = [os.path.basename(folder) for folder in predict_folders]
        num_cameras = len(camera_folders)
        camera_index_map = {camera: idx for idx, camera in enumerate(camera_folders)}
        camera_pair_matrices = {cls_id: np.zeros((num_cameras, num_cameras), dtype=int) for cls_id in class_map.values()}

        for folder in predict_folders:
            folder_class_counts = {cls_id: 0 for cls_id in class_map.values()}
            _, _, folder_classes = extract_features_from_folder(folder)
            folder_camera = os.path.basename(folder)
            
            for cls_id in folder_classes:
                if cls_id in folder_class_counts:
                    folder_class_counts[cls_id] += 1
            
            for cls_id, count in folder_class_counts.items():
                class_counts[cls_id].append(count)
            
            for other_folder in predict_folders:
                if folder != other_folder:
                    other_camera = os.path.basename(other_folder)
                    if folder_camera in camera_index_map and other_camera in camera_index_map:
                        for cls_id in class_map.values():
                            count_in_pair = sum(1 for c in folder_classes if c == cls_id)
                            camera_pair_matrices[cls_id][camera_index_map[folder_camera], camera_index_map[other_camera]] += count_in_pair

        # Save camera pair matrices
        for cls_id, matrix in camera_pair_matrices.items():
            class_name = reverse_class_map[cls_id]
            camera_pair_matrix_file = os.path.join(matrices_folder, f"{class_name}.json")
            with open(camera_pair_matrix_file, 'w') as f:
                json.dump(matrix.tolist(), f, indent=4)
        
        logging.info("Class counts and camera pair matrices saved")

    # Main process
    predict_folders = [os.path.join(root, d) for root, dirs, files in os.walk(detect_folder) for d in dirs]
    predict_folders.sort()

    all_features = []
    all_paths = []
    all_classes = []

    logging.info("Extracting features from all folders")
    for folder in predict_folders:
        features, paths, classes = extract_features_from_folder(folder)
        all_features.extend(features)
        all_paths.extend(paths)
        all_classes.extend(classes)

    all_features_np = np.array(all_features)
    all_features_sparse = csr_matrix(all_features_np)

    matching_pairs = sparse_multiplication(all_features_sparse, all_paths, all_classes)
    logging.info("Matching pairs identified")

    vehicle_id_map = save_matched_images(matching_pairs)
    logging.info("Matched images saved")

    save_class_counts(all_paths, all_classes)
    logging.info("Class counts and matrices saved")