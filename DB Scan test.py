import numpy as np
import cv2 as cv
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def visualize(image, faces, labels, cluster_centers, print_flag=False, fps=None):
    output = image.copy()
    if fps:
        cv.putText(output, f'FPS: {fps:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if faces is not None:
        for idx, face in enumerate(faces):
            if isinstance(face, np.ndarray) and face.shape[0] >= 15:
                coords = face[:4].astype(np.int32)
                score = face[-1]
                label = labels[idx]
                color = (0, 255, 0) if label != -1 else (0, 0, 255)  # -1 is the noise label in DBSCAN
                cv.rectangle(output, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), color, 2)
                cv.putText(output, f'Cluster {label}', (coords[0], coords[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv.putText(output, f'{score:.4f}', (coords[0], coords[1]-30), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Draw cluster centers
                if label != -1:  # Only draw cluster centers for valid clusters
                    center = cluster_centers[label]
                    cv.circle(output, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

    return output

def process_folder(folder_path):
    model_path = 'face_detection_yunet_2023mar.onnx'
    score_threshold = 0.6
    nms_threshold = 0.3
    top_k = 5000
    vis = True
    save = True
    eps = 0.5
    min_samples = 5

    yunet = cv.FaceDetectorYN.create(
        model=model_path,
        config='',
        input_size=(320, 320),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
        backend_id=cv.dnn.DNN_BACKEND_DEFAULT,
        target_id=cv.dnn.DNN_TARGET_CPU
    )

    all_features = []
    image_paths = []

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv.imread(image_path)

            if image is not None:
                yunet.setInputSize((image.shape[1], image.shape[0]))
                _, faces = yunet.detect(image)

                if faces is not None and faces.size > 0:
                    faces = faces.reshape(-1, 15)  # Ensure correct shape
                    for face in faces:
                        if face.shape[0] >= 15:  # Ensure each face has the correct number of elements
                            features = face[:4]  # Use bounding box coordinates as features
                            all_features.append(features)
                            image_paths.append((image_path, face))
    
    # Convert features to numpy array and standardize
    all_features = np.array(all_features)
    if len(all_features) == 0:
        print("No faces detected in the images.")
        return

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)

    # Perform clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_features)
    unique_labels = np.unique(labels)
    
    # Compute cluster centers (mean of points in each cluster)
    cluster_centers = []
    for label in unique_labels:
        if label != -1:  # Skip noise points
            cluster_points = scaled_features[labels == label]
            cluster_centers.append(np.mean(cluster_points, axis=0))
    cluster_centers = np.array(cluster_centers)

    # Visualize and save results
    for image_path, faces in image_paths:
        image = cv.imread(image_path)
        vis_image = visualize(image, faces, labels, cluster_centers)
        
        if save:
            output_path = os.path.join(folder_path, f'clustered_{os.path.basename(image_path)}')
            cv.imwrite(output_path, vis_image)
            print(f'Result saved as {output_path}.')
        
        if vis:
            cv.imshow('Face Detection', vis_image)
            cv.waitKey(0)
            cv.destroyAllWindows()

if __name__ == '__main__':
    folder_path = 'Group pics Classmates'
    process_folder(folder_path)
