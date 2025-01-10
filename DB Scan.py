import os
import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Counter

def str2bool(v: str) -> bool:
    if v.lower() in ['true', 'yes', 'on', 'y', 't']:
        return True
    elif v.lower() in ['false', 'no', 'off', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

def visualize(image, faces, print_flag=False, fps=None):
    output = image.copy()
    if fps:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    for idx, face in enumerate(faces):
        if print_flag:
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

        coords = face[:-1].astype(np.int32)
        cv.rectangle(output, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
        cv.circle(output, (coords[4], coords[5]), 2, (255, 0, 0), 2)
        cv.circle(output, (coords[6], coords[7]), 2, (0, 0, 255), 2)
        cv.circle(output, (coords[8], coords[9]), 2, (0, 255, 0), 2)
        cv.circle(output, (coords[10], coords[11]), 2, (255, 0, 255), 2)
        cv.circle(output, (coords[12], coords[13]), 2, (0, 255, 255), 2)
        cv.putText(output, '{:.4f}'.format(face[-1]), (coords[0], coords[1]+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    return output

def process_images_from_folder(folder_path, output_folder, model_path='face_detection_yunet_2023mar.onnx'):
    yunet = cv.FaceDetectorYN.create(
        model=model_path,
        config='',
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
        backend_id=cv.dnn.DNN_BACKEND_DEFAULT,
        target_id=cv.dnn.DNN_TARGET_CPU
    )

    embeddings = []
    image_face_map = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            image = cv.imread(file_path)
            if image is not None:
                yunet.setInputSize((image.shape[1], image.shape[0]))
                _, faces = yunet.detect(image)
                
                if faces is not None:
                    image_embeddings = []
                    for face in faces:
                        coords = face[:-1]
                        embedding = np.concatenate([
                            coords[:4],  # Bounding box
                            coords[4:],  # Landmarks
                            [(coords[2] * coords[3]) / (image.shape[0] * image.shape[1])],  # Face size ratio
                            [face[-1]]  # Detection confidence
                        ])
                        embeddings.append(embedding)
                        image_embeddings.append(embedding)
                    
                    image_face_map[file_path] = image_embeddings
                    
                    # Visualize detected faces (optional)
                    output_image = visualize(image, faces)
                    cv.imwrite(os.path.join(output_folder, f'vis_{filename}'), output_image)
    
    embeddings = np.array(embeddings)
    print(f'Extracted {len(embeddings)} embeddings')
    
    if len(embeddings) > 0:
        # Normalize the embeddings
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(embeddings)
        
        # Perform DBSCAN clustering with optimized parameters
        clustering = DBSCAN(eps=0.5, min_samples=2, metric='euclidean').fit(normalized_embeddings)
        labels = clustering.labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f'Number of clusters: {n_clusters}')
        
        # Create folders based on cluster labels
        for label in set(labels):
            if label != -1:
                label_folder = os.path.join(output_folder, f'cluster_{label}')
                os.makedirs(label_folder, exist_ok=True)
        
        noise_folder = os.path.join(output_folder, 'noise')
        os.makedirs(noise_folder, exist_ok=True)
        
        # Process and move images
        label_index = 0
        for image_path, image_embeddings in image_face_map.items():
            image_labels = labels[label_index:label_index+len(image_embeddings)]
            label_index += len(image_embeddings)
            
            # Determine the most common label for this image
            label_counts = Counter(image_labels)
            most_common_label = label_counts.most_common(1)[0][0]
            
            filename = os.path.basename(image_path)
            if most_common_label != -1:
                new_path = os.path.join(output_folder, f'cluster_{most_common_label}', filename)
            else:
                new_path = os.path.join(noise_folder, filename)
            
            os.rename(image_path, new_path)
            print(f'Moved {image_path} to {new_path}')
    else:
        print('No faces detected or no embeddings extracted.')

def main():
    folder_path = 'Group pics Classmates'
    output_folder = 'Clustered Pics'
    
    process_images_from_folder(folder_path, output_folder)

if __name__ == '__main__':
    main()