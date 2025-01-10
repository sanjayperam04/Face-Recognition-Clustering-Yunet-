import os
import cv2
import face_recognition
import numpy as np
import shutil

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def load_images_from_folder(folder):
    images = []
    image_names = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            image = face_recognition.load_image_file(img_path)
            images.append(image)
            image_names.append(filename)
    return images, image_names

def cluster_faces(images, image_names):
    face_encodings = []
    for image in images:
        encodings = face_recognition.face_encodings(image)
        if encodings:
            face_encodings.append(encodings[0])
        else:
            face_encodings.append(None)

    unique_faces = []
    clustered_images = {}
    
    for i, encoding in enumerate(face_encodings):
        if encoding is not None:
            matches = face_recognition.compare_faces(unique_faces, encoding)
            if True in matches:
                index = matches.index(True)
                clustered_images[index].append(image_names[i])
            else:
                unique_faces.append(encoding)
                clustered_images[len(unique_faces) - 1] = [image_names[i]]
        else:
            # Handle images with no faces detected
            clustered_images[-1] = clustered_images.get(-1, []) + [image_names[i]]

    return clustered_images

def save_clustered_images(clustered_images, source_folder, output_folder):
    for cluster_id, image_names in clustered_images.items():
        cluster_folder = os.path.join(output_folder, f'cluster_{cluster_id}')
        create_folder(cluster_folder)
        for image_name in image_names:
            src_path = os.path.join(source_folder, image_name)
            dst_path = os.path.join(cluster_folder, image_name)
            shutil.copy(src_path, dst_path)

def main(source_folder, output_folder):
    images, image_names = load_images_from_folder(source_folder)
    clustered_images = cluster_faces(images, image_names)
    save_clustered_images(clustered_images, source_folder, output_folder)
    print(f"Images have been clustered into {len(clustered_images)} folders.")

if __name__ == "__main__":
    source_folder = "Group pics Classmates"
    output_folder = "Clustered Pics"
    main(source_folder, output_folder)