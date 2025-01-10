import os
import numpy as np
from PIL import Image
import face_recognition
from sklearn.cluster import DBSCAN
from collections import defaultdict
import logging
import shutil
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import concurrent.futures
import cv2
import uuid

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Store match result information"""
    cluster_id: int
    matched_images: List[str]
    confidence_scores: List[float]

class FaceClusteringSystem:
    def __init__(self, 
                 min_face_size: int = 20, 
                 tolerance: float = 0.6,
                 debug_mode: bool = True):
        self.min_face_size = min_face_size
        self.tolerance = tolerance
        self.debug_mode = debug_mode
        self.face_clusters = defaultdict(list)
        self.cluster_encodings = {}  # Store encodings for each cluster
        
    def _load_and_detect_faces(self, image_path: str) -> List[Tuple[np.ndarray, tuple]]:
        """Load image and detect faces, returning encodings and locations"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Detect faces using both models if needed
            face_locations = face_recognition.face_locations(image, model="cnn")
            if not face_locations:
                face_locations = face_recognition.face_locations(image, model="hog")
            
            if not face_locations:
                return []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            return list(zip(face_encodings, face_locations))
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return []

    def process_directory(self, directory_path: str):
        """Process directory and create clusters"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            os.path.join(directory_path, f) 
            for f in os.listdir(directory_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        # Process all images and collect face encodings
        all_encodings = []
        image_paths = []
        face_locations = []
        
        for image_path in image_files:
            faces = self._load_and_detect_faces(image_path)
            for encoding, location in faces:
                all_encodings.append(encoding)
                image_paths.append(image_path)
                face_locations.append(location)
        
        if not all_encodings:
            logger.warning("No faces found in directory")
            return
        
        # Perform clustering
        clustering = DBSCAN(
            eps=self.tolerance,
            min_samples=1,
            metric="euclidean",
            n_jobs=-1
        )
        
        clustering.fit(all_encodings)
        
        # Organize clusters
        self.face_clusters.clear()
        self.cluster_encodings.clear()
        
        for idx, label in enumerate(clustering.labels_):
            self.face_clusters[label].append((
                image_paths[idx],
                face_locations[idx],
                all_encodings[idx]
            ))
            
            # Store representative encoding for each cluster
            if label not in self.cluster_encodings:
                self.cluster_encodings[label] = all_encodings[idx]
        
        logger.info(f"Created {len(self.face_clusters)} clusters from {len(image_files)} images")

    def find_matching_images(self, query_image_path: str, output_dir: str = None) -> Optional[MatchResult]:
        """
        Find and return images containing matching faces
        
        Args:
            query_image_path: Path to query image
            output_dir: Optional directory to save matched images
            
        Returns:
            MatchResult object with matching information
        """
        # Detect faces in query image
        query_faces = self._load_and_detect_faces(query_image_path)
        if not query_faces:
            logger.error("No faces detected in query image")
            return None
        
        best_matches = None
        best_cluster_id = None
        best_confidence = float('inf')
        
        # For each face in query image
        for query_encoding, _ in query_faces:
            # Compare with each cluster
            for cluster_id, cluster_encoding in self.cluster_encodings.items():
                # Calculate distance to cluster representative
                distance = np.linalg.norm(query_encoding - cluster_encoding)
                
                if distance <= self.tolerance and distance < best_confidence:
                    best_confidence = distance
                    best_cluster_id = cluster_id
                    best_matches = self.face_clusters[cluster_id]
        
        if best_matches is None:
            logger.info("No matches found")
            return None
        
        # Extract unique image paths and confidence scores
        matched_images = []
        confidence_scores = []
        seen_images = set()
        
        for img_path, _, encoding in best_matches:
            if img_path not in seen_images:
                matched_images.append(img_path)
                # Calculate confidence score (1 - normalized distance)
                distance = np.linalg.norm(query_encoding - encoding)
                confidence = 1 - (distance / self.tolerance)
                confidence_scores.append(confidence)
                seen_images.add(img_path)
        
        # Save matched images if output directory is provided
        if output_dir:
            self._save_matched_images(
                matched_images,
                query_image_path,
                output_dir,
                best_cluster_id
            )
        
        return MatchResult(
            cluster_id=best_cluster_id,
            matched_images=matched_images,
            confidence_scores=confidence_scores
        )
    
    def _save_matched_images(self, 
                           matched_images: List[str],
                           query_image_path: str,
                           output_dir: str,
                           cluster_id: int):
        """Save matched images to output directory"""
        # Create output directory
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Copy query image
        query_filename = os.path.basename(query_image_path)
        query_output = os.path.join(cluster_dir, f"query_{query_filename}")
        shutil.copy2(query_image_path, query_output)
        
        # Copy matched images
        for idx, image_path in enumerate(matched_images):
            # Create a filename that includes confidence score
            filename = os.path.basename(image_path)
            base, ext = os.path.splitext(filename)
            output_path = os.path.join(cluster_dir, f"match_{idx+1}_{base}{ext}")
            shutil.copy2(image_path, output_path)
        
        logger.info(f"Saved {len(matched_images)} matched images to {cluster_dir}")

def main():
    # Example usage
    system = FaceClusteringSystem(tolerance=0.6)
    
    # Process directory of images
    image_dir = "/Users/rupinajay/Developer/Yunet-Face-Recog-Clustering/Group pics Classmates"
    system.process_directory(image_dir)
    
    # Find matches for a query image and save results
    query_image = "/Users/rupinajay/Downloads/IMG_1775.JPG"
    output_dir = "/Users/rupinajay/Developer/Yunet-Face-Recog-Clustering/Clustering"
    
    matches = system.find_matching_images(query_image, output_dir)
    
    if matches:
        print(f"\nFound matches in cluster {matches.cluster_id}:")
        for img_path, confidence in zip(matches.matched_images, matches.confidence_scores):
            print(f"- {img_path} (confidence: {confidence:.2%})")
    else:
        print("No matches found")

if __name__ == "__main__":
    main()