import numpy as np
import cv2 as cv

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
        # Draw face bounding box
        cv.rectangle(output, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
        # Draw landmarks
        cv.circle(output, (coords[4], coords[5]), 2, (255, 0, 0), 2)
        cv.circle(output, (coords[6], coords[7]), 2, (0, 0, 255), 2)
        cv.circle(output, (coords[8], coords[9]), 2, (0, 255, 0), 2)
        cv.circle(output, (coords[10], coords[11]), 2, (255, 0, 255), 2)
        cv.circle(output, (coords[12], coords[13]), 2, (0, 255, 255), 2)
        # Put score
        cv.putText(output, '{:.4f}'.format(face[-1]), (coords[0], coords[1]+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    return output

def main():
    backends = (cv.dnn.DNN_BACKEND_DEFAULT,
                cv.dnn.DNN_BACKEND_HALIDE,
                cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
                cv.dnn.DNN_BACKEND_OPENCV)
    targets = (cv.dnn.DNN_TARGET_CPU,
               cv.dnn.DNN_TARGET_OPENCL,
               cv.dnn.DNN_TARGET_OPENCL_FP16,
               cv.dnn.DNN_TARGET_MYRIAD)

    # Hardcoded paths
    model_path = 'face_detection_yunet_2023mar.onnx'
    image_path = 'Group pics Classmates/IMG_1777.JPG'

    # Inference parameters
    score_threshold = 0.6
    nms_threshold = 0.3
    top_k = 5000
    vis = True
    save = False

    # Instantiate yunet
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

    # Process image
    image = cv.imread(image_path)

    if image is not None:
        yunet.setInputSize((image.shape[1], image.shape[0]))
        _, faces = yunet.detect(image) # faces: None, or nx15 np.array

        vis_image = visualize(image, faces)
        if save:
            print('result.jpg saved.')
            cv.imwrite('result.jpg', vis_image)
        if vis:
            cv.namedWindow('Face Detection', cv.WINDOW_AUTOSIZE)
            cv.imshow('Face Detection', vis_image)
            cv.waitKey(0)
    else:
        print('Image not found or unable to open.')

if __name__ == '__main__':
    main()
