---

# Face Detection with OpenCV DNN Module

This project demonstrates face detection using OpenCV's DNN module with the **libfacedetection** model. It supports both image and real-time video input using ONNX models, offering high-speed inference with various computation backends.

---

## Features
- Detects faces with bounding boxes and facial landmarks.
- Displays FPS (Frames Per Second) for real-time performance.
- Supports multiple computation backends and target devices:
  - **Backends:** Default, Halide, Intel's OpenVINO, OpenCV
  - **Targets:** CPU, OpenCL, OpenCL FP16, VPU
- Configurable confidence and NMS thresholds.
- Option to visualize or save the result image.

---

## Requirements
- Python 3.x
- OpenCV with DNN module
- NumPy

Install dependencies using:
```bash
pip install numpy opencv-python-headless
```

---

## Usage
### 1. Detect Faces in an Image:
```bash
python main.py --input <path_to_image> --model <path_to_onnx_model>
```

### 2. Real-Time Face Detection using Camera:
```bash
python main.py --model <path_to_onnx_model>
```

### 3. Save Detection Result:
```bash
python main.py --input <path_to_image> --model <path_to_onnx_model> --save True
```

---

## Arguments
- `--input`, `-i`: Path to the input image. If omitted, the default camera is used.
- `--model`, `-m`: Path to the ONNX model file.
- `--backend`: Computation backend. Default is OpenCV.
- `--target`: Target computation device. Default is CPU.
- `--score_threshold`: Confidence threshold for face detection (default: 0.6).
- `--nms_threshold`: Non-Max Suppression threshold (default: 0.3).
- `--top_k`: Maximum number of faces to detect (default: 5000).
- `--vis`: Visualize result (default: True).
- `--save`: Save output as `result.jpg` (default: False).

---

## Example Commands
### Image Detection:
```bash
python main.py --input sample.jpg --model yunet.onnx --vis True --save True
```

### Real-Time Detection:
```bash
python main.py --model yunet.onnx
```

---

## Output
- Displays the image with detected faces, bounding boxes, landmarks, and confidence scores.
- Real-time video output with FPS displayed.

---

## Acknowledgements
- Utilizes OpenCV's **FaceDetectorYN** class for face detection.
- Based on **libfacedetection** ONNX model for fast and accurate face detection.

---

## License
This project is licensed under the MIT License.

---

## Author
**Sanjay P N**  
For any questions or support, feel free to reach out!

---

This README covers all essential aspects of your project, including installation, usage, and configuration. If you need any modifications or additional details, let me know!
