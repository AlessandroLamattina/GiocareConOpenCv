import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

model_path = '/absolute/path/to/lite-model_efficientdet_lite0_detection_metadata_1.tflite'
BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.detections.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    print('detection result: {}'.format(result))


options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    result_callback=print_result)

with ObjectDetector.create_from_options(options) as detector:
    display_names

