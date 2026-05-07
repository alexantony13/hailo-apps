# region imports
# Standard library imports
import os

os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

# Third-party imports
import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")

# Local application-specific imports
import hailo
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.python.core.common.defines import CLASSIFICATION_TYPE
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.pipeline_apps.instance_segmentation_classification.instance_segmentation_classification_pipeline import (
    GStreamerInstanceSegmentationClassificationApp,
)

hailo_logger = get_logger(__name__)
# endregion imports


class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_skip = 2
        self.tracking_enabled = True


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 128),
    (255, 165, 0),
    (0, 128, 128),
    (128, 128, 0),
]


def _get_track_id(detection):
    track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
    if len(track) == 1:
        return track[0].get_id()
    return None


def _get_classification(detection):
    classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
    for classification in classifications:
        if classification.get_classification_type() == CLASSIFICATION_TYPE:
            return classification
    return classifications[0] if classifications else None


def _overlay_mask(reduced_frame, mask_overlay, detection, track_id):
    masks = detection.get_objects_typed(hailo.HAILO_CONF_CLASS_MASK)
    if len(masks) == 0 or reduced_frame is None:
        return

    mask = masks[0]
    mask_height = mask.get_height()
    mask_width = mask.get_width()
    if mask_height <= 0 or mask_width <= 0:
        return

    bbox = detection.get_bbox()
    data = np.array(mask.get_data()).reshape((mask_height, mask_width))

    roi_width = int(bbox.width() * reduced_frame.shape[1])
    roi_height = int(bbox.height() * reduced_frame.shape[0])
    if roi_width <= 0 or roi_height <= 0:
        return

    resized_mask_data = cv2.resize(data, (roi_width, roi_height), interpolation=cv2.INTER_LINEAR)

    x_min = int(bbox.xmin() * reduced_frame.shape[1])
    y_min = int(bbox.ymin() * reduced_frame.shape[0])
    x_max = x_min + roi_width
    y_max = y_min + roi_height

    y_min = max(y_min, 0)
    x_min = max(x_min, 0)
    y_max = min(y_max, reduced_frame.shape[0])
    x_max = min(x_max, reduced_frame.shape[1])

    if x_max <= x_min or y_max <= y_min:
        return

    color = COLORS[track_id % len(COLORS)]
    mask_overlay[y_min:y_max, x_min:x_max] = (
        resized_mask_data[: y_max - y_min, : x_max - x_min, np.newaxis] > 0.5
    ) * color


def _draw_classification(reduced_frame, detection, classification):
    if reduced_frame is None or classification is None:
        return

    bbox = detection.get_bbox()
    x_min = int(bbox.xmin() * reduced_frame.shape[1])
    y_min = int(bbox.ymin() * reduced_frame.shape[0])
    y_min = max(y_min - 8, 16)

    label = classification.get_label()
    confidence = classification.get_confidence()
    text = f"{label}: {confidence:.2f}"
    cv2.putText(
        reduced_frame,
        text,
        (max(x_min, 4), y_min),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        reduced_frame,
        text,
        (max(x_min, 4), y_min),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def app_callback(element, buffer, user_data):
    hailo_logger.debug("Callback triggered. Current frame count=%d", user_data.get_count())

    if buffer is None:
        hailo_logger.warning("Received None buffer in callback.")
        return

    if user_data.get_count() % user_data.frame_skip != 0:
        return

    string_to_print = f"Frame count: {user_data.get_count()}\n"

    pad = element.get_static_pad("src")
    video_format, width, height = get_caps_from_pad(pad)
    if width is None or height is None:
        return

    reduced_frame = None
    mask_overlay = None
    if user_data.use_frame and video_format is not None:
        frame = get_numpy_from_buffer(buffer, video_format, width, height)
        reduced_frame = cv2.resize(
            frame,
            (width // 4, height // 4),
            interpolation=cv2.INTER_AREA,
        )
        mask_overlay = np.zeros_like(reduced_frame)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    annotations = []

    for detection_index, detection in enumerate(detections):
        label = detection.get_label()
        confidence = detection.get_confidence()
        track_id = _get_track_id(detection) if user_data.tracking_enabled else None
        visual_id = track_id if track_id is not None else detection_index
        classification = _get_classification(detection)

        string_to_print += (
            f"Detection: ID: {track_id if track_id is not None else 'N/A'} "
            f"Label: {label} Confidence: {confidence:.2f}"
        )
        if classification is not None:
            string_to_print += (
                f" | Classification: {classification.get_label()} "
                f"({classification.get_confidence():.2f})"
            )
        string_to_print += "\n"

        if user_data.use_frame:
            _overlay_mask(reduced_frame, mask_overlay, detection, visual_id)
            annotations.append((detection, classification))

    print(string_to_print)

    if user_data.use_frame and reduced_frame is not None:
        if mask_overlay is not None:
            reduced_frame = cv2.addWeighted(reduced_frame, 1, mask_overlay, 0.5, 0)
        for detection, classification in annotations:
            _draw_classification(reduced_frame, detection, classification)
        reduced_frame = cv2.cvtColor(reduced_frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(reduced_frame)

    return


def main():
    hailo_logger.info("Starting Instance Segmentation Classification App.")
    user_data = user_app_callback_class()
    app = GStreamerInstanceSegmentationClassificationApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
