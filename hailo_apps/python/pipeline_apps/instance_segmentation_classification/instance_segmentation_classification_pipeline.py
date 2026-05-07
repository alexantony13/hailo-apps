# region imports
# Standard library imports
import json
import os
import tempfile
from pathlib import Path

import setproctitle

from hailo_apps.python.core.common.core import (
    configure_multi_model_hef_path,
    get_pipeline_parser,
    get_resource_path,
    handle_list_models_flag,
    resolve_hef_paths,
)
from hailo_apps.python.core.common.defines import (
    ALL_DETECTIONS_CROPPER_POSTPROCESS_SO_FILENAME,
    CLASSIFICATION_POSTPROCESS_FUNCTION,
    CLASSIFICATION_POSTPROCESS_SO_FILENAME,
    INSTANCE_SEGMENTATION_CLASSIFICATION_APP_TITLE,
    INSTANCE_SEGMENTATION_CLASSIFICATION_CROPPER_POSTPROCESS_FUNCTION,
    INSTANCE_SEGMENTATION_CLASSIFICATION_NO_TRACKING_CROPPER_POSTPROCESS_FUNCTION,
    INSTANCE_SEGMENTATION_CLASSIFICATION_PIPELINE,
    INSTANCE_SEGMENTATION_MODEL_PREFIX_V5,
    INSTANCE_SEGMENTATION_MODEL_PREFIX_V8,
    INSTANCE_SEGMENTATION_POSTPROCESS_FUNCTION,
    INSTANCE_SEGMENTATION_POSTPROCESS_SO_FILENAME,
    INSTANCE_SEGMENTATION_POSTPROCESS_V8_SO_FILENAME,
    JSON_FILE_EXTENSION,
    RESOURCES_JSON_DIR_NAME,
    RESOURCES_SO_DIR_NAME,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback,
)
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    CROPPER_PIPELINE,
    DISPLAY_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
)

hailo_logger = get_logger(__name__)
# endregion imports


class GStreamerInstanceSegmentationClassificationApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_pipeline_parser()

        self._set_parser_argument_default(
            parser,
            "--batch-size",
            2,
            "Segmentation model batch size. Default is 2 for this application.",
        )
        self._set_parser_argument_default(
            parser,
            "--width",
            640,
            "Output width in pixels. Default is 640 for this application.",
        )
        self._set_parser_argument_default(
            parser,
            "--height",
            640,
            "Output height in pixels. Default is 640 for this application.",
        )
        self._make_parser_argument_repeatable(
            parser,
            "--labels",
            (
                "Path to a text file with custom class labels. Repeat once to provide "
                "separate labels in model order: --labels <segmentation_labels> "
                "--labels <classification_labels>. If supplied once, the same labels "
                "file is used for both models."
            ),
        )
        configure_multi_model_hef_path(parser)
        parser.add_argument(
            "--classification-batch-size",
            type=int,
            default=8,
            help="Batch size for the per-object classification model.",
        )
        parser.add_argument(
            "--max-classification-crops",
            type=int,
            default=8,
            help="Maximum detected-object crops to classify per frame.",
        )
        parser.add_argument(
            "--disable-tracking",
            action="store_true",
            help="Disable object tracking between segmentation and classification.",
        )
        handle_list_models_flag(parser, INSTANCE_SEGMENTATION_CLASSIFICATION_PIPELINE)

        hailo_logger.info("Initializing GStreamer Instance Segmentation Classification App...")
        super().__init__(parser, user_data)

        models = resolve_hef_paths(
            hef_paths=self.options_menu.hef_path,
            app_name=INSTANCE_SEGMENTATION_CLASSIFICATION_PIPELINE,
            arch=self.arch,
        )
        self.hef_path_segmentation = str(models[0].path)
        self.hef_path_classification = str(models[1].path)
        self.classification_batch_size = self.options_menu.classification_batch_size
        self.max_classification_crops = max(1, self.options_menu.max_classification_crops)
        self.disable_tracking = self.options_menu.disable_tracking
        user_data.tracking_enabled = not self.disable_tracking
        os.environ["HAILO_MAX_CLASSIFICATION_CROPS_PER_FRAME"] = str(
            self.max_classification_crops
        )
        self.segmentation_labels_path, self.classification_labels_path = self._get_label_paths(
            self.options_menu.labels
        )
        if self.classification_labels_path is not None:
            os.environ["HAILO_CLASSIFICATION_LABELS_PATH"] = str(self.classification_labels_path)
        else:
            os.environ.pop("HAILO_CLASSIFICATION_LABELS_PATH", None)

        self.config_file = self._get_segmentation_config_file(self.hef_path_segmentation)
        self.post_process_so_segmentation = self._get_segmentation_postprocess_so(
            self.hef_path_segmentation
        )
        self.post_process_so_classifier = get_resource_path(
            INSTANCE_SEGMENTATION_CLASSIFICATION_PIPELINE,
            RESOURCES_SO_DIR_NAME,
            self.arch,
            CLASSIFICATION_POSTPROCESS_SO_FILENAME,
        )
        self.post_process_so_cropper = get_resource_path(
            INSTANCE_SEGMENTATION_CLASSIFICATION_PIPELINE,
            RESOURCES_SO_DIR_NAME,
            self.arch,
            ALL_DETECTIONS_CROPPER_POSTPROCESS_SO_FILENAME,
        )

        self.app_callback = app_callback
        setproctitle.setproctitle(INSTANCE_SEGMENTATION_CLASSIFICATION_APP_TITLE)
        self.create_pipeline()

    @staticmethod
    def _set_parser_argument_default(parser, option_string, default, help_text):
        for action in parser._actions:
            if option_string in getattr(action, "option_strings", []):
                action.default = default
                action.help = help_text
                return

    @staticmethod
    def _make_parser_argument_repeatable(parser, option_string, help_text):
        for action in parser._actions[:]:
            if option_string in getattr(action, "option_strings", []):
                parser._remove_action(action)
                for group in parser._action_groups:
                    if action in group._group_actions:
                        group._group_actions.remove(action)
                for opt in action.option_strings:
                    if opt in parser._option_string_actions:
                        del parser._option_string_actions[opt]
                parser.add_argument(*action.option_strings, action="append", default=None, help=help_text)
                return

    @staticmethod
    def _get_label_paths(labels_option):
        if labels_option in (None, [], ""):
            return None, None
        label_paths = labels_option if isinstance(labels_option, list) else [labels_option]
        if len(label_paths) > 2:
            raise ValueError(
                "--labels accepts at most two files: segmentation labels, then classification labels."
            )
        for label_path in label_paths:
            if not Path(label_path).exists():
                raise FileNotFoundError(f"Labels file not found: {label_path}")
        if len(label_paths) == 1:
            return Path(label_paths[0]), Path(label_paths[0])
        return Path(label_paths[0]), Path(label_paths[1])

    @staticmethod
    def _read_labels_file(labels_path):
        with Path(labels_path).open("r", encoding="utf-8") as labels_file:
            return [
                line.strip()
                for line in labels_file
                if line.strip() and not line.lstrip().startswith("#")
            ]

    def _write_segmentation_config_with_labels(self, config_path):
        labels = self._read_labels_file(self.segmentation_labels_path)
        if not labels:
            raise ValueError(f"Labels file is empty: {self.segmentation_labels_path}")

        with Path(config_path).open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        config["labels"] = labels
        config["num_classes"] = len(labels)

        tmp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix="hailo_seg_labels_",
            delete=False,
        )
        with tmp_file:
            json.dump(config, tmp_file)
        return Path(tmp_file.name)

    def _validate_segmentation_model(self, hef_path):
        hef_stem = Path(hef_path).stem
        hef_name = Path(hef_path).name

        if INSTANCE_SEGMENTATION_MODEL_PREFIX_V5 in hef_stem:
            hailo_logger.info("Detected YOLOv5 seg model family (%s)", hef_stem)
        elif INSTANCE_SEGMENTATION_MODEL_PREFIX_V8 in hef_stem:
            hailo_logger.info("Detected YOLOv8 seg model family (%s)", hef_stem)
        else:
            raise ValueError(
                f"First --hef-path must be an instance segmentation model, got '{hef_name}'. "
                "Use order: --hef-path <yolov5*_seg|yolov8*_seg> --hef-path <classifier>."
            )

    def _get_segmentation_config_file(self, hef_path):
        hef_stem = Path(hef_path).stem

        self._validate_segmentation_model(hef_path)
        config_path = get_resource_path(
            INSTANCE_SEGMENTATION_CLASSIFICATION_PIPELINE,
            RESOURCES_JSON_DIR_NAME,
            self.arch,
            hef_stem + JSON_FILE_EXTENSION,
        )
        if self.segmentation_labels_path is not None:
            return self._write_segmentation_config_with_labels(config_path)
        return config_path

    def _get_segmentation_postprocess_so(self, hef_path):
        hef_stem = Path(hef_path).stem
        so_filename = (
            INSTANCE_SEGMENTATION_POSTPROCESS_V8_SO_FILENAME
            if INSTANCE_SEGMENTATION_MODEL_PREFIX_V8 in hef_stem
            else INSTANCE_SEGMENTATION_POSTPROCESS_SO_FILENAME
        )
        return get_resource_path(
            INSTANCE_SEGMENTATION_CLASSIFICATION_PIPELINE,
            RESOURCES_SO_DIR_NAME,
            self.arch,
            so_filename,
        )

    def get_pipeline_string(self):
        source_pipeline = self.get_source_pipeline()

        segmentation_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path_segmentation,
            post_process_so=self.post_process_so_segmentation,
            post_function_name=INSTANCE_SEGMENTATION_POSTPROCESS_FUNCTION,
            batch_size=self.batch_size,
            config_json=self.config_file,
            name="segmentation_inference",
        )
        segmentation_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(
            segmentation_pipeline,
            name="segmentation_wrapper",
        )

        tracker_pipeline = None
        if not self.disable_tracking:
            tracker_pipeline = TRACKER_PIPELINE(class_id=-1, keep_past_metadata=True)

        classification_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path_classification,
            post_process_so=self.post_process_so_classifier,
            post_function_name=CLASSIFICATION_POSTPROCESS_FUNCTION,
            batch_size=self.classification_batch_size,
            name="classification_inference",
        )
        classification_cropper_pipeline = CROPPER_PIPELINE(
            inner_pipeline=classification_pipeline,
            so_path=self.post_process_so_cropper,
            function_name=(
                INSTANCE_SEGMENTATION_CLASSIFICATION_NO_TRACKING_CROPPER_POSTPROCESS_FUNCTION
                if self.disable_tracking
                else INSTANCE_SEGMENTATION_CLASSIFICATION_CROPPER_POSTPROCESS_FUNCTION
            ),
            name="classification_cropper",
        )

        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink,
            sync=self.sync,
            show_fps=self.show_fps,
        )

        pipeline_stages = [
            source_pipeline,
            segmentation_pipeline_wrapper,
            tracker_pipeline,
            classification_cropper_pipeline,
            user_callback_pipeline,
            display_pipeline,
        ]
        return " ! ".join(stage for stage in pipeline_stages if stage)


def main():
    hailo_logger.info("Starting Hailo Instance Segmentation Classification App...")
    user_data = app_callback_class()
    app = GStreamerInstanceSegmentationClassificationApp(dummy_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
