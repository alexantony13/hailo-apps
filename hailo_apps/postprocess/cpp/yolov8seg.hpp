/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 *
 * YOLOv8 Instance Segmentation postprocess header.
 **/
#pragma once

#include "hailo_objects.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

__BEGIN_DECLS

class Yolov8segParams
{
public:
    float iou_threshold;
    float score_threshold;
    std::vector<int> input_shape;
    std::vector<int> strides;

    Yolov8segParams()
    {
        iou_threshold = 0.6;
        score_threshold = 0.25;
        input_shape = {640, 640};
        strides = {32, 16, 8};
    }
};

Yolov8segParams *init(const std::string config_path, const std::string function_name);
void yolov8seg(HailoROIPtr roi, void *params_void_ptr);
void free_resources(void *params_void_ptr);
void filter(HailoROIPtr roi, void *params_void_ptr);
void filter_letterbox(HailoROIPtr roi, void *params_void_ptr);

__END_DECLS
