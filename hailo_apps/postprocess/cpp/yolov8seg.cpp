#include "yolov8seg.hpp"

#include "common/labels/coco_eighty.hpp"
#include "common/math.hpp"
#include "common/nms.hpp"
#include "common/tensors.hpp"
#include "hailo_common.hpp"
#include "json_config.hpp"
#include "mask_decoding.hpp"

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/schema.h"

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#if __GNUC__ > 8
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

using namespace xt::placeholders;

namespace
{
constexpr int REGRESSION_LENGTH = 15;
constexpr int BOX_CHANNELS = 4 * (REGRESSION_LENGTH + 1);
constexpr int SCORE_CHANNELS = 80;
constexpr int MASK_COEFFICIENTS = 32;

struct ScaleTensors
{
    HailoTensorPtr boxes;
    HailoTensorPtr scores;
    HailoTensorPtr masks;
    int stride;
};

float dequantize_value(uint8_t value, float qp_scale, float qp_zp)
{
    return (float(value) - qp_zp) * qp_scale;
}

std::vector<xt::xarray<double>> get_centers(const std::vector<int> &strides, const std::vector<int> &network_dims)
{
    std::vector<xt::xarray<double>> centers(strides.size());

    for (uint i = 0; i < strides.size(); i++)
    {
        int strided_width = network_dims[0] / strides[i];
        int strided_height = network_dims[1] / strides[i];

        xt::xarray<int> grid_x = xt::arange(0, strided_width);
        xt::xarray<int> grid_y = xt::arange(0, strided_height);
        auto mesh = xt::meshgrid(grid_x, grid_y);
        grid_x = std::get<1>(mesh);
        grid_y = std::get<0>(mesh);

        // Use standard meshgrid indexing to build centers
        auto ct_row = (xt::flatten(grid_y) + 0.5) * strides[i];
        auto ct_col = (xt::flatten(grid_x) + 0.5) * strides[i];
        centers[i] = xt::stack(xt::xtuple(ct_col, ct_row, ct_col, ct_row), 1);
    }

    return centers;
}

int stride_from_tensor(const HailoTensorPtr &tensor, int input_width)
{
    return input_width / static_cast<int>(tensor->height());
}

std::vector<ScaleTensors> collect_scale_tensors(const std::vector<HailoTensorPtr> &tensors, int input_width)
{
    std::map<int, ScaleTensors> grouped;

    for (const auto &tensor : tensors)
    {
        if (tensor->shape().size() != 3)
        {
            continue;
        }

        int channels = static_cast<int>(tensor->shape()[2]);
        int stride = stride_from_tensor(tensor, input_width);
        auto &group = grouped[stride];
        group.stride = stride;

        if (channels == BOX_CHANNELS)
        {
            group.boxes = tensor;
        }
        else if (channels == SCORE_CHANNELS)
        {
            group.scores = tensor;
        }
        else if (channels == MASK_COEFFICIENTS && stride != 4)
        {
            group.masks = tensor;
        }
    }

    std::vector<ScaleTensors> scales;
    for (auto &[_, group] : grouped)
    {
        if (group.boxes && group.scores && group.masks)
        {
            scales.push_back(group);
        }
    }

    std::sort(scales.begin(), scales.end(), [](const ScaleTensors &a, const ScaleTensors &b) {
        return a.stride < b.stride;
    });
    return scales;
}

HailoTensorPtr find_proto_tensor(const std::vector<HailoTensorPtr> &tensors, int input_width)
{
    for (const auto &tensor : tensors)
    {
        if (tensor->shape().size() != 3)
        {
            continue;
        }

        int channels = static_cast<int>(tensor->shape()[2]);
        int stride = stride_from_tensor(tensor, input_width);
        if (channels == MASK_COEFFICIENTS && stride == 4)
        {
            return tensor;
        }
    }

    return nullptr;
}

void fill_dequantized_box(
    std::array<float, BOX_CHANNELS> &decoded_box,
    const xt::xarray<uint8_t> &quantized_boxes,
    int proposal_index,
    float qp_scale,
    float qp_zp)
{
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < REGRESSION_LENGTH + 1; j++)
        {
            decoded_box[i * (REGRESSION_LENGTH + 1) + j] =
                dequantize_value(quantized_boxes(proposal_index, i, j), qp_scale, qp_zp);
        }
    }
}

std::vector<HailoDetection> decode_scale(
    const ScaleTensors &scale,
    const xt::xarray<double> &centers,
    const std::vector<int> &network_dims,
    float score_threshold)
{
    auto boxes_ptr = scale.boxes;
    auto scores_ptr = scale.scores;
    auto masks_ptr = scale.masks;

    auto box_tensor = common::get_xtensor(boxes_ptr);
    auto score_tensor = common::dequantize(
        common::get_xtensor(scores_ptr),
        scores_ptr->quant_info().qp_scale,
        scores_ptr->quant_info().qp_zp);
    auto mask_tensor = common::dequantize(
        common::get_xtensor(masks_ptr),
        masks_ptr->quant_info().qp_scale,
        masks_ptr->quant_info().qp_zp);

    int num_proposals = box_tensor.shape(0) * box_tensor.shape(1);
    auto quantized_boxes = xt::reshape_view(box_tensor, {num_proposals, 4, REGRESSION_LENGTH + 1});
    auto masks = xt::reshape_view(mask_tensor, {num_proposals, MASK_COEFFICIENTS});
    auto regression_distance = xt::reshape_view(xt::arange(0, REGRESSION_LENGTH + 1), {1, 1, REGRESSION_LENGTH + 1});

    std::vector<HailoDetection> detections;
    detections.reserve(num_proposals / 4);

    for (int proposal_index = 0; proposal_index < num_proposals; proposal_index++)
    {
        auto score_offset = proposal_index * SCORE_CHANNELS;
        int class_index = 0;
        float confidence = score_tensor.data()[score_offset];
        for (int c = 1; c < SCORE_CHANNELS; c++)
        {
            float score = score_tensor.data()[score_offset + c];
            if (score > confidence)
            {
                confidence = score;
                class_index = c;
            }
        }
        if (confidence < score_threshold)
        {
            continue;
        }

        std::array<float, BOX_CHANNELS> box{};
        fill_dequantized_box(
            box,
            quantized_boxes,
            proposal_index,
            boxes_ptr->quant_info().qp_scale,
            boxes_ptr->quant_info().qp_zp);
        common::softmax_2D(box.data(), 4, REGRESSION_LENGTH + 1);

        auto box_view = xt::adapt(box, {4UL, static_cast<size_t>(REGRESSION_LENGTH + 1)});
        auto box_distance = box_view * regression_distance;
        xt::xarray<float> reduced_distances = xt::sum(box_distance, {2});
        auto strided_distances = reduced_distances * scale.stride;

        auto distance_view1 = xt::view(strided_distances, xt::all(), xt::range(_, 2)) * -1;
        auto distance_view2 = xt::view(strided_distances, xt::all(), xt::range(2, _));
        auto distance_view = xt::concatenate(xt::xtuple(distance_view1, distance_view2), 1);
        auto decoded_box = centers + distance_view;

        float xmin = decoded_box(proposal_index, 0) / network_dims[0];
        float ymin = decoded_box(proposal_index, 1) / network_dims[1];
        float xmax = decoded_box(proposal_index, 2) / network_dims[0];
        float ymax = decoded_box(proposal_index, 3) / network_dims[1];

        if (!std::isfinite(xmin) || !std::isfinite(ymin) || !std::isfinite(xmax) || !std::isfinite(ymax))
        {
            continue;
        }

        xmin = std::clamp(xmin, 0.0f, 1.0f);
        ymin = std::clamp(ymin, 0.0f, 1.0f);
        xmax = std::clamp(xmax, 0.0f, 1.0f);
        ymax = std::clamp(ymax, 0.0f, 1.0f);

        float width = xmax - xmin;
        float height = ymax - ymin;
        const float min_norm_w = 2.0f / static_cast<float>(network_dims[0]);
        const float min_norm_h = 2.0f / static_cast<float>(network_dims[1]);
        if (width <= min_norm_w || height <= min_norm_h)
        {
            continue;
        }

        // Reject boxes that become zero-sized after integer quantization in pixel space.
        int net_xmin = static_cast<int>(std::floor(xmin * network_dims[0]));
        int net_xmax = static_cast<int>(std::ceil(xmax * network_dims[0]));
        int net_ymin = static_cast<int>(std::floor(ymin * network_dims[1]));
        int net_ymax = static_cast<int>(std::ceil(ymax * network_dims[1]));
        if ((net_xmax - net_xmin) <= 0 || (net_ymax - net_ymin) <= 0)
        {
            continue;
        }

        // Also validate proto-space crop dimensions (proto is stride 4 in this model).
        int proto_w = network_dims[0] / 4;
        int proto_h = network_dims[1] / 4;
        int proto_xmin = static_cast<int>(std::floor(xmin * proto_w));
        int proto_xmax = static_cast<int>(std::ceil(xmax * proto_w));
        int proto_ymin = static_cast<int>(std::floor(ymin * proto_h));
        int proto_ymax = static_cast<int>(std::ceil(ymax * proto_h));
        if ((proto_xmax - proto_xmin) <= 0 || (proto_ymax - proto_ymin) <= 0)
        {
            continue;
        }

        HailoBBox bbox(xmin, ymin, width, height);

        int hailo_class_index = class_index + 1;
        if (hailo_class_index < 0 || hailo_class_index >= static_cast<int>(common::coco_eighty.size()))
        {
            continue;
        }
        HailoDetection detection(bbox, hailo_class_index, common::coco_eighty[hailo_class_index], confidence);

        auto mask_coefficients = xt::eval(xt::view(masks, proposal_index, xt::all()));
        std::vector<float> mask_data(mask_coefficients.size());
        memcpy(mask_data.data(), mask_coefficients.data(), sizeof(float) * mask_coefficients.size());
        detection.add_object(std::make_shared<HailoMatrix>(mask_data, mask_coefficients.size(), 1));

        detections.push_back(detection);
    }
    return detections;
}

std::vector<HailoDetection> yolov8seg_post(
    const std::vector<HailoTensorPtr> &tensors,
    const std::vector<int> &network_dims,
    float iou_threshold,
    float score_threshold)
{
    auto scales = collect_scale_tensors(tensors, network_dims[0]);
    if (scales.empty())
    {
        return {};
    }

    std::vector<int> strides;
    strides.reserve(scales.size());
    for (const auto &scale : scales)
    {
        strides.push_back(scale.stride);
    }
    auto centers = get_centers(strides, network_dims);

    std::vector<HailoDetection> all_detections;
    for (size_t i = 0; i < scales.size(); i++)
    {
        auto detections = decode_scale(scales[i], centers[i], network_dims, score_threshold);
        all_detections.insert(all_detections.end(), detections.begin(), detections.end());
    }

    common::nms(all_detections, iou_threshold);
    return all_detections;
}
} // namespace

Yolov8segParams *init(const std::string config_path, const std::string function_name)
{
    auto *params = new Yolov8segParams();
    (void)function_name;
    if (!fs::exists(config_path))
    {
        return params;
    }

    char config_buffer[4096];
    const char *json_schema = R""""({
      "$schema": "http://json-schema.org/draft-07/schema#",
      "type": "object",
      "properties": {
        "iou_threshold": {"type": "number"},
        "score_threshold": {"type": "number"},
        "input_shape": {"type": "array", "items": {"type": "number"}}
      }
    })"""";

    std::FILE *fp = fopen(config_path.c_str(), "r");
    if (fp == nullptr)
    {
        throw std::runtime_error("JSON config file is not valid");
    }
    rapidjson::FileReadStream stream(fp, config_buffer, sizeof(config_buffer));
    if (common::validate_json_with_schema(stream, json_schema))
    {
        rapidjson::Document doc;
        doc.ParseStream(stream);
        if (doc.HasMember("iou_threshold"))
        {
            params->iou_threshold = doc["iou_threshold"].GetFloat();
        }
        if (doc.HasMember("score_threshold"))
        {
            params->score_threshold = doc["score_threshold"].GetFloat();
        }
        if (doc.HasMember("input_shape"))
        {
            auto config_input_shape = doc["input_shape"].GetArray();
            params->input_shape.clear();
            for (uint j = 0; j < config_input_shape.Size(); j++)
            {
                params->input_shape.push_back(config_input_shape[j].GetInt());
            }
        }
    }
    fclose(fp);
    return params;
}

void free_resources(void *params_void_ptr)
{
    delete reinterpret_cast<Yolov8segParams *>(params_void_ptr);
}

void filter(HailoROIPtr roi, void *params_void_ptr)
{
    auto *params = reinterpret_cast<Yolov8segParams *>(params_void_ptr);
    if (!roi->has_tensors())
    {
        return;
    }

    std::vector<int> network_dims = {params->input_shape[0], params->input_shape[1]};
    auto tensors = roi->get_tensors();
    auto proto_tensor_ptr = find_proto_tensor(tensors, network_dims[0]);
    if (proto_tensor_ptr == nullptr)
    {
        return;
    }

    auto detections = yolov8seg_post(tensors, network_dims, params->iou_threshold, params->score_threshold);
    auto proto_tensor = common::dequantize(
        common::get_xtensor(proto_tensor_ptr),
        proto_tensor_ptr->quant_info().qp_scale,
        proto_tensor_ptr->quant_info().qp_zp);

    decode_masks(detections, proto_tensor);
    hailo_common::add_detections(roi, detections);
}

void yolov8seg(HailoROIPtr roi, void *params_void_ptr)
{
    filter(roi, params_void_ptr);
}

void filter_letterbox(HailoROIPtr roi, void *params_void_ptr)
{
    filter(roi, params_void_ptr);
    HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
    auto detections = hailo_common::get_hailo_detections(roi);
    constexpr float kMinBoxSize = 1.0f / 640.0f;
    for (auto &detection : detections)
    {
        auto detection_bbox = detection->get_bbox();
        float xmin = (detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin();
        float ymin = (detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin();
        float xmax = (detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin();
        float ymax = (detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin();

        if (!std::isfinite(xmin) || !std::isfinite(ymin) || !std::isfinite(xmax) || !std::isfinite(ymax))
        {
            continue;
        }

        xmin = std::clamp(xmin, 0.0f, 1.0f);
        ymin = std::clamp(ymin, 0.0f, 1.0f);
        xmax = std::clamp(xmax, 0.0f, 1.0f);
        ymax = std::clamp(ymax, 0.0f, 1.0f);

        if (xmax <= xmin)
        {
            xmax = std::min(1.0f, xmin + kMinBoxSize);
        }
        if (ymax <= ymin)
        {
            ymax = std::min(1.0f, ymin + kMinBoxSize);
        }

        float width = xmax - xmin;
        float height = ymax - ymin;
        if (width < kMinBoxSize)
        {
            width = kMinBoxSize;
            if (xmin + width > 1.0f)
            {
                xmin = std::max(0.0f, 1.0f - width);
            }
        }
        if (height < kMinBoxSize)
        {
            height = kMinBoxSize;
            if (ymin + height > 1.0f)
            {
                ymin = std::max(0.0f, 1.0f - height);
            }
        }

        detection->set_bbox(HailoBBox(xmin, ymin, width, height));
    }
    roi->clear_scaling_bbox();
}
