/**
 * Copyright (c) 2020-2026 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#include "classification_postprocess.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "classification_labels.hpp"
#include "common/tensors.hpp"
#include "hailo_xtensor.hpp"

namespace
{
constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.20f;
constexpr const char *CLASSIFICATION_TYPE = "classification";
constexpr size_t IMAGENET_CLASS_COUNT = 1000;
constexpr size_t IMAGENET_WITH_BACKGROUND_CLASS_COUNT = 1001;

template <typename T>
int argmax_vec(const std::vector<T> &values)
{
    return static_cast<int>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

std::vector<float> softmax_vec(const std::vector<float> &logits)
{
    std::vector<float> probabilities;
    probabilities.reserve(logits.size());

    const float max_value = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (const auto value : logits)
    {
        const float probability = std::exp(value - max_value);
        probabilities.emplace_back(probability);
        sum += probability;
    }

    if (sum <= 0.0f)
    {
        return probabilities;
    }

    for (auto &probability : probabilities)
    {
        probability /= sum;
    }
    return probabilities;
}

bool scores_look_like_probabilities(const std::vector<float> &scores)
{
    if (scores.empty())
    {
        return false;
    }

    const bool all_normalized = std::all_of(scores.begin(), scores.end(), [](const auto score) {
        return std::isfinite(score) && score >= 0.0f && score <= 1.0f;
    });
    if (!all_normalized)
    {
        return false;
    }

    const float sum = std::accumulate(scores.begin(), scores.end(), 0.0f);
    return sum > 0.95f && sum < 1.05f;
}

std::vector<float> tensor_to_scores(HailoTensorPtr &tensor)
{
    const auto xtensor = common::get_xtensor_float(tensor);
    std::vector<float> scores;
    scores.reserve(xtensor.size());
    for (const auto score : xtensor)
    {
        scores.emplace_back(score);
    }
    return scores;
}

std::vector<std::string> read_custom_labels()
{
    std::vector<std::string> labels;
    const char *labels_path = std::getenv("HAILO_CLASSIFICATION_LABELS_PATH");
    if (labels_path == nullptr)
    {
        return labels;
    }

    std::ifstream labels_file(labels_path);
    std::string line;
    while (std::getline(labels_file, line))
    {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#')
        {
            continue;
        }
        const auto last = line.find_last_not_of(" \t\r\n");
        labels.emplace_back(line.substr(first, last - first + 1));
    }
    return labels;
}

std::string get_label(int class_id, size_t probability_count)
{
    static const std::vector<std::string> custom_labels = read_custom_labels();
    if (!custom_labels.empty())
    {
        if (class_id >= 0 && class_id < static_cast<int>(custom_labels.size()))
        {
            return custom_labels[class_id];
        }
        return "class_" + std::to_string(class_id);
    }

    int imagenet_class_id = class_id;
    if (probability_count == IMAGENET_WITH_BACKGROUND_CLASS_COUNT)
    {
        if (class_id == 0)
        {
            return "";
        }
        imagenet_class_id = class_id - 1;
    }

    if (imagenet_class_id < 0 || imagenet_class_id >= static_cast<int>(IMAGENET_CLASS_COUNT))
    {
        return "";
    }

    static ImageNetLabels labels;
    return labels.imagenet_labelstring(imagenet_class_id);
}
} // namespace

void filter(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }

    const auto tensors = roi->get_tensors();
    if (tensors.empty())
    {
        return;
    }

    HailoTensorPtr tensor = tensors[0];
    std::vector<float> scores = tensor_to_scores(tensor);
    if (scores.empty())
    {
        return;
    }

    const std::vector<float> probabilities = scores_look_like_probabilities(scores) ? scores : softmax_vec(scores);
    const int class_id = argmax_vec(probabilities);
    const float confidence = probabilities[class_id];

    if (confidence < DEFAULT_CONFIDENCE_THRESHOLD)
    {
        return;
    }

    const std::string label = get_label(class_id, probabilities.size());
    if (label.empty())
    {
        return;
    }

    roi->remove_objects_typed(HAILO_CLASSIFICATION);
    roi->add_object(std::make_shared<HailoClassification>(
        CLASSIFICATION_TYPE,
        class_id,
        label,
        confidence));
}
