/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <memory>

#include "common/macros.h"
#include "gaussian_process.h"
#include "resource_model.h"

namespace xllm_service {

// GP-based resource model for the steady pool.
// Uses 3 independent GPs with shared 3D input:
//   Input: [token_rate, avg_input_len, avg_output_len]
//   GP[hbm]:       -> per-GPU HBM usage (GB)
//   GP[compute]:   -> per-GPU compute SM utilization [0, 1]
//   GP[bandwidth]: -> per-GPU HBM-to-SRAM bandwidth utilization [0, 1]
class GPSteadyResourceModel final : public ResourceModel {
 public:
  GPSteadyResourceModel(std::unique_ptr<GaussianProcess> gp_hbm,
                         std::unique_ptr<GaussianProcess> gp_compute,
                         std::unique_ptr<GaussianProcess> gp_bandwidth);
  ~GPSteadyResourceModel() override = default;

  // Legacy interface: maps model_heat to token_rate with zero moments.
  int32_t compute_gpu_target(int64_t model_heat,
                             const GpuHardwareSpec& hw) const override;
  ResourceNeeds compute_resource_needs(int64_t model_heat) const override;
  std::string name() const override;

  // Primary GP interface: compute 3D resource needs from traffic stats.
  ResourceNeeds calc_3d_resources(double token_rate,
                                  double avg_input_len,
                                  double avg_input_len2,
                                  double avg_output_len) const override;

 private:
  DISALLOW_COPY_AND_ASSIGN(GPSteadyResourceModel);

  std::unique_ptr<GaussianProcess> gp_hbm_;
  std::unique_ptr<GaussianProcess> gp_compute_;
  std::unique_ptr<GaussianProcess> gp_bandwidth_;
};

}  // namespace xllm_service
