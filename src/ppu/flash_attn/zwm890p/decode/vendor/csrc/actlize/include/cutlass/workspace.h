/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Utilities for initializing workspaces
*/

#pragma once

#if !defined(__HGGCCC_RTC__)
#include <hggc.h>
#include <hggc_runtime.h>

#include "cutlass/trace.h"
#endif

#include "cutlass.h"
#include "cutlass/ppu_host_adapter.hpp"

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int MinWorkspaceAlignment = 16;

#if !defined(__HGGCCC_RTC__)
static Status
zero_workspace(void* workspace, size_t workspace_size, hggcStream_t stream = nullptr, HostAdapter *host_adapter = nullptr) {
  if (workspace_size > 0) {
    if (workspace == nullptr) {
      CUTLASS_TRACE_HOST("  error: device workspace must not be null");
      return Status::kErrorWorkspaceNull;
    }

    CUTLASS_TRACE_HOST("  clearing workspace");

#if defined(CUTLASS_ENABLE_HOST_ADAPTER) && CUTLASS_ENABLE_HOST_ADAPTER
    //
    // Use the device host adapter
    //
    CUTLASS_ASSERT(host_adapter);
    if (host_adapter) {
      if (Status::kSuccess != host_adapter->memsetDevice(workspace, static_cast<uint8_t>(0), workspace_size, stream)) {
        return Status::kErrorInternal;
      }
    }
    else {
      return Status::kErrorInternal;
    }
#else
    hggcError_t result = hggcMemsetAsync(workspace, 0, workspace_size, stream);
    if (hggcSuccess != result) {
      result = hggcGetLastError(); // to clear the error bit
      CUTLASS_TRACE_HOST("  hggcMemsetAsync() returned error " << hggcGetErrorString(result));
      return Status::kErrorInternal;
    }
#endif
  }

  return Status::kSuccess;
}
#endif

#if !defined(__HGGCCC_RTC__)
template <typename T>
Status
fill_workspace(void* workspace, T fill_value, size_t fill_count, hggcStream_t stream = nullptr, HostAdapter *host_adapter = nullptr) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "Unsupported fill type");
  if (fill_count > 0) {
    if (workspace == nullptr) {
      CUTLASS_TRACE_HOST("  error: device workspace must not be null");
      return Status::kErrorWorkspaceNull;
    }

    CUTLASS_TRACE_HOST("  filling workspace");

#if defined(CUTLASS_ENABLE_HOST_ADAPTER) && CUTLASS_ENABLE_HOST_ADAPTER
    //
    // Use the device host adapter
    //
    CUTLASS_ASSERT(host_adapter);
    if (host_adapter) {
      if (Status::kSuccess != host_adapter->memsetDevice(workspace, fill_value, fill_count, stream)) {
        return Status::kErrorInternal;
      }
    }
    else {
      return Status::kErrorInternal;
    }
#else
    HGdeviceptr d_workspace = reinterpret_cast<HGdeviceptr>(workspace);
    HGresult result = HGGC_SUCCESS;
    if (sizeof(T) == 4) {
      result = hgMemsetD32Async(d_workspace, reinterpret_cast<uint32_t&>(fill_value), fill_count, stream);
    }
    else if (sizeof(T) == 2) {
      result = hgMemsetD16Async(d_workspace, reinterpret_cast<uint16_t&>(fill_value), fill_count, stream);
    }
    else if (sizeof(T) == 1) {
      result = hgMemsetD8Async(d_workspace, reinterpret_cast<uint8_t&>(fill_value), fill_count, stream);
    }

    if (HGGC_SUCCESS != result) {
      const char** error_string_ptr = nullptr;
      (void) hgGetErrorString(result, error_string_ptr);
      if (error_string_ptr != nullptr) {
        CUTLASS_TRACE_HOST("  hgMemsetD" << sizeof(T) * 8 << "Async() returned error " << *error_string_ptr);
      }
      else {
        CUTLASS_TRACE_HOST("  hgMemsetD" << sizeof(T) * 8 << "Async() returned unrecognized error");
      }
      return Status::kErrorInternal;
    }
#endif
  }

  return Status::kSuccess;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
