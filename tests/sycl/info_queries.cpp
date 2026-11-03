/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#include <iostream>
#include <vector>
#include <algorithm>

#include "sycl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(info_queries, reset_device_fixture)

BOOST_AUTO_TEST_CASE(device_queries) {
  
  sycl::device d{sycl::default_selector{}};
  sycl::queue q{d};

  std::string device = d.get_info<sycl::info::device::name>();
  BOOST_TEST(device.length() > 0);
  std::cout << "Default-selected queue runs on device: " << device << std::endl;

  BOOST_CHECK_NO_THROW(d.get_info<sycl::info::device::AdaptiveCpp_priority_range>());
  
  // TODO Add tests for more queries
}
BOOST_AUTO_TEST_CASE(kernel_specific_queries) {
  
  sycl::queue q;

  sycl::program p{q.get_context()};
  sycl::kernel k = p.get_kernel<void>();

  auto wg_size =
      k.get_info<sycl::info::kernel_device_specific::work_group_size>(
          q.get_device());
  BOOST_TEST(wg_size > 0);

  auto max_sgs =
      k.get_info<sycl::info::kernel_device_specific::max_num_sub_groups>(
          q.get_device());
  BOOST_TEST(max_sgs > 0);

  auto max_sg_size =
      k.get_info<sycl::info::kernel_device_specific::max_sub_group_size>(
          q.get_device());
  BOOST_TEST(max_sg_size > 0);
  
  // TODO Add tests for more queries
}

namespace atomic_capabilities_query
{
  template <class T>
  bool has(const std::vector<T>& values, T candidate)
  {
    return std::find(values.begin(), values.end(), candidate) != values.end();
  }

  template<class T>
  bool unique(const std::vector<T>& values)
  {
    for (size_t i = 0; i < values.size(); i++) 
    {
      if (std::find(values.begin() + i +1, values.end(), values[i]) != values.end())
      {
        return false;
      }
    }
    return true;
  }
}

BOOST_AUTO_TEST_CASE(atomic_device_order_and_scope_capabilities)
{
  const sycl::device dev = sycl::queue{}.get_device();
  const auto orders = dev.get_info<sycl::info::device::atomic_memory_order_capabilities>();
  const auto scopes = dev.get_info<sycl::info::device::atomic_memory_scope_capabilities>();
  
  using atomic_capabilities_query::has;
  using atomic_capabilities_query::unique;

  BOOST_TEST(unique(orders));
  BOOST_TEST(unique(scopes));
  
  BOOST_TEST(has(orders, sycl::memory_order::relaxed));
  BOOST_TEST(has(scopes, sycl::memory_scope::sub_group));
  BOOST_TEST(has(scopes, sycl::memory_scope::work_group));

  const bool has_acquire = has(orders, sycl::memory_order::acquire);
  const bool has_release = has(orders, sycl::memory_order::release);
  const bool has_acq_rel = has(orders, sycl::memory_order::acq_rel);

  BOOST_TEST(has_acquire == has_release);
  BOOST_TEST(has_acquire == has_acq_rel);

  if (has(orders, sycl::memory_order::seq_cst)) 
  {
    BOOST_TEST(has_acquire);
    BOOST_TEST(has_release);
    BOOST_TEST(has_acq_rel);
  }
  if (has(scopes, sycl::memory_scope::system)) 
  {
    BOOST_TEST(has(scopes, sycl::memory_scope::device));
  }
  if (has(scopes, sycl::memory_scope::device)) 
  {
    BOOST_TEST(has(scopes, sycl::memory_scope::work_group));
  }
}

BOOST_AUTO_TEST_SUITE_END()
