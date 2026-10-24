/* SPDX-FileCopyrightText: 2024 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/film/write.h"
#include "util/atomic.h"

#ifndef __KERNEL_GPU__
#  include <atomic>
#endif

CCL_NAMESPACE_BEGIN

/* -------------------------------------------------------------------- */
/** \name Deep Sample Writing
 *
 * Functions for writing deep samples during path tracing.
 *
 * Every camera path writes a sample where its camera ray meets a surface or volume (bounce 0,
 * including the surfaces behind transparent ones). The path remembers that sample (its slot,
 * path / shadow_path deep_slot), and every light contribution it adds to the combined pass is
 * also added to the sample's RGB: each deep sample carries the light of the paths that hit it,
 * not the pixel's mixed colour. `stop` counts the paths that ended there (1, set back to 0 when
 * the path carries on through the surface transparently; a volume segment's opacity), and the
 * output driver turns stop / light per depth into deep alpha and colour (DeepOutputDriver).
 * \{ */

#ifdef __DEEP_OUTPUT__

#  define DEEP_SLOT_NONE 0xFFFFFFFFu

/**
 * Deep sample data structure for kernel use.
 * Matches DeepSampleData in session/deep_buffers.h.
 */
struct ccl_align(32) KernelDeepSample {
  float r, g, b, a;
  float z;
  float z_back;
  /* Paths that ended at this sample (see above), and the number of path samples merged in. */
  float stop;
  float count;
};

/* Makes a newly claimed sample's fields visible before its depth, which publishes it to the other
 * paths of the pixel (they skip slots whose depth is still 0). */
#  if defined(__KERNEL_CUDA__) || defined(__KERNEL_HIP__)
#    define deep_publish_fence() __threadfence()
#  elif !defined(__KERNEL_GPU__)
#    define deep_publish_fence() std::atomic_thread_fence(std::memory_order_release)
#  else
#    define deep_publish_fence() ((void)0)
#  endif

/* Same depth for the merge: surfaces within the merge threshold, volume segments that touch. */
ccl_device_inline bool deep_same_depth(const float z,
                                       const float z_back,
                                       const bool volume,
                                       const float sz,
                                       const float sz_back,
                                       const float threshold)
{
  const bool svolume = sz_back > sz + 1e-6f;
  if (volume != svolume) {
    return false;
  }
  return volume ? (z <= sz_back + threshold && z_back >= sz - threshold) :
                  fabsf(z - sz) < threshold;
}

/**
 * Add a camera path's hit to the pixel's deep samples and return the slot it went to
 * (DEEP_SLOT_NONE only when there is no buffer).
 *
 * Paths at the same depth (deep_merge_threshold) share one sample: its light, stops and path count
 * add up as they are traced, so a pixel needs a slot per surface (or volume segment) it sees, not
 * per path sample. A new depth claims a slot; when the pixel's slots are full the path joins the
 * nearest depth instead of being lost. Slots are zeroed before the render (DeepRenderBuffers),
 * every field but the depths is accumulated atomically, and a claimed slot is published by writing
 * its front depth last (> 0: depths start at the near clip).
 *
 * \param alpha: Shader / segment opacity (kept for the alpha-only output).
 * \param stop: Paths ending here: 1 for surfaces, the opacity for volume segments.
 * \param z: Front depth (distance from camera)
 * \param z_back: Back depth (same as z for surfaces, different for volumes)
 */
ccl_device_inline uint32_t film_write_deep_sample_transparent(
    KernelGlobals kg,
    const uint32_t pixel_index,
    ccl_global KernelDeepSample *ccl_restrict deep_samples,
    ccl_global uint32_t *ccl_restrict sample_counts,
    const float alpha,
    const float z,
    const float z_back,
    const float stop = 1.0f)
{
  /* Bounds check: ensure pixel index is within allocated buffer. */
  const uint32_t num_pixels = kernel_data.film.deep_width * kernel_data.film.deep_height;
  const uint32_t max_samples = kernel_data.film.deep_max_samples;
  if (pixel_index >= num_pixels || max_samples == 0 || !(z > 0.0f)) {
    return DEEP_SLOT_NONE;
  }
  const uint64_t base = (uint64_t)pixel_index * max_samples;
  if (base + max_samples > (uint64_t)DEEP_SLOT_NONE) {
    return DEEP_SLOT_NONE;
  }

  const bool volume = z_back > z + 1e-6f;
  const float threshold = kernel_data.film.deep_merge_threshold;
  uint32_t slot = DEEP_SLOT_NONE;

  /* A sample at this depth already? */
  const uint32_t n = min(sample_counts[pixel_index], max_samples);
  for (uint32_t i = 0; i < n; i++) {
    const ccl_global KernelDeepSample *s = &deep_samples[base + i];
    const float sz = s->z;
    if (sz > 0.0f && deep_same_depth(z, z_back, volume, sz, s->z_back, threshold)) {
      slot = (uint32_t)(base + i);
      break;
    }
  }

  if (slot == DEEP_SLOT_NONE) {
    const uint32_t idx = atomic_fetch_and_add_uint32(&sample_counts[pixel_index], 1);
    if (idx < max_samples) {
      /* A new depth (another path of the pixel may claim the same one at the same time: the
       * output driver merges such pairs). */
      slot = (uint32_t)(base + idx);
      deep_samples[slot].z_back = z_back;
      deep_samples[slot].a = alpha;
      deep_publish_fence();
      deep_samples[slot].z = z;
    }
    else {
      /* Full: join the nearest published depth (volumes prefer volumes). */
      atomic_fetch_and_add_uint32(&sample_counts[pixel_index], (uint32_t)-1);
      float best = FLT_MAX;
      for (uint32_t i = 0; i < max_samples; i++) {
        const ccl_global KernelDeepSample *s = &deep_samples[base + i];
        const float sz = s->z;
        if (!(sz > 0.0f)) {
          continue;
        }
        const bool svolume = s->z_back > sz + 1e-6f;
        const float d = fabsf(sz - z) + (svolume != volume ? 1e30f : 0.0f);
        if (d < best) {
          best = d;
          slot = (uint32_t)(base + i);
        }
      }
      if (slot == DEEP_SLOT_NONE) {
        return DEEP_SLOT_NONE;
      }
    }
  }

  /* RGB gathers the path's light as it is traced (film_deep_add_light). */
  atomic_add_and_fetch_float(&deep_samples[slot].stop, stop);
  atomic_add_and_fetch_float(&deep_samples[slot].count, 1.0f);
  return slot;
}

/**
 * Write an opaque deep sample (alpha 1).
 */
ccl_device_inline uint32_t film_write_deep_sample(
    KernelGlobals kg,
    const uint32_t pixel_index,
    ccl_global KernelDeepSample *ccl_restrict deep_samples,
    ccl_global uint32_t *ccl_restrict sample_counts,
    const float z,
    const float z_back)
{
  return film_write_deep_sample_transparent(
      kg, pixel_index, deep_samples, sample_counts, 1.0f, z, z_back);
}

/**
 * Write a volumetric deep sample (the segment's opacity is also its stop).
 *
 * \param z_entry: Depth where ray enters the volume.
 * \param z_exit: Depth where ray exits the volume.
 */
ccl_device_inline uint32_t film_write_deep_sample_volume(
    KernelGlobals kg,
    const uint32_t pixel_index,
    ccl_global KernelDeepSample *ccl_restrict deep_samples,
    ccl_global uint32_t *ccl_restrict sample_counts,
    const float alpha,
    const float z_entry,
    const float z_exit)
{
  /* Volumes use z_back to represent the volume thickness. */
  return film_write_deep_sample_transparent(
      kg, pixel_index, deep_samples, sample_counts, alpha, z_entry, z_exit, alpha);
}

/* Add a light contribution of a path to its deep sample. */
ccl_device_inline void film_deep_add_light(KernelGlobals kg,
                                           const uint32_t slot,
                                           const Spectrum contribution)
{
  if (slot == DEEP_SLOT_NONE || !kernel_data.film.use_deep_output) {
    return;
  }
  ccl_global KernelDeepSample *deep_samples = (ccl_global KernelDeepSample *)
                                                  kernel_data.film.deep_samples_ptr;
  if (!deep_samples) {
    return;
  }
  const float3 rgb = spectrum_to_rgb(contribution);
  atomic_add_and_fetch_float(&deep_samples[slot].r, rgb.x);
  atomic_add_and_fetch_float(&deep_samples[slot].g, rgb.y);
  atomic_add_and_fetch_float(&deep_samples[slot].b, rgb.z);
}

/* The path carried on through the sample's surface (a transparent bounce): it didn't end there. */
ccl_device_inline void film_deep_pass_through(KernelGlobals kg, const uint32_t slot)
{
  if (slot == DEEP_SLOT_NONE || !kernel_data.film.use_deep_output) {
    return;
  }
  ccl_global KernelDeepSample *deep_samples = (ccl_global KernelDeepSample *)
                                                  kernel_data.film.deep_samples_ptr;
  if (deep_samples) {
    /* Take back this path's stop (1: pass-through happens on surfaces); other paths may share the
     * sample. */
    atomic_add_and_fetch_float(&deep_samples[slot].stop, -1.0f);
  }
}

/** \} */

#endif /* __DEEP_OUTPUT__ */

CCL_NAMESPACE_END
