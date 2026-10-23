/* SPDX-FileCopyrightText: 2024 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/film/write.h"
#include "util/atomic.h"

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

/**
 * Append a deep sample for the current pixel and return its slot (DEEP_SLOT_NONE when the
 * pixel's samples are full).
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
  if (pixel_index >= num_pixels) {
    return DEEP_SLOT_NONE;
  }

  /* Atomically increment sample count and get our slot index. */
  const uint32_t sample_idx = atomic_fetch_and_add_uint32(&sample_counts[pixel_index], 1);

  /* Check against maximum samples per pixel. */
  if (sample_idx >= kernel_data.film.deep_max_samples) {
    /* Decrement count since we couldn't store the sample. */
    atomic_fetch_and_add_uint32(&sample_counts[pixel_index], -1);
    return DEEP_SLOT_NONE;
  }

  /* Calculate offset into sample data buffer
   * Layout: pixel[0].samples[0..max], pixel[1].samples[0..max], ... */
  const uint64_t offset = (uint64_t)pixel_index * kernel_data.film.deep_max_samples + sample_idx;
  if (offset >= (uint64_t)DEEP_SLOT_NONE) {
    return DEEP_SLOT_NONE;
  }

  /* RGB gathers the path's light as it is traced (film_deep_add_light). */
  deep_samples[offset].r = 0.0f;
  deep_samples[offset].g = 0.0f;
  deep_samples[offset].b = 0.0f;
  deep_samples[offset].a = alpha;
  deep_samples[offset].z = z;
  deep_samples[offset].z_back = z_back;
  deep_samples[offset].stop = stop;
  deep_samples[offset].count = 1.0f;
  return (uint32_t)offset;
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
    deep_samples[slot].stop = 0.0f;
  }
}

/** \} */

#endif /* __DEEP_OUTPUT__ */

CCL_NAMESPACE_END
