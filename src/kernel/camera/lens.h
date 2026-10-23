/* SPDX-FileCopyrightText: 2026 Natron 3D fork
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Real lens camera: each camera ray is traced from the sensor back through a lens prescription
 * (spherical and flat surfaces, clear apertures, a bladed iris) and leaves the front element into
 * the scene. Depth of field, the cat's eye at the frame edges, distortion, focus breathing,
 * natural vignetting and (per channel) axial / lateral colour all come from the trace.
 *
 * Lens space: millimetres, z along the axis from the front vertex towards the sensor. Camera space
 * puts the entrance pupil at the camera position, so framing and focus distance match a pinhole
 * camera with the same focal length. The host (Camera::update) lays the lens out, solves the
 * sensor distance for the focus distance and tabulates where on the rear element the rays that get
 * through land (exit pupil bounds per sensor radius), so few samples are wasted. */

#pragma once

#include "kernel/globals.h"

#include "kernel/camera/projection.h"
#include "kernel/util/differential.h"

CCL_NAMESPACE_BEGIN

/* Inside the iris: a disc, or a regular polygon of `blades` sides (vertex at `rotation`). */
ccl_device_inline bool lens_inside_iris(ccl_constant KernelCamera *cam, const float x, const float y)
{
  const float R = cam->lens_stop_radius;
  const float r2 = x * x + y * y;
  if (r2 > R * R) {
    return false;
  }
  const int n = cam->lens_blades;
  if (n < 3) {
    return true;
  }
  const float sector = M_2PI_F / n;
  float a = atan2f(y, x) - cam->lens_blades_rotation;
  a -= sector * floorf(a / sector);
  /* angle from the edge normal at the middle of this sector */
  const float t = a - 0.5f * sector;
  return sqrtf(r2) * cosf(t) <= R * cosf(0.5f * sector);
}

/* Traces a ray from the sensor side out of the front element for colour channel ch (0 R, 1 G,
 * 2 B). o, d in lens space (d.z < 0); on success they hold the exit point and direction. */
ccl_device_inline bool lens_trace_out(ccl_constant KernelCamera *cam,
                                      const int ch,
                                      ccl_private float3 *o,
                                      ccl_private float3 *d)
{
  float3 P = *o;
  float3 D = *d;
  for (int i = cam->lens_num_surfaces - 1; i >= 0; i--) {
    const float4 s = cam->lens_surface[i];
    const float z = s.x, r = s.y, semi = s.z;
    float t;
    float3 N;
    if (r == 0.0f) {
      if (fabsf(D.z) < 1e-12f) {
        return false;
      }
      t = (z - P.z) / D.z;
      N = make_float3(0.0f, 0.0f, 1.0f);
    }
    else {
      const float3 C = make_float3(0.0f, 0.0f, z + r);
      const float3 oc = P - C;
      const float B = dot(oc, D);
      const float Cq = dot(oc, oc) - r * r;
      const float disc = B * B - Cq;
      if (disc < 0.0f) {
        return false;
      }
      const float sq = sqrtf(disc);
      /* the root on the vertex's cap */
      const bool closer = (D.z > 0.0f) != (r < 0.0f);
      t = closer ? -B - sq : -B + sq;
      N = (P + t * D - C) / r;
    }
    if (t < 0.0f) {
      return false;
    }
    P += t * D;
    if (s.w != 0.0f) {
      if (!lens_inside_iris(cam, P.x, P.y)) {
        return false;
      }
      continue;
    }
    if (P.x * P.x + P.y * P.y > semi * semi) {
      return false;
    }
    /* refract into the medium in front of this surface */
    const float4 nf = cam->lens_ior[i];
    const float nFrom = ch == 0 ? nf.x : (ch == 1 ? nf.y : nf.z);
    float nTo = 1.0f;
    if (i > 0) {
      const float4 nt = cam->lens_ior[i - 1];
      nTo = ch == 0 ? nt.x : (ch == 1 ? nt.y : nt.z);
    }
    if (nFrom != nTo) {
      if (dot(N, D) > 0.0f) {
        N = -N;
      }
      const float cosi = -dot(N, D);
      const float eta = nFrom / nTo;
      const float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
      if (k < 0.0f) {
        return false;
      }
      D = normalize(eta * D + (eta * cosi - sqrtf(k)) * N);
    }
  }
  *o = P;
  *d = D;
  return true;
}

/* Exit pupil bounds at sensor radius rs: rear-plane box (radial x0..x1, tangential +-yh) and the
 * light that reaches that radius, linear between the tabulated radii. */
ccl_device_inline float4 lens_pupil_at(ccl_constant KernelCamera *cam, const float rs)
{
  const float f = clamp(rs / max(cam->lens_pupil_rho_max, 1e-6f), 0.0f, 1.0f) *
                  (LENS_PUPIL_BINS - 1);
  const int i = min((int)f, LENS_PUPIL_BINS - 2);
  const float u = f - i;
  return cam->lens_pupil[i] * (1.0f - u) + cam->lens_pupil[i + 1] * u;
}

ccl_device Spectrum camera_sample_lens(KernelGlobals kg,
                                       const float2 raster_xy,
                                       float2 rand_lens,
                                       ccl_private Ray *ray)
{
  ccl_constant KernelCamera *cam = &kernel_data.cam;

  /* The pinhole direction gives the sensor point (the image is inverted). */
  const ProjectionTransform rastertocamera = cam->rastertocamera;
  const float3 Pcamera = transform_perspective(&rastertocamera, make_float3(raster_xy));
  const float sx = -cam->lens_focal_length * Pcamera.x / Pcamera.z;
  const float sy = -cam->lens_focal_length * Pcamera.y / Pcamera.z;
  const float rs = sqrtf(sx * sx + sy * sy);

  /* Lens Colour: one channel per sample (its third of the first random number). */
  int ch = 1;
  if (cam->lens_chromatic) {
    const float c3 = rand_lens.x * 3.0f;
    ch = min((int)c3, 2);
    rand_lens.x = c3 - ch;
  }

  /* A point in the exit pupil's box on the rear element's plane, turned to the sensor point's
   * azimuth. */
  const float4 pb = lens_pupil_at(cam, rs);
  if (pb.w <= 0.0f) {
    return zero_spectrum();
  }
  const float px = pb.x + rand_lens.x * (pb.y - pb.x);
  const float py = (2.0f * rand_lens.y - 1.0f) * pb.z;
  const float c = rs > 1e-9f ? sx / rs : 1.0f, s = rs > 1e-9f ? sy / rs : 0.0f;
  const float zRear = cam->lens_surface[cam->lens_num_surfaces - 1].x;
  float3 o = make_float3(sx, sy, cam->lens_sensor_z);
  const float3 target = make_float3(c * px - s * py, s * px + c * py, zRear);
  float3 d = normalize(target - o);
  const float cosT = -d.z;
  const float dz = cam->lens_sensor_z - zRear;
  const float area = (pb.y - pb.x) * 2.0f * pb.z;

  if (!lens_trace_out(cam, ch, &o, &d)) {
    return zero_spectrum();
  }

  /* Irradiance weight (pbrt's realistic camera: cos^4 times the sampled area over the distance
   * squared), 1 at the centre of the frame; Vignetting < 1 evens out the light across it. */
  float w = area * cosT * cosT * cosT * cosT / (dz * dz) / cam->lens_light_ref;
  if (cam->lens_vignetting < 1.0f) {
    w *= powf(cam->lens_light_ref / pb.w, 1.0f - cam->lens_vignetting);
  }

  /* Lens space -> camera space (scene units, the entrance pupil at the origin, +z forward). */
  const float k = cam->lens_mm_to_scene;
  float3 P = make_float3(o.x * k, o.y * k, (cam->lens_entrance_pupil_z - o.z) * k);
  float3 D = normalize(make_float3(d.x, d.y, -d.z));

  Transform cameratoworld = cam->cameratoworld;
  if (cam->num_motion_steps) {
    transform_motion_array_interpolate(&cameratoworld,
                                       kernel_data_array(camera_motion),
                                       cam->num_motion_steps,
                                       ray->time);
  }
  ray->P = transform_point(&cameratoworld, P);
  ray->D = normalize(transform_direction(&cameratoworld, D));

#ifdef __RAY_DIFFERENTIALS__
  /* As the pinhole camera's: the lens doesn't change the footprint much. */
  const float3 Dcenter = transform_direction(&cameratoworld, Pcamera);
  const float3 Dcenter_normalized = normalize(Dcenter);
  ray->dP = differential_zero_compact();
  differential3 dD;
  dD.dx = normalize(Dcenter + make_float3(cam->dx)) - Dcenter_normalized;
  dD.dy = normalize(Dcenter + make_float3(cam->dy)) - Dcenter_normalized;
  ray->dD = differential_make_compact(dD);
#endif

  ray->P += cam->nearclip * ray->D;
  ray->tmin = 0.0f;
  ray->tmax = cam->cliplength;

  Spectrum T = zero_spectrum();
  if (cam->lens_chromatic) {
    if (ch == 0) {
      T.x = 3.0f * w;
    }
    else if (ch == 1) {
      T.y = 3.0f * w;
    }
    else {
      T.z = 3.0f * w;
    }
  }
  else {
    T = make_spectrum(w);
  }
  return T;
}

CCL_NAMESPACE_END
