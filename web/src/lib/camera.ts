/**
 * Orbit camera controller.
 *
 * Produces a 4x4 camera-to-world matrix (OpenGL convention: Y-up, -Z forward)
 * from spherical coordinates (theta, phi, radius) + target point.
 */

export interface CameraState {
  theta: number    // azimuth (radians)
  phi: number      // elevation (radians), clamped to avoid gimbal lock
  radius: number   // distance from target
  target: [number, number, number]
  fov: number      // vertical FOV in degrees
}

export const DEFAULT_CAMERA: CameraState = {
  theta: 0,
  phi: 0.3,
  radius: 3.0,
  target: [0, 0, 0],
  fov: 60,
}

/** Clamp phi to avoid singularity at poles. */
const clampPhi = (phi: number) => Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, phi))

/** Apply orbit (drag) delta. */
export function orbit(cam: CameraState, dTheta: number, dPhi: number): CameraState {
  return {
    ...cam,
    theta: cam.theta + dTheta,
    phi: clampPhi(cam.phi + dPhi),
  }
}

/** Apply zoom (scroll / pinch) delta. */
export function zoom(cam: CameraState, delta: number): CameraState {
  return {
    ...cam,
    radius: Math.max(0.1, cam.radius * (1 + delta)),
  }
}

/** Apply pan delta in screen-space. */
export function pan(cam: CameraState, dx: number, dy: number): CameraState {
  const { theta } = cam
  // Right vector in world space
  const rx = -Math.sin(theta)
  const rz = Math.cos(theta)
  // Up is world Y
  const scale = cam.radius * 0.002
  return {
    ...cam,
    target: [
      cam.target[0] + rx * dx * scale,
      cam.target[1] + dy * scale,
      cam.target[2] + rz * dx * scale,
    ],
  }
}

/** Build 4x4 camera-to-world matrix (row-major, 16 floats) from camera state. */
export function getCameraToWorld(cam: CameraState): number[] {
  const { theta, phi, radius, target } = cam

  // Eye position in spherical coords
  const cosPhi = Math.cos(phi)
  const eyeX = target[0] + radius * cosPhi * Math.cos(theta)
  const eyeY = target[1] + radius * Math.sin(phi)
  const eyeZ = target[2] + radius * cosPhi * Math.sin(theta)

  // Look-at: forward = normalize(target - eye)
  let fx = target[0] - eyeX
  let fy = target[1] - eyeY
  let fz = target[2] - eyeZ
  const fLen = Math.sqrt(fx * fx + fy * fy + fz * fz)
  fx /= fLen; fy /= fLen; fz /= fLen

  // World up
  const ux = 0, uy = 1, uz = 0

  // Right = forward x up
  let rx = fy * uz - fz * uy
  let ry = fz * ux - fx * uz
  let rz = fx * uy - fy * ux
  const rLen = Math.sqrt(rx * rx + ry * ry + rz * rz)
  rx /= rLen; ry /= rLen; rz /= rLen

  // True up = right x forward
  const tux = ry * fz - rz * fy
  const tuy = rz * fx - rx * fz
  const tuz = rx * fy - ry * fx

  // OpenGL convention: axes = [right, up, -forward], translation = eye
  // Laid out as row-major 4x4 (row 0 = [r.x, u.x, -f.x, eye.x]):
  return [
    rx,   tux,  -fx,  eyeX,
    ry,   tuy,  -fy,  eyeY,
    rz,   tuz,  -fz,  eyeZ,
    0,    0,    0,    1,
  ]
}

/** Compute intrinsics from FOV and canvas size. */
export function getIntrinsics(cam: CameraState, width: number, height: number) {
  const fy = height / (2 * Math.tan((cam.fov * Math.PI / 180) / 2))
  const fx = fy  // square pixels
  return { fx, fy, cx: width / 2, cy: height / 2 }
}
