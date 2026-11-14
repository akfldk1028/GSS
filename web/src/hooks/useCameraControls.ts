import { useCallback, useEffect, useRef, useState } from 'react'
import {
  type CameraState, DEFAULT_CAMERA,
  orbit, zoom, pan,
  getCameraToWorld, getIntrinsics,
} from '../lib/camera'

interface UseCameraControlsOptions {
  canvas: HTMLCanvasElement | null
  onFrame: (c2w: number[], fx: number, fy: number, cx: number, cy: number) => void
  initialCamera?: Partial<CameraState>
}

/**
 * Attaches mouse + touch event listeners to a canvas for orbit/pan/zoom.
 * Calls onFrame whenever the camera changes.
 */
export function useCameraControls({ canvas, onFrame, initialCamera }: UseCameraControlsOptions) {
  const [camera, setCamera] = useState<CameraState>({ ...DEFAULT_CAMERA, ...initialCamera })
  const cameraRef = useRef(camera)
  cameraRef.current = camera

  const isDragging = useRef(false)
  const isPanning = useRef(false)
  const lastPos = useRef({ x: 0, y: 0 })
  const lastPinchDist = useRef(0)
  const rafId = useRef<number>(0)
  const dirty = useRef(false)

  // Send frame on camera change
  const sendFrame = useCallback(() => {
    if (!canvas) return
    const cam = cameraRef.current
    const c2w = getCameraToWorld(cam)
    const { fx, fy, cx, cy } = getIntrinsics(cam, canvas.width, canvas.height)
    onFrame(c2w, fx, fy, cx, cy)
  }, [canvas, onFrame])

  // Animation loop: only send when dirty
  useEffect(() => {
    let running = true
    const loop = () => {
      if (!running) return
      if (dirty.current) {
        dirty.current = false
        sendFrame()
      }
      rafId.current = requestAnimationFrame(loop)
    }
    rafId.current = requestAnimationFrame(loop)
    return () => {
      running = false
      cancelAnimationFrame(rafId.current)
    }
  }, [sendFrame])

  // Mark dirty when camera changes
  useEffect(() => {
    dirty.current = true
  }, [camera])

  // Mouse events
  useEffect(() => {
    if (!canvas) return

    const onMouseDown = (e: MouseEvent) => {
      e.preventDefault()
      isDragging.current = true
      isPanning.current = e.button === 2 || e.shiftKey
      lastPos.current = { x: e.clientX, y: e.clientY }
    }

    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return
      const dx = e.clientX - lastPos.current.x
      const dy = e.clientY - lastPos.current.y
      lastPos.current = { x: e.clientX, y: e.clientY }

      if (isPanning.current) {
        setCamera(c => pan(c, -dx, dy))
      } else {
        setCamera(c => orbit(c, -dx * 0.005, -dy * 0.005))
      }
    }

    const onMouseUp = () => {
      isDragging.current = false
      isPanning.current = false
    }

    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      setCamera(c => zoom(c, e.deltaY * 0.001))
    }

    const onContextMenu = (e: MouseEvent) => e.preventDefault()

    canvas.addEventListener('mousedown', onMouseDown)
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    canvas.addEventListener('wheel', onWheel, { passive: false })
    canvas.addEventListener('contextmenu', onContextMenu)

    return () => {
      canvas.removeEventListener('mousedown', onMouseDown)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
      canvas.removeEventListener('wheel', onWheel)
      canvas.removeEventListener('contextmenu', onContextMenu)
    }
  }, [canvas])

  // Touch events (mobile)
  useEffect(() => {
    if (!canvas) return

    const getTouchDist = (t1: Touch, t2: Touch) => {
      const dx = t1.clientX - t2.clientX
      const dy = t1.clientY - t2.clientY
      return Math.sqrt(dx * dx + dy * dy)
    }

    const onTouchStart = (e: TouchEvent) => {
      e.preventDefault()
      if (e.touches.length === 1) {
        isDragging.current = true
        lastPos.current = { x: e.touches[0].clientX, y: e.touches[0].clientY }
      } else if (e.touches.length === 2) {
        lastPinchDist.current = getTouchDist(e.touches[0], e.touches[1])
        // Two-finger pan center
        lastPos.current = {
          x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
          y: (e.touches[0].clientY + e.touches[1].clientY) / 2,
        }
      }
    }

    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault()
      if (e.touches.length === 1 && isDragging.current) {
        const dx = e.touches[0].clientX - lastPos.current.x
        const dy = e.touches[0].clientY - lastPos.current.y
        lastPos.current = { x: e.touches[0].clientX, y: e.touches[0].clientY }
        setCamera(c => orbit(c, -dx * 0.005, -dy * 0.005))
      } else if (e.touches.length === 2) {
        // Pinch zoom
        const dist = getTouchDist(e.touches[0], e.touches[1])
        const delta = (lastPinchDist.current - dist) * 0.005
        lastPinchDist.current = dist
        setCamera(c => zoom(c, delta))

        // Two-finger pan
        const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2
        const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2
        const dx = cx - lastPos.current.x
        const dy = cy - lastPos.current.y
        lastPos.current = { x: cx, y: cy }
        setCamera(c => pan(c, -dx, dy))
      }
    }

    const onTouchEnd = () => {
      isDragging.current = false
    }

    canvas.addEventListener('touchstart', onTouchStart, { passive: false })
    canvas.addEventListener('touchmove', onTouchMove, { passive: false })
    canvas.addEventListener('touchend', onTouchEnd)

    return () => {
      canvas.removeEventListener('touchstart', onTouchStart)
      canvas.removeEventListener('touchmove', onTouchMove)
      canvas.removeEventListener('touchend', onTouchEnd)
    }
  }, [canvas])

  /** Force send current camera on next rAF (e.g., after WS connects). */
  const requestFrame = useCallback(() => {
    dirty.current = true
  }, [])

  return { camera, setCamera, requestFrame }
}
