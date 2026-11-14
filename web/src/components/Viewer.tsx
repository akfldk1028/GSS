import { useCallback, useEffect, useRef, useState } from 'react'
import { useRenderStream } from '../hooks/useRenderStream'
import { useCameraControls } from '../hooks/useCameraControls'

interface ViewerProps {
  modelId: string
}

export function Viewer({ modelId }: ViewerProps) {
  const { canvasRef, modelInfo, stats, sendCamera, sendResize } = useRenderStream(modelId)
  const containerRef = useRef<HTMLDivElement>(null)
  const [resolution, setResolution] = useState({ width: 640, height: 480 })
  const [canvasEl, setCanvasEl] = useState<HTMLCanvasElement | null>(null)
  const resizeTimer = useRef<ReturnType<typeof setTimeout>>(null)

  // Sync canvas ref → state so useCameraControls gets a non-null element
  useEffect(() => {
    setCanvasEl(canvasRef.current)
  }, [canvasRef])

  // Resize canvas to fit container (debounced)
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect
      const w = Math.round(Math.min(width, 1920))
      const h = Math.round(Math.min(height, 1080))
      if (w > 0 && h > 0) {
        setResolution({ width: w, height: h })
        if (resizeTimer.current) clearTimeout(resizeTimer.current)
        resizeTimer.current = setTimeout(() => sendResize(w, h), 150)
      }
    })

    observer.observe(container)
    return () => observer.disconnect()
  }, [sendResize])

  // Update canvas size
  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas) {
      canvas.width = resolution.width
      canvas.height = resolution.height
    }
  }, [resolution, canvasRef])

  // Camera controls
  const onFrame = useCallback((
    c2w: number[], fx: number, fy: number, cx: number, cy: number,
  ) => {
    sendCamera(c2w, fx, fy, cx, cy)
  }, [sendCamera])

  const { requestFrame } = useCameraControls({
    canvas: canvasEl,
    onFrame,
    initialCamera: {
      theta: 0,
      phi: 0.3,
      radius: 3.0,
      target: [0, 0, 0],
      fov: 60,
    },
  })

  // Trigger initial frame when model connects
  useEffect(() => {
    if (modelInfo) {
      requestFrame()
    }
  }, [modelInfo, requestFrame])

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* HUD */}
      <div style={{
        position: 'absolute', top: 12, left: 12, zIndex: 10,
        background: 'rgba(0,0,0,0.7)', color: '#fff',
        padding: '8px 14px', borderRadius: 8, fontSize: 13,
        fontFamily: 'monospace', lineHeight: 1.6,
        pointerEvents: 'none',
      }}>
        <div style={{ color: stats.connected ? '#4f4' : '#f44' }}>
          {stats.connected ? 'LIVE' : 'DISCONNECTED'}
        </div>
        {modelInfo && (
          <div>{modelInfo.num_gaussians.toLocaleString()} Gaussians</div>
        )}
        <div>{stats.fps} FPS &middot; {stats.latencyMs}ms</div>
        <div>{resolution.width}x{resolution.height}</div>
        {stats.error && <div style={{ color: '#f84' }}>{stats.error}</div>}
      </div>

      {/* Controls help */}
      <div style={{
        position: 'absolute', bottom: 12, left: 12, zIndex: 10,
        background: 'rgba(0,0,0,0.5)', color: '#aaa',
        padding: '6px 12px', borderRadius: 8, fontSize: 11,
        fontFamily: 'sans-serif', pointerEvents: 'none',
      }}>
        Drag: orbit &middot; Scroll: zoom &middot; Shift+Drag: pan &middot; Touch: orbit/pinch
      </div>

      {/* Canvas */}
      <div ref={containerRef} style={{ flex: 1, overflow: 'hidden', background: '#111' }}>
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '100%',
            display: 'block',
            touchAction: 'none',  // prevent browser gestures
          }}
        />
      </div>
    </div>
  )
}
