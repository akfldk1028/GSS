import { useCallback, useEffect, useRef, useState } from 'react'

export interface ModelInfo {
  model_id: string
  num_gaussians: number
  method: string
}

export interface StreamStats {
  fps: number
  latencyMs: number
  connected: boolean
  error: string | null
}

/**
 * WebSocket hook that sends camera poses and receives JPEG frames.
 *
 * Protocol matches server/routes/render.py:
 *   → {"type":"camera","camera_to_world":[16],"fx","fy","cx","cy"}
 *   ← binary JPEG
 */
export function useRenderStream(modelId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [stats, setStats] = useState<StreamStats>({
    fps: 0, latencyMs: 0, connected: false, error: null,
  })

  // FPS tracking
  const frameTimesRef = useRef<number[]>([])
  const lastSendRef = useRef<number>(0)
  // Flow control: only allow next send after receiving previous frame
  const pendingRef = useRef(false)

  // Connect
  useEffect(() => {
    if (!modelId) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/render/${modelId}`
    const ws = new WebSocket(wsUrl)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => {
      setStats(s => ({ ...s, connected: true, error: null }))
    }

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        // JSON message
        const msg = JSON.parse(event.data)
        if (msg.type === 'connected') {
          setModelInfo({
            model_id: msg.model_id,
            num_gaussians: msg.num_gaussians,
            method: msg.method,
          })
        } else if (msg.type === 'error') {
          setStats(s => ({ ...s, error: msg.error }))
        } else if (msg.type === 'resized') {
          // acknowledged
        } else if (msg.type === 'pong') {
          // keepalive
        }
      } else {
        // Binary JPEG frame → draw to canvas
        const now = performance.now()
        const latency = now - lastSendRef.current

        // Track frame times for FPS
        frameTimesRef.current.push(now)
        const cutoff = now - 1000
        frameTimesRef.current = frameTimesRef.current.filter(t => t > cutoff)
        const fps = frameTimesRef.current.length

        pendingRef.current = false  // allow next camera send
        setStats(s => ({ ...s, fps, latencyMs: Math.round(latency), error: null }))

        // Draw JPEG to canvas via createImageBitmap (no URL/Image GC overhead)
        const blob = new Blob([event.data], { type: 'image/jpeg' })
        createImageBitmap(blob).then(bitmap => {
          const canvas = canvasRef.current
          if (canvas) {
            const ctx = canvas.getContext('2d')
            if (ctx) {
              ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
            }
          }
          bitmap.close()
        })
      }
    }

    ws.onclose = () => {
      setStats(s => ({ ...s, connected: false }))
      setModelInfo(null)
    }

    ws.onerror = () => {
      setStats(s => ({ ...s, error: 'WebSocket connection failed' }))
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [modelId])

  // Send camera pose (with flow control — skip if previous frame not received)
  const sendCamera = useCallback((
    cameraToWorld: number[],
    fx: number, fy: number, cx: number, cy: number,
  ) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    if (pendingRef.current) return  // still waiting for previous frame

    pendingRef.current = true
    lastSendRef.current = performance.now()
    ws.send(JSON.stringify({
      type: 'camera',
      camera_to_world: cameraToWorld,
      fx, fy, cx, cy,
    }))
  }, [])

  // Send resize
  const sendResize = useCallback((width: number, height: number) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    ws.send(JSON.stringify({ type: 'resize', width, height }))
  }, [])

  return { canvasRef, modelInfo, stats, sendCamera, sendResize }
}
