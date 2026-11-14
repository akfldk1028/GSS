import { useEffect, useState } from 'react'

interface Model {
  model_id: string
  num_gaussians: number
  gpu_memory_mb: number
  method: string
}

interface ModelSelectorProps {
  onSelect: (modelId: string) => void
}

export function ModelSelector({ onSelect }: ModelSelectorProps) {
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/render/models')
      .then(r => r.json())
      .then(data => {
        setModels(data.models || [])
        setLoading(false)
        if (data.error) setError(data.error)
        // Auto-select first model
        if (data.models?.length > 0) {
          onSelect(data.models[0].model_id)
        }
      })
      .catch(e => {
        setError(e.message)
        setLoading(false)
      })
  }, [onSelect])

  if (loading) {
    return (
      <div style={centerStyle}>
        <div style={{ fontSize: 18, color: '#aaa' }}>Loading models...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={centerStyle}>
        <div style={{ fontSize: 18, color: '#f44' }}>Error: {error}</div>
        <div style={{ fontSize: 14, color: '#888', marginTop: 8 }}>
          Make sure the server is running:<br />
          <code>uvicorn server.app:app --host 0.0.0.0 --port 8000</code>
        </div>
      </div>
    )
  }

  if (models.length === 0) {
    return (
      <div style={centerStyle}>
        <div style={{ fontSize: 18, color: '#aaa' }}>No models loaded</div>
        <div style={{ fontSize: 14, color: '#888', marginTop: 8 }}>
          Load a model first via POST /render/load
        </div>
      </div>
    )
  }

  return (
    <div style={centerStyle}>
      <div style={{ fontSize: 20, color: '#fff', marginBottom: 16 }}>Select Model</div>
      {models.map(m => (
        <button
          key={m.model_id}
          onClick={() => onSelect(m.model_id)}
          style={buttonStyle}
        >
          <div style={{ fontWeight: 'bold' }}>{m.model_id}</div>
          <div style={{ fontSize: 12, color: '#aaa' }}>
            {m.num_gaussians.toLocaleString()} Gaussians &middot; {m.gpu_memory_mb.toFixed(0)}MB
          </div>
        </button>
      ))}
    </div>
  )
}

const centerStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  height: '100%',
  fontFamily: 'sans-serif',
}

const buttonStyle: React.CSSProperties = {
  background: '#333',
  color: '#fff',
  border: '1px solid #555',
  borderRadius: 8,
  padding: '12px 24px',
  margin: 4,
  cursor: 'pointer',
  textAlign: 'left',
  width: 280,
}
