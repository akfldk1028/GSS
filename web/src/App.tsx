import { useCallback, useState } from 'react'
import { Viewer } from './components/Viewer'
import { ModelSelector } from './components/ModelSelector'

export default function App() {
  const [modelId, setModelId] = useState<string | null>(null)

  const handleSelect = useCallback((id: string) => {
    setModelId(id)
  }, [])

  return (
    <div style={{
      width: '100vw',
      height: '100dvh',
      background: '#111',
      overflow: 'hidden',
      position: 'relative',
    }}>
      {modelId ? (
        <>
          <Viewer modelId={modelId} />
          <button
            onClick={() => setModelId(null)}
            style={{
              position: 'absolute', top: 12, right: 12, zIndex: 20,
              background: 'rgba(0,0,0,0.7)', color: '#fff',
              border: '1px solid #555', borderRadius: 6,
              padding: '6px 14px', cursor: 'pointer', fontSize: 13,
            }}
          >
            Back
          </button>
        </>
      ) : (
        <ModelSelector onSelect={handleSelect} />
      )}
    </div>
  )
}
