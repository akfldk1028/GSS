import json, math
for scene in ['room', 'train']:
    print(f'=== {scene} ===')
    walls = json.load(open(f'data/runs/{scene}/interim/s06b_plane_regularization/walls.json'))
    spaces_raw = json.load(open(f'data/runs/{scene}/interim/s06b_plane_regularization/spaces.json'))
    scale = spaces_raw.get('coordinate_scale', 1.0)
    print(f'  coordinate_scale: {scale:.4f}')
    print(f'  walls: {len(walls)}')
    for w in walls:
        hr = w.get('height_range', [0, 0])
        cl = w.get('center_line_2d', [[0, 0], [0, 0]])
        dx = cl[1][0] - cl[0][0]
        dz = cl[1][1] - cl[0][1]
        length = math.sqrt(dx * dx + dz * dz)
        height = hr[1] - hr[0]
        print(f'    wall {w["id"]}: raw_len={length:.2f} raw_h={height:.2f} -> {length/scale:.2f}m x {height/scale:.2f}m  synth={w.get("synthetic", False)}')
    spaces = spaces_raw.get('spaces', [])
    for s in spaces:
        fh = s["floor_height"]
        ch = s["ceiling_height"]
        print(f'    space {s["id"]}: raw floor={fh:.2f} ceil={ch:.2f} -> {fh/scale:.2f}~{ch/scale:.2f}m  h={( ch-fh)/scale:.2f}m')
    # Check planes too
    planes = json.load(open(f'data/runs/{scene}/interim/s06_planes/planes.json'))
    wall_planes = [p for p in planes if p['label'] == 'wall']
    floor_planes = [p for p in planes if p['label'] == 'floor']
    ceil_planes = [p for p in planes if p['label'] == 'ceiling']
    print(f'  s06 planes: {len(planes)} total, {len(wall_planes)} wall, {len(floor_planes)} floor, {len(ceil_planes)} ceiling')
    print()