"""Generate presentation images for best results only.
Runs visualize_planes_3d and visualize_ifc for each good scene.
"""
import subprocess
import sys
import shutil
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
RUNS = PROJ / "data" / "runs"
DATA = PROJ / "data"
OUT = PROJ / "GSS-RESEARCH" / "GSS-PT" / "260309" / "best_results"
OUT.mkdir(parents=True, exist_ok=True)

# Best scenes only
BEST_SCENES = ["jh", "drjohnson", "playroom", "synthetic_L"]

def run_for_scene(scene: str):
    """Temporarily symlink data/interim and data/processed to a run, then visualize."""
    run_dir = RUNS / scene
    if not run_dir.exists():
        print(f"SKIP {scene}: no run data")
        return

    # Backup existing interim/processed if they are not symlinks
    interim = DATA / "interim"
    processed = DATA / "processed"
    interim_bak = DATA / "interim_bak"
    processed_bak = DATA / "processed_bak"

    # Save originals
    if interim.exists() and not interim.is_symlink():
        if interim_bak.exists():
            shutil.rmtree(interim_bak)
        interim.rename(interim_bak)
    elif interim.is_symlink():
        interim.unlink()

    if processed.exists() and not processed.is_symlink():
        if processed_bak.exists():
            shutil.rmtree(processed_bak)
        processed.rename(processed_bak)
    elif processed.is_symlink():
        processed.unlink()

    # Create symlinks to run data
    run_interim = run_dir / "interim"
    run_processed = run_dir / "processed"

    if run_interim.exists():
        # On Windows, use junction for directories
        subprocess.run(["cmd", "/c", "mklink", "/J", str(interim), str(run_interim)],
                       capture_output=True)
    if run_processed.exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(processed), str(run_processed)],
                       capture_output=True)

    try:
        # 1. visualize_planes_3d --save
        print(f"\n=== {scene}: planes_3d ===")
        subprocess.run([sys.executable, str(PROJ / "scripts" / "visualize_planes_3d.py"), "--save"],
                       cwd=str(PROJ), timeout=60)
        # Move generated images
        for img in (PROJ / "docs" / "images").glob("planes_*.png"):
            dest = OUT / f"{scene}_{img.name}"
            shutil.copy2(img, dest)
            print(f"  -> {dest.name}")

        # 2. visualize_ifc --save
        print(f"\n=== {scene}: ifc ===")
        ifc_files = list(processed.glob("*.ifc"))
        ifc_file = ifc_files[0] if ifc_files else None
        if ifc_file and ifc_file.exists():
            subprocess.run([sys.executable, str(PROJ / "scripts" / "visualize_ifc.py"),
                           "--ifc", str(ifc_file), "--save"],
                          cwd=str(PROJ), timeout=60)
            for img in (PROJ / "docs" / "images").glob("ifc_*.png"):
                dest = OUT / f"{scene}_{img.name}"
                shutil.copy2(img, dest)
                print(f"  -> {dest.name}")
            for img in (PROJ / "docs" / "images").glob("bim_*.png"):
                dest = OUT / f"{scene}_{img.name}"
                shutil.copy2(img, dest)
                print(f"  -> {dest.name}")
        else:
            print(f"  No IFC found for {scene}")

    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        # Cleanup junctions (rmdir removes junction without deleting target)
        if interim.exists():
            subprocess.run(["cmd", "/c", "rmdir", str(interim)], capture_output=True)
        if processed.exists():
            subprocess.run(["cmd", "/c", "rmdir", str(processed)], capture_output=True)

        # Restore originals
        if interim_bak.exists():
            interim_bak.rename(interim)
        if processed_bak.exists():
            processed_bak.rename(processed)


if __name__ == "__main__":
    # Also run compare_3dgs_vs_bim (already uses only best scenes)
    print("=== compare_3dgs_vs_bim ===")
    subprocess.run([sys.executable, str(PROJ / "scripts" / "compare_3dgs_vs_bim.py")],
                   cwd=str(PROJ), timeout=120)
    for img in (PROJ / "docs" / "images").glob("3dgs_vs_bim*.png"):
        shutil.copy2(img, OUT / img.name)
        print(f"  -> {img.name}")

    # Run per-scene visualizations
    for scene in BEST_SCENES:
        run_for_scene(scene)

    # Also run synthetic_L test
    print("\n=== synthetic_L polygon test ===")
    subprocess.run([sys.executable, str(PROJ / "scripts" / "test_polygon_building.py")],
                   cwd=str(PROJ), timeout=120)
    for img in (PROJ / "docs" / "images").glob("synthetic_L*.png"):
        shutil.copy2(img, OUT / img.name)
        print(f"  -> {img.name}")

    print(f"\n=== Done! Images in {OUT} ===")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}")
