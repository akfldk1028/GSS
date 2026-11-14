# GSS TODO

## 🔴 GS-SLAM 환경 세팅 (WSL2)

### 환경
- WSL: Ubuntu 22.04, user `cgxr`, conda at `/home/cgxr/miniconda3/`
- GPU: RTX 5080 16GB, driver 581.83, nvcc 미설치 (conda pytorch 번들 사용)
- gcc: 11.4
- `/mnt/c/DK/GSS/clone/`에 3개 레포 클론 완료

### Step 0: WSL 네이티브에 레포 복사 (/mnt/c는 I/O 느림)
```bash
mkdir -p ~/slam
cp -r /mnt/c/DK/GSS/clone/MASt3R-SLAM ~/slam/
cp -r /mnt/c/DK/GSS/clone/MonoGS ~/slam/
cp -r /mnt/c/DK/GSS/clone/SLAM3R ~/slam/
```

### Step 1: MASt3R-SLAM (1순위, CVPR 2025)
```bash
# conda env
conda create -n mast3r_slam python=3.11 -y
conda activate mast3r_slam
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 빌드
cd ~/slam/MASt3R-SLAM
git submodule update --init --recursive
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .

# 체크포인트 (~2GB)
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/

# 테스트
bash ./scripts/download_tum.sh
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml --no-viz
# 출력: logs/default/<seq_name>.ply + .txt
```
- ⚠️ RGBFiles 로더는 `*.png`만 glob → jpg 이미지면 변환 필요
- ⚠️ `--no-viz` 필수 (WSL headless)

### Step 2: SLAM3R (3순위지만 빌드 쉬움, CVPR 2025 Highlight)
```bash
conda create -n slam3r python=3.11 cmake=3.14.0 -y
conda activate slam3r
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

cd ~/slam/SLAM3R
pip install -r requirements.txt
# (선택) XFormers 가속
pip install xformers==0.0.28.post2
cd slam3r/pos_embed/curope/ && python setup.py build_ext --inplace && cd ../../../

# 체크포인트 자동 다운로드 (HuggingFace)
# 테스트: Replica 데모 데이터 → Google Drive에서 다운로드 필요
python recon.py --dataset <이미지폴더> --test_name test1 --save_dir results/
# 출력: results/test1/test1_recon.ply
```

### Step 3: MonoGS (2순위, CVPR 2024 Highlight)
```bash
cd ~/slam/MonoGS
conda env create -f environment.yml
conda activate MonoGS
# wandb 비활성화: GSS step.py에서 config 생성 시 자동 처리
# 테스트
python slam.py --config configs/mono/tum/fr3_office.yaml --eval
# 출력: results/.../point_cloud/final/point_cloud.ply
```
- ⚠️ environment.yml의 pytorch/CUDA 버전이 구형 (1.12) → 충돌 시 수동 설치
- ⚠️ config YAML 기반 (CLI에서 이미지 폴더 직접 지정 불가)

### Step 4: GSS 파이프라인 연동
두 가지 방식:
1. **(추천) WSL 안에서 GSS 전체 실행**
   ```bash
   cd /mnt/c/DK/GSS  # 또는 WSL 네이티브에 clone
   pip install -e .
   # configs/steps/s02_slam.yaml 수정
   # repo_path: "/home/cgxr/slam/MASt3R-SLAM"
   # conda_env: "mast3r_slam"
   gss run --config configs/pipeline_slam.yaml
   ```
2. Windows에서 `wsl -- conda run ...` 호출 → step.py subprocess 수정 필요 (비추천)

### 검증 체크리스트
- [ ] MASt3R-SLAM: conda env + 빌드
- [ ] MASt3R-SLAM: TUM 데이터 → PLY 출력
- [ ] MASt3R-SLAM: GSS 프레임(jh/drjohnson) → PLY
- [ ] s06 호환: 법선 + 포인트 수 확인
- [ ] E2E: `pipeline_slam.yaml` 성공
- [ ] 품질 비교: COLMAP 결과 vs SLAM 결과

---

## 🟡 기존 파이프라인 개선

### s06 성능 최적화
- [ ] Voxel downsampling: RANSAC 전 21M → 2M pts (101s → ~30s)
- [ ] Normal 추정 가속: 다운샘플 cloud에서 추정

### s07 IFC Export
- [ ] s06 개선 결과 기반 IFC 재생성
- [ ] IfcLocalPlacement 설정
- [ ] 다층 건물 지원

### 검증
- [ ] gsplat 2DGS vs PlanarGS 품질 비교
- [ ] s03_planargs → s06 → s07 integration test

---

## 🟢 낮은 우선순위

### BIM 품질
- [ ] Opening 검출 (문/창문) — LOD 300
- [ ] Wall thickness 추정 (현재 고정 0.2m)
- [ ] 가구 인식 (IfcFurnishingElement)

### 성능/확장
- [ ] GPU-accelerated RANSAC (cuML)
- [ ] 대형 point cloud 스트리밍
- [ ] 다중 방 분할 / 층 자동 분할

### 문서화
- [ ] 촬영 프로토콜 가이드
