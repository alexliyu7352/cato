# 專用 3D→2D 日式動畫神經渲染器｜R0 工程原型

**狀態：2026-10-17 的 CPU 工程驗證完成；尚未完成高品質 Ghibli-like 教師資料、授權驗證、真實 Blender 資產的端到端整合，以及 GPU 訓練。**

本原型不是即用型「吉卜力濾鏡」，也不能宣稱已接近專業日本動畫效果。它提供 Agent 完全透過 CLI 運行的可重現管線：3D/深度/法線/語義遮罩/線稿 → 兩條小型生成器 → 訓練 → 推論 → 時序檢查 → 實際效能報告。

## A / B 的確切含義

- **A**：固定單一風格、RGB + depth + normal + semantic one-hot + line 共同輸入 11.78M 參數的生成器。含低解析度注意力與簡化的 SPADE 類語義調節；並非直接從 E²GAN 複製訓練好的模型。
- **B**：首先透過傳統 NPR 做簡化的材質調色，再由 6.20M 參數模型進行殘差修飾。**目前只是「預先風格化資產路線」的最小代理版本，尚未實作 StyleMVD/多視角 UV 烘焙和 3D 模型變形。**
- 兩條路線採用同樣來源圖與基本 G-buffer，輸出一套固定風格，不需要文字 Prompt；真正的目標風格必須由日後合規教師配對資料訓練。

## 執行（Ubuntu/Linux）

要求 Python 3.10+、PyTorch、NumPy、Pillow、OpenCV；產生 3D 煙霧測試場景另外使用 VTK 9.x。正式 3D 場景建議使用 Blender 4.5 LTS+，目前機器未安裝 Blender。

```bash
python -m pip install -e '.[smoke]'

# 12 個真 3D 影格及 G-buffer 的煙霧測試資料
python -m r0.scene --out samples/smoke --frames 4
python -m r0.audit --manifest samples/smoke/manifest.jsonl
pytest -q

# 真實訓練迴圈，但此時 target 是「演算法產生的假標籤」
python -m r0.train --route A --steps 40 --patch 96 --allow-smoke --out results/longer_smoke
python -m r0.train --route B --steps 40 --patch 96 --allow-smoke --out results/longer_smoke

# 比較與時序誤差：不可當成藝術品質分數
python -m r0.evaluate --model-a results/longer_smoke/route_A.pt \
  --model-b results/longer_smoke/route_B.pt --allow-smoke --out results/eval_40steps
python -m r0.temporal_eval --model-a results/longer_smoke/route_A.pt \
  --model-b results/longer_smoke/route_B.pt --out results/temporal_40steps.json
python -m r0.benchmark --height 192 --width 320 --out results/benchmark_320x192.json
python -m r0.demo_video
```

沒有 `--allow-smoke` 時，訓練與評估會拒絕把演算法產生的假標籤當成真教師資料。

## 在既有 Blender .blend 上取得真實通道

```bash
blender -b /path/to/real_scene.blend --python scripts/blender_capture.py -- \
  --out /path/to/shot001 --start 1 --end 24 --width 1280 --height 720
```

目前只實作 Blender 多層 EXR 的獨立輸出腳本，**尚未在本環境運行**，且還缺少 Multi-layer EXR → 標準配對資料 Manifest 的正式攝入器。攝入器需要準確解析 cryptomatte 與 motion vector，不能用不具抗鋸齒性的 Object Index 取代。這是下一個工程子節點。

## 高品質教師輸入的授權／驗收闸門

在 `scripts/teacher_rights_template.json` 填入實際模型及**訓練獨立學生模型的權利來源**（包括大模型底座和 LoRA）；在 `teacher_qa_template.jsonl` 使用 Agent 對每張圖進行結構、風格、身份審核並保存證據。輸入圖片和教師輸出檔名以 `scene_frame.png` 對應。

```bash
python -m r0.teacher_ingest \
  --original samples/smoke/manifest.jsonl \
  --targets /path/to/licensed_teacher_outputs \
  --rights /path/to/teacher_rights.json \
  --review /path/to/teacher_qa.jsonl \
  --out datasets/teacher_train/manifest.jsonl
python -m r0.audit --manifest datasets/teacher_train/manifest.jsonl --quality
```

只有合格且取得權利的配對樣本會進入資料集。為防止相鄰影格洩漏，這個匯入工具會按場景群組分配 train、val、test；但為了徹底防止**跨場景共用資產**的洩漏，正式版還要加入資產組與角色組的 holdout。

## 真正下一步

1. 配置至少一台可接受的 NVIDIA GPU 工作機（在本輪已檢查的 CPU 環境上不適合進行大規模教師生成或模型訓練）。
2. 指定或取得具合法使用權、可批量輸出的 3D 資產：場景、角色、鏡頭片段；推薦先考察 Kenney Nature Kit（CC0）和其他有清晰權利的現成模型。
3. 驗證至少兩套 Ghibli-like 高品質教師的相同輸入 img2img 對照，先查明其輸出可否訓練學生，不下載來源不明的模型。
4. 完成 Blender EXR 真實 G-buffer 轉換、真正的多視角資產預風格化路線 B；之後才進入大規模訓練及品質 A/B 測試。

詳細實測記錄：`notes/R0_EXECUTION_REPORT.md`；技術／論文判斷：`notes/RESEARCH_DECISIONS.md`。

## 2026-10-17 補充：Google AI Pro Colab 與研究模式

已加入 `notebooks/R0_Colab_GPU_Validation.ipynb`，支援：GPU preflight、GitHub 私有倉庫（需 Colab Secret）或 ZIP 上傳、驗證測試、GPU A/B 訓練、CUDA FP16 延遲基準及 Google Drive checkpoint 封存。詳細 GPU 條件與執行步驟請見 `notes/R0_COLAB_GPU_RUNBOOK.md`。

**研究期**可用 `python -m r0.teacher_ingest --mode research --original ... --targets ... --teacher-id ... --out ...` 匯入未經正式權利／人工 QA 審核的教師圖片。這些樣本標記為 `research_teacher_unreviewed`，不會通過 `r0.audit --quality` 生產級檢查。正式模式仍需 `--rights` 與 `--review`。

`assets/r0_smoke_fixture.zip` 保存了舊版 12 個 3D 工程 fixture（不是手繪效果）。Git clone 後可解壓到專案根目錄，以取得測試所需的 `samples/smoke/`。大型訓練結果和模型權重沒有納入 Git。
