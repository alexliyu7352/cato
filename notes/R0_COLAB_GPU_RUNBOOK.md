# R0 GPU 概念驗證 — Google AI Pro / Google Colab

Updated: 2026-10-17. **此文檔是部署要求與腳本使用說明，尚未取得 Google Colab 上的實測結果。**

## 確認 Google AI Pro 配額

Google 官方在 2026-09-22 宣布 Google AI 方案包含進階 Colab 福利；Google One 說明頁列出 **Google AI Pro：每月 200 Colab Compute Units (CCU)**。同帳戶登入 https://colab.research.google.com/，在 `Settings > Subscription` 檢查已套用的方案與剩餘 CCU。免費試用不適用進階 Colab 福利。

來源：
- https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/
- https://support.google.com/googleone/answer/14534406
- https://research.google.com/colaboratory/intl/zh-TW/faq.html

**CCU 不是 GPU 小時**。GPU 種類、消耗速率和連續執行時間都不保證。Colab 的交互式 Runtime 可能斷線，Notebook 應在階段邊界先存 checkpoint 到 Drive。Google AI Pro 並不等於 Google AI Ultra 背景執行。免費或 Pro 不可當作永久無人值守訓練伺服器。不可嘗試繞過逾時、代理、禁止活動等 Colab 政策。

## 我們需要什麼 GPU？（工程目標，待實測）

| 階段 | 最低試驗配置 | 更理想配置 | 備註 |
|---|---|---|---|
| R0 渲染器/資料/數十步 Smoke | T4 16GB VRAM | L4 24GB | 只檢查 CUDA、模型梯度、影格效能；不驗證 Ghibli 品質 |
| 512px 配對學生（10–30M 參數） | 16GB VRAM + batch=1, patch=128/256 | 24GB GPU、RAM ≥ 16 GB | 視 discriminator/attention/時序訓練峰值內存調整 |
| 720p / 1080p 推論比較 | 16GB 起 | 24GB+ | 需要實測，不承諾即時效果 |
| 本地 FLUX/SDXL 參考教師生成 | SDXL 小量可能 16GB（需最佳化） | 40GB A100 / 48GB 類 | FLUX 權重本身較大；量化和 offload 改變性能，不能直接當學生 GPU 條件 |

系統：Python ≥ 3.10，Linux，CUDA 可用的 PyTorch（Colab 已預裝，仍以 `torch.cuda.is_available()` 確認），CPU RAM 16GB 以上優先。初期快取／Teacher 配對資料建議準備 Drive 20–50GB；超過 20,000 張圖片時需另算。

## Notebook 快速執行

1. 在 Colab 選擇 **Runtime > Change runtime type > GPU**。首次用 T4 型號或能取得的同級卡即可。
2. `File > Upload notebook` 匯入 `notebooks/R0_Colab_GPU_Validation.ipynb`。
3. 尚未建立 GitHub Repo 時使用 Notebook 預設 `SOURCE_MODE='upload'`，上傳本交付包 `Anime_Neural_Renderer_R0_Colab_Bundle_20261017.zip`。
4. 當 GitHub 私有 Repo 建立並推送後，可設 `SOURCE_MODE='github'`。在 Colab Secret 加入只讀範圍的 `GH_PAT`，不要把 token 寫在 notebook 的 code cell 或輸出中。
5. 順序執行 Notebook：GPU 資訊 → 安裝／測試 → 50-step A/B smoke → fp16 GPU benchmark → 保存 checkpoint 和 JSON 到 Google Drive。
6. 如果已有真的教師 target（以來源影格名字命名），設定 Notebook `TEACHER_TARGETS` 與 `TEACHER_MODEL_ID` 後，使用 research mode 產生臨時配對 Manifest；正式品質測試仍需清理/評估。

## 研究資料模式

技術路線驗證允許不輸入授權文件、不預先作人工 QA 的配對資料攝入：

```bash
python -m r0.teacher_ingest --mode research \
  --original samples/smoke/manifest.jsonl \
  --targets /path/to/teacher_pngs \
  --teacher-id experiment_teacher_v1 \
  --out datasets/teacher_research/manifest.jsonl
python -m r0.audit --manifest datasets/teacher_research/manifest.jsonl
```

此模式會將每筆資料標註為 `research_teacher_unreviewed`，**不等於已經通過藝術品質或正式生產審查**。正式模式仍維持原本嚴格的 rights + review 檢查。研究樣本需至少三個獨立場景群組，會依場景切分訓練/驗證/測試，但真正的資產組隔離仍需在擴大資料集時實作。

## GitHub 保存狀態

本地 Git 存在初始 `268bd36` commit 與後續 R0 Colab/GPU 配置 commit。GitHub connector 能讀取使用者 `alexliyu7352` 的已安裝倉庫，能對已存在的 repo 寫入 commit，**但未提供建立 repository 的 action**。已檢查遠端桌面連線的 VPS，也沒有 `gh` 已登入憑證。不能宣稱已推送到 GitHub。

暫定新私人倉庫名稱 `alexliyu7352/anime-neural-renderer`。若需使用者完成最小一步： https://github.com/new 建立同名 Private Repo（空倉庫、不勾選初始化 README/LICENSE/.gitignore），即可直接將本地 Git bundle / 專案包導入並推送；連接器也能在 Repo 存在後繼續提交新版本。

在 GitHub 新建後，具備此工作區的 `git` 指令：

```bash
cd anime_renderer_r0
git remote add origin https://github.com/alexliyu7352/anime-neural-renderer.git
git branch -M main
git push -u origin main
```

**不要**把模型權重、大型訓練資料或 Colab 密鑰推送到 Git。`.gitignore` 已忽略結果目錄、權重；Checkpoint 存 Google Drive，Git 保存程式／配置／報告／小型固定測試素材。

## R0 成功定義

- `pytest` 通過；兩種學生能在實際 GPU 上跑 50 step，並保存 checkpoint。
- 輸出 CUDA 真實推論延遲和峰值 PyTorch CUDA allocation：320×192、640×360；按 GPU 顯存情況選擇性測試 720p。
- 只有引入高品質教師 target，並經跨場景畫質與結構評估，才能討論 Ghibli 質感是否接近目標。
