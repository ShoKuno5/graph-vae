# Graph‑VAE – DDP Workflow

本リポジトリでは **Wisteria (PJM)** + **Singularity** 環境でのマルチ GPU 学習 (DDP)、評価、そしてオフライン W\&B ログのクラウド同期までを想定しています。以下のコマンド例を自分のパスに合わせてコピー & ペーストすれば、そのまま動作します。

---

## 📂 ディレクトリ構成 (ホスト)

```
/work/01/jh210022o/q25030/graph-vae/
├ images/                            # Singularity イメージ置き場
│   └ gvae_cuda.sif
├ experiments/
│   └ wisteria/
│       ├ run_gpu_ddp.sh            # DDP で学習
│       └ run_gpu_eval.sh           # 学習済み ckpt 評価
├ runs/                              # ★ すべての生成物がここに入る
│   └ <TIMESTAMP>/
│       ├ graphvae_ddp_amp.pt        # ckpt
│       ├ tb/                        # TensorBoard
│       └ wandb/                     # オフライン W&B
└ README.md (本ファイル)
```

> **TIP**: `runs/` 以下は Git 管理対象外 ( `.gitignore` 済み ) です。

---

## 🚀 学習ジョブ (run\_gpu\_ddp.sh)

```bash
#── pjsub でジョブ投入 ─────────────────────────────
pjsub experiments/wisteria/run_gpu_ddp.sh
#   └ EXP 変数は内部で自動生成 (YYYYMMDD_HHMMSS)
```

* 学習スクリプト `train_graphvae_ddp.py` には `--log_dir /workspace/runs/$EXP` が渡され、
  **repo 直下 `runs/<TIMESTAMP>/`** に ckpt, TensorBoard, W\&B (offline) がまとめて出力されます。
* GPU・ノード・ジョブ時間などは script 内 `#PJM` 行で調整してください。

---

## 🧪 評価ジョブ (run\_gpu\_eval.sh)

```bash
#── 直前の最新学習結果を評価 (EXP 未指定) ───────────
pjsub experiments/wisteria/run_gpu_eval.sh

#── EXP を明示する場合 ───────────────────────────
EXP=20250601_215345 pjsub experiments/wisteria/run_gpu_eval.sh
```

* `EXP` を省略すると `runs/` 内で **最新タイムスタンプ dir** が自動選択されます。
* `eval.txt` が同じディレクトリに保存されます。

---

## 📊 TensorBoard のローカル閲覧

```bash
# SSH トンネルを張る例 (ローカル 6006 → Wisteria 6006)
ssh -L 6006:localhost:6006 q25030@wisteria.gsic.titech.ac.jp

# Wisteria 側で
module load singularity/3.7.3
singularity exec -B runs:/runs images/gvae_cuda.sif \
  tensorboard --logdir /runs --port 6006 --bind_all
```

ブラウザで `http://localhost:6006` を開けば可視化できます。

---

## 🏷️ オフライン W\&B → クラウド同期

### 1. コンテナ内で対話ログイン

```bash
RUNS=/work/01/jh210022o/q25030/graph-vae/runs
IMG=/work/01/jh210022o/q25030/graph-vae/images/gvae_cuda.sif
EXP=20250601_215345   # オフライン run を含む dir

singularity shell -B ${RUNS}:/workspace/runs ${IMG}
# ==== 以下コンテナ内 ====
wandb login                     # API Key を入力 (最初だけ)
wandb sync --entity shokuno-the-university-of-tokyo \
           --project graph-vae \
           /workspace/runs/${EXP}/wandb/offline-run-*
exit
```

### 2. 非対話 (API Key を env で渡す)

```bash
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
singularity exec -B ${RUNS}:/workspace/runs ${IMG} bash -c "\
  wandb login --no-tty --relogin \$WANDB_API_KEY && \
  wandb sync --entity shokuno-the-university-of-tokyo --project graph-vae \
             /workspace/runs/${EXP}/wandb/offline-run-*"
```

同期完了すると URL が表示され、ブラウザからダッシュボードを閲覧できます。

---

## 🔖 早見表 (超短縮)

| やりたいこと         | コマンド                                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **学習開始**       | `pjsub experiments/wisteria/run_gpu_ddp.sh`                                                                               |
| **最新 ckpt 評価** | `pjsub experiments/wisteria/run_gpu_eval.sh`                                                                              |
| **特定 EXP 評価**  | `EXP=<TIMESTAMP> pjsub experiments/wisteria/run_gpu_eval.sh`                                                              |
| **W\&B 同期**    | `wandb sync --entity shokuno-the-university-of-tokyo --project graph-vae /workspace/runs/<TIMESTAMP>/wandb/offline-run-*` |

---

### 参考

* Singularity & PJM の詳細は `experiments/wisteria/*.sh` 内コメントを参照
* `gvae/train/train_graphvae_ddp.py` に学習ロジック、`gvae/eval/eval_stable.py` に評価ロジックが実装されています。

Happy Graph‑VAE hacking! 🚀

###　interactiveなジョブ投入例

### model　 

pjsub --interact -g jh210022a -L rscgrp=interactive-a,jobenv=singularity

module load singularity/3.7.3 cuda/12.0 

ROOT=/work/01/jh210022o/q25030 \
CODE=$ROOT/graph-vae \
IMG=$CODE/images/gvae_cuda.sif \
DATA=$ROOT/datasets \
RUNS=$CODE/runs


singularity exec --nv \
  -B "$CODE":/workspace/graph-vae \
  -B "$DATA":/dataset \
  -B "$RUNS":/workspace/runs \
  "$IMG" \
  python /workspace/graph-vae/graphvae/models/model.py

### train

pjsub --interact -g jh210022a -L rscgrp=interactive-a,jobenv=singularity

module load singularity/3.7.3 cuda/12.0 

ROOT=/work/01/jh210022o/q25030 \
CODE=$ROOT/graph-vae \
IMG=$CODE/images/gvae_cuda.sif \
DATA=$ROOT/datasets \
RUNS=$CODE/runs

### GPU で実行
singularity exec --nv \
  -B "$CODE":/workspace/graph-vae \
  -B "$DATA":/dataset \
  -B "$RUNS":/workspace/runs \
  "$IMG" \
  bash -c 'export PYTHONPATH=/workspace/graph-vae:$PYTHONPATH && \
           torchrun --nproc_per_node=4 --tee 3\
                    -m graphvae.train.train_ddp \
                    --dataset enzymes \
                    --epochs 300 \
                    --feature_type deg'
