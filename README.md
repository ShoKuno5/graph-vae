# モデル評価

本プロジェクトで学習した DDP チェックポイント（graphvae_ddp_amp.pt）を評価する方法をまとめています。

## 1. ローカル環境での実行例

```bash
python -m gvae.eval.eval \
  --model graphvae \
  --ckpt runs/graphvae_ddp_amp.pt \
  --sample 300        # サンプル数は任意で調整
```

## 2. Singularity 環境での実行例

```bash
# プロジェクトルートに移動
cd /work/jh210022o/q25030/graph-vae
export CODE=$(pwd)

singularity exec --nv \
  -B "$CODE":/workspace/graph-vae \
  "$IMG" bash -c '
    python -m gvae.eval.eval \
      --model graphvae \
      --ckpt /workspace/graph-vae/runs/graphvae_ddp_amp.pt \
      --sample 300
'
```

## 3. アーカイブ作成

成果物をまとめたい場合は以下を実行してください。

```bash
zip -r archive.zip . -x '*.sif' '*.pt'
```

## Running Experiments

以下のスクリプトで学習と評価を実行できます。`EXP` 変数に実験名を指定
すると `runs/EXP` 以下に結果が保存されます。

```bash
# 学習
EXP=my_exp experiments/wisteria/run_gpu_ddp.sh

# 評価 (上記と同じ EXP を指定)
EXP=my_exp experiments/wisteria/run_gpu_eval.sh
```
