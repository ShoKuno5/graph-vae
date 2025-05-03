# gvae/train/cli_train.py   ← 今の15行版を書き換え
import argparse, yaml
from gvae.train import train_graphvae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ★ ここを修正 ★
    train_graphvae.train(cfg)     # train() を呼び出す
                                  # → ファイル内で定義されている関数名に合わせる

if __name__ == "__main__":
    main()
