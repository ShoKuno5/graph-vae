import argparse, yaml
from gvae.train import train_graphvae

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)          # dict になる
    train_graphvae.train(cfg)            # 新 API で呼ぶ

if __name__ == "__main__":
    main()
