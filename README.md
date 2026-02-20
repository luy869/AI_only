# ✦ Athena — Self-Hosted AI Discord Bot

> **完全ローカル動作** のAIアシスタントBot。外部API依存ゼロ、デュアルGPU活用、プライバシー完全保護。

## ✨ 機能一覧

| コマンド | 説明 |
|---|---|
| `/chat` | AI会話（文脈記憶付き） |
| `/clear` | 会話履歴リセット |
| `/code` | コードレビュー・生成・解説 |
| `/translate` | テキスト翻訳 |
| `/summarize` | テキスト要約 |
| `/imagine` | テキストから画像生成 |
| `/ping` | レイテンシ計測 |
| `/stats` | システム統計情報 |
| `/help` | コマンドガイド |
| `/reload` | [Owner] Cogホットリロード |
| `/config` | [Owner] 設定表示・変更 |
| `/shutdown` | [Owner] 安全停止 |

## 🏗️ アーキテクチャ

```
┌─────────────────────────────────────────┐
│           Dual GPU Architecture         │
├─────────────────┬───────────────────────┤
│  GTX 1070 (8GB) │  GTX 1660 Ti (6GB)   │
│  CUDA:0         │  CUDA:1              │
├─────────────────┼───────────────────────┤
│  LLM推論        │  画像生成             │
│  Mistral-7B     │  Stable Diffusion    │
│  GGUF Q4_K_M    │  v1.5                │
│  ~4.5GB VRAM    │  ~4GB VRAM           │
└─────────────────┴───────────────────────┘
```

## 📋 前提条件

- **OS**: Ubuntu 20.04+ / 対応Linuxディストリビューション
- **Docker**: 24.0+
- **Docker Compose**: v2.0+
- **NVIDIA Driver**: 525+ (CUDA 11.8対応)
- **NVIDIA Container Toolkit**: インストール済み
- **Discord Bot Token**: [Discord Developer Portal](https://discord.com/developers/applications) で取得

### NVIDIA Container Toolkit インストール

```bash
# Ubuntu / Debian
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 🚀 セットアップ手順

### 1. リポジトリ準備

```bash
cd /path/to/athena
```

### 2. 環境変数設定

```bash
cp .env.example .env
```

`.env` を編集して最低限以下を設定:

```env
DISCORD_TOKEN=your-bot-token-here
OWNER_ID=your-discord-user-id
```

### 3. モデルのダウンロード

```bash
chmod +x scripts/download_models.sh
./scripts/download_models.sh
```

> ⚠️ 約8.5GB のダウンロードが発生します。初回のみ必要です。

### 4. Docker ビルド & 起動

```bash
docker compose up --build -d
```

### 5. ログ確認

```bash
docker compose logs -f athena
```

正常起動時のログ:
```
✦ Athena is online! Athena#1234 (ID: 123456789)
  Guilds: 1
🔄 Loading ML models in background …
  ✓ LLM model loaded
  ✓ Image generation model loaded
  ✓ Queue workers started
✅ All models loaded — Athena is fully operational
```

### 6. Discord Bot の招待

Discord Developer Portal → OAuth2 → URL Generator:
- **Scopes**: `bot`, `applications.commands`
- **Bot Permissions**: `Send Messages`, `Use Slash Commands`, `Attach Files`, `Embed Links`

生成されたURLでサーバーに招待。

## 📁 プロジェクト構成

```
.
├── bot/
│   ├── __init__.py
│   ├── main.py              # エントリーポイント
│   ├── config.py             # 設定管理 (pydantic-settings)
│   ├── cogs/
│   │   ├── chat.py           # 会話コマンド
│   │   ├── imagine.py        # 画像生成コマンド
│   │   ├── utility.py        # ユーティリティコマンド
│   │   └── admin.py          # 管理コマンド
│   ├── services/
│   │   ├── llm_service.py    # LLM推論エンジン
│   │   ├── image_service.py  # 画像生成エンジン
│   │   └── queue_service.py  # タスクキュー管理
│   └── utils/
│       ├── embed_builder.py  # Embed統一デザイン
│       └── rate_limiter.py   # レート制限
├── models/                    # MLモデル格納先
├── cache/                     # 生成画像キャッシュ
├── logs/                      # ログファイル
├── scripts/
│   └── download_models.sh    # モデルDLスクリプト
├── Dockerfile                 # マルチステージビルド
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .gitignore
```

## ⚙️ 設定一覧

すべて `.env` で設定可能。詳細は `.env.example` を参照。

| 設定 | デフォルト | 説明 |
|---|---|---|
| `DISCORD_TOKEN` | — | Discord Bot Token（必須） |
| `OWNER_ID` | — | Bot管理者のDiscord ID |
| `LLM_GPU_DEVICE` | `0` | LLM用GPUインデックス |
| `SD_GPU_DEVICE` | `1` | SD用GPUインデックス |
| `LLM_N_CTX` | `4096` | コンテキストウィンドウ |
| `LLM_TEMPERATURE` | `0.7` | 生成温度 |
| `SD_DEFAULT_STEPS` | `25` | 画像生成ステップ数 |
| `RATE_LIMIT_PER_MINUTE` | `10` | ユーザーあたりのレート制限 |

## 🔧 トラブルシューティング

### CUDA out of memory

```
LLM_N_CTX=2048     # コンテキストウィンドウを縮小
LLM_MAX_TOKENS=512 # 最大トークン数を縮小
```

### モデルのロードが遅い

初回起動時は HuggingFace からモデルをダウンロードするため時間がかかります。
事前に `scripts/download_models.sh` を実行してください。

### GPU が認識されない

```bash
# Docker から GPU が見えるか確認
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Slash Commands が表示されない

Bot起動後、Discord側への同期に最大1時間かかることがあります。
Botをサーバーから一度除外し、再招待すると即座に反映されます。

## 📝 Docker を使わない場合

```bash
# Python 3.11 の仮想環境を作成
python3.11 -m venv .venv
source .venv/bin/activate

# 依存関係インストール
pip install -r requirements.txt

# llama-cpp-python (CUDA対応) をビルド
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python==0.3.4

# モデルダウンロード
bash scripts/download_models.sh

# 環境変数設定
cp .env.example .env
# .env を編集

# 起動
python -m bot.main
```

## 📄 ライセンス

MIT License
