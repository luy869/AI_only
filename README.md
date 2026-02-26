# ✦ Athena — Self-Hosted AI Discord Bot

> **完全ローカル動作** のAIアシスタントBot。外部API依存ゼロ、デュアルGPU活用、プライバシー完全保護。

## ✨ 機能一覧

| コマンド | 説明 |
|---|---|
| `/chat` | AI会話（文脈記憶付き） |
| `/clear` | 会話履歴リセット |
| `/code` | コードの生成・解説・レビュー |
| `/translate` | テキスト翻訳（多言語対応） |
| `/summarize` | テキスト要約 |
| `/imagine` | テキストから画像生成 (FLUX.1) |
| `/ping` | レイテンシの計測 |
| `/stats` | システム統計情報の表示 |
| `/help` | コマンドガイドの表示 |
| `/reload` | [Admin] Cogのホットリロード |
| `/shutdown` | [Admin] Botの安全な停止 |

---

## 🏗️ アーキテクチャ

本プロジェクトはデュアルGPU構成を前提としており、リソース消費の激しい画像生成とLLM推論を分離して実行します。

```
┌─────────────────────────────────────────┐
│           Dual GPU Architecture         │
├─────────────────┬───────────────────────┤
│  Primary GPU      │  Secondary GPU        │
│  (VRAM 16GB+)     │  (VRAM 10GB+)         │
├─────────────────┼───────────────────────┤
│  画像生成          │  LLM推論              │
│  FLUX.1-schnell   │  Gemma 3 12B          │
│  4-bit量子化      │  Ollama               │
│  ~15GB VRAM       │  ~8GB VRAM            │
│  ※VAE/CLIPはCPUへ │                       │
└─────────────────┴───────────────────────┘
```

---

## 📖 コマンド詳細ガイド

### 画像生成 (`/imagine`)
FLUX.1-schnell に最適化されたパラメータを使用します。

- **prompt**: 生成したい画像の説明（英語推奨）。
- **steps** (Default: 4): 拡散ステップ数。Schnellモデルは **4〜8ステップ** で十分な品質が得られます。
- **guidance_scale** (Default: 0.0): このモデルは **0.0** が推奨値です。
- **width / height**: 画像サイズ。最大 1536px まで指定可能。
- **seed**: 生成の再現性を確保するための数値。

> 💡 **Tips**: 生成後に表示される「拡大 (2x)」ボタンを押すと、CPUリソースを使用して画像を鮮明にするアップスケール処理（Real-ESRGAN）を実行します。

### AIアシスタント (`/chat`, `/code` 等)
- **文脈保持**: `/chat` は過去の対話履歴を保持します。
- **特定用途**: `/code` (プログラミング支援)、`/translate` (翻訳)、`/summarize` (要約) はそれぞれ特化したシステムプロンプトを使用して推論効率を高めています。

### システム管理
- **/stats**: GPUごとのVRAM使用状況や、モデルのロード完了フラグをリアルタイムで監視。
- **/help**: 使用可能な全コマンドのリストと簡単な解説を表示。

---

## ⚙️ 技術的な最適化 (OOM対策)

16GB程度のVRAMで巨大な FLUX モデルを安定動作させるため、以下の最適化を実装しています。

1. **4-bit 量子化 (`bitsandbytes`)**: Transformer 等を量子化し、メモリ消費を大幅に削減。
2. **VAE/CLIP CPU オフロード**: VRAM 枯渇を防ぐため、一部パーツをシステムRAM側で実行。
3. **bfloat16 統一**: 最新のGPUアーキテクチャに最適化されたデータ型を使用し、メモリオーバーヘッドを回避。
4. **VAE Tiling & Slicing**: 巨大な画像をタイル状に分割してデコードし、メモリピークを抑制。
5. **GPU アイソレーション**: Docker Compose で各サービスを特定のGPUに隔離。

---

## � 前提条件

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

---

## �🚀 セットアップ手順

### 1. リポジトリ準備

```bash
git clone https://github.com/luy869/AI_only.git athena
cd athena
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

---

## 📁 プロジェクト構成

```
.
├── bot/
│   ├── main.py               # エントリーポイント
│   ├── config.py             # 設定管理
│   ├── cogs/                 # コマンド実装 (chat, imagine etc.)
│   ├── services/             # LLM/画像生成 エンジン
│   └── utils/                # ユーティリティ (Embed作成, レート制限)
├── models/                   # MLモデル格納先
├── cache/                    # 生成画像キャッシュ
├── scripts/
│   └── download_models.sh   # モデルDLスクリプト
├── Dockerfile                # マルチステージビルド
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## ⚙️ 設定一覧

すべて `.env` で設定可能。詳細は `.env.example` を参照してください。

| 設定 | デフォルト | 説明 |
|---|---|---|
| `DISCORD_TOKEN` | — | Discord Bot Token（必須） |
| `OWNER_ID` | — | Bot管理者のDiscord ID |
| `SD_MODEL_ID` | `FLUX.1-schnell` | 画像生成モデルID |
| `SD_GPU_DEVICE` | `0` | 画像生成用GPUインデックス |
| `LLM_N_CTX` | `4096` | コンテキストウィンドウ |
| `LLM_TEMPERATURE` | `0.7` | 生成温度 |
| `SD_DEFAULT_STEPS` | `4` | 画像生成ステップ数 |
| `RATE_LIMIT_PER_MINUTE` | `10` | ユーザーあたりのレート制限 |

---

## 🔧 トラブルシューティング

### Q: 画像生成で "CUDA out of memory" が出る
- `athena-bot` コンテナのメモリ割り当て（`docker-compose.yml`）を確認してください。
- 高解像度（1024px超）の生成時にメモリが不足する場合は、解像度を下げて試してください。

### Q: 推論が CPU で動いているように見える
- `nvidia-smi` を実行し、コンテナからGPUが見えているか確認してください。
- 環境変数 `SD_GPU_DEVICE` 等が正しく設定されているか確認してください。

### Q: モデルのロードが遅い
- 初回起動時は HuggingFace からモデルをダウンロードするため時間がかかります。
- 事前に `scripts/download_models.sh` を実行してください。

### Q: Slash Commands が表示されない
- Bot起動後、Discord側への同期に最大1時間かかることがあります。
- Botをサーバーから一度除外し、再招待すると即座に反映される場合があります。

---

## 📝 Docker を使わない場合

```bash
# Python 3.11 の仮想環境を作成
python3.11 -m venv .venv
source .venv/bin/activate

# 依存関係インストール
pip install -r requirements.txt

# llama-cpp-python (CUDA対応) をビルド
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# モデルダウンロード
bash scripts/download_models.sh

# 環境変数の設定後、起動
python -m bot.main
```
