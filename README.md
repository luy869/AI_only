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

## 🚀 セットアップ

### 1. 準備
```bash
git clone https://github.com/luy869/AI_only.git athena
cd athena
cp .env.example .env
```

### 2. モデル取得 & 起動
```bash
# モデルの事前ロード（初回のみ）
chmod +x scripts/download_models.sh
./scripts/download_models.sh

# 起動
docker compose up --build -d
```

---

## 🔧 トラブルシューティング

### Q: 画像生成で "CUDA out of memory" が出る
- `athena-bot` コンテナのメモリ割り当て（`docker-compose.yml`）を確認してください。
- 高解像度（1024px超）の生成時にメモリが不足する場合は、解像度を下げて試してください。

### Q: 推論が CPU で動いているように見える
- `nvidia-smi` を実行し、コンテナからGPUが見えているか確認してください。
- 環境変数 `SD_GPU_DEVICE` 等が正しく設定されているか確認してください。

---

## 📄 ライセンス

MIT License
