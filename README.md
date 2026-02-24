# ✦ Athena — Self-Hosted AI Discord Bot

> **完全ローカル動作** のAIアシスタントBot。外部API依存ゼロ、デュアルGPU活用、プライバシー完全保護。

## ✨ 主要機能

Athena はマルチモーダルなAIアシスタントとして、以下の機能を提供します。

### 🗨️ AIチャット & アシスタント
- **/chat [message]**: 文脈を記憶するAIチャット。Gemma 3 12B による高度な推論が可能。
- **/code [prompt]**: コードの生成、レビュー、バグ修正の提案。
- **/translate [text] [to]**: 高精度な多言語翻訳。
- **/summarize [text]**: 長文の要約。

### 🎨 画像生成 (FLUX.1)
- **/imagine [prompt]**: 最新の **FLUX.1-schnell** モデルを使用した高品質な画像生成。
- **🔍 拡大 (2x)**: 生成された画像の解像度を Real-ESRGAN で 2倍にアップスケールします（ボタン操作）。

---

## 🏗️ アーキテクチャ

デュアルGPU構成を最大限に活用し、LLMと画像生成を完全に分離して実行します。

```
┌─────────────────────────────────────────┐
│           Dual GPU Architecture         │
├─────────────────┬───────────────────────┤
│  RTX 5080 (16GB)  │  RTX 3080 (10GB)      │
│  CUDA:0           │  CUDA:1               │
├─────────────────┼───────────────────────┤
│  画像生成          │  LLM推論              │
│  FLUX.1-schnell   │  Gemma 3 12B          │
│  4-bit量子化      │  Ollama               │
│  ~15GB VRAM       │  ~8GB VRAM            │
│  ※VAE/CLIPはCPUへ │                       │
└─────────────────┴───────────────────────┘
```

---

## � コマンドガイド

### `/imagine` (画像生成)
FLUX.1-schnell に最適化されたパラメータを使用します。

- **prompt**: 生成したい画像の説明（英語推奨）。
- **steps** (Default: 4): 拡散ステップ数。Schnellモデルは **4〜8ステップ** で十分な品質が得られます。
- **guidance_scale** (Default: 0.0): Flux.1-schnell は **0.0** が推奨値です。
- **width / height**: 画像サイズ。最大 1536px まで指定可能。
- **seed**: 生成の再現性を確保するための数値。

> 💡 **Tips**: 生成後に表示される「拡大 (2x)」ボタンを押すと、CPUリソースを使用して画像をより鮮明にします。

### `/chat` (AI会話)
- **文脈保持**: 過去の数ターンの会話を記憶し、自然な対話が可能です。
- **履歴クリア**: 会話が噛み合わなくなった場合は `/clear` で履歴をリセットしてください。

### `/stats` (システム情報)
- システムの負荷状況、各GPUのVRAM使用量、モデルのロード状態をリアルタイムで確認できます。

---

## ⚙️ 技術的な最適化 (OOM対策済み)

本プロジェクトでは、16GBのVRAMで巨大な FLUX モデルを安定動作させるために以下の高度な最適化を実装しています。

1. **4-bit 量子化 (`bitsandbytes`)**: Transformer と T5 エンコーダを量子化し、VRAM消費を大幅に削減。
2. **VAE/CLIP CPU オフロード**: デコード時にVRAMが枯渇するのを防ぐため、VAE と CLIP プロセッサを CPU (RAM) 側で実行。
3. **bfloat16 統一**: RTX 5080 (Blackwell) に最適化されたデータ型を使用し、計算エラーとメモリオーバーヘッドを回避。
4. **VAE Tiling & Slicing**: 巨大な画像のデコードをタイル状に分割して処理し、メモリピークを抑制。
5. **GPU 隔離**: Docker Compose で各サービスに特定の `device_ids` を割り当て、リソース競合を完全に防止。

---

## 🚀 セットアップ手順

### 1. 準備
```bash
git clone https://github.com/luy869/AI_only.git athena
cd athena
cp .env.example .env
```

### 2. 環境設定
`.env` に Discord Bot Token を記述します。

### 3. モデル取得 & 起動
```bash
# モデルの事前ロード（初回のみ）
chmod +x scripts/download_models.sh
./scripts/download_models.sh

# 起動
docker compose up --build -d
```

---

## � トラブルシューティング

### Q: 画像生成で "CUDA out of memory" が出る
- `athena-bot` コンテナのメモリ割り当てを確認してください。本構成では画像サイズ `1024x1024` までは安定動作を確認しています。
- システムの物理RAMが 16GB 未満の場合、スワップ設定が必要になる場合があります。

### Q: LLM の回答が遅い
- Gemma 3 12B は非常に強力なモデルですが、RTX 3080 での推論には 10〜30秒ほどかかる場合があります。
- `/stats` で GPU 1 (RTX 3080) が正しく使用されているか確認してください。

---

## 📄 ライセンス

MIT License
