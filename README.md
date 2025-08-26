# 建築物LOD2自動モデリングツール（HEAT ベース）

建築物の屋根輪郭（ルーフライン）を機械学習で抽出し、LOD2（Level of Detail 2）モデリングに必要な屋根エッジ情報を得るための簡易 CLI ツールです。内部で HEAT ライブラリを利用して推論を行います。

本リポジトリは簡素化されたワークフローに基づき、入力は RGB 画像と深度画像（PNG）を前提とします。LAS/OBJ の取り込みはこの縮小版 CLI では扱いません。

## 概要

- Typer 製 CLI「extract-roofline」の run サブコマンドを提供
- HEAT の学習済みチェックポイント（.pth）をロードして屋根の角点と辺を検出
- 出力は parameters.json。オプションで可視化 PNG を保存

## 必要な環境

- Python: 3.13 系（pyproject: ">=3.13,<4"）
- GPU: 任意（GPU 利用可なら既定で GPU を利用）

HEAT/PyTorch に関する注意:
- 依存 heat は Git ブランチ（2025-dev）から取得します（ネットワーク必須）。
- 一部環境で torch/torchvision の解決に失敗する場合があります。その際は Poetry の venv に事前インストールしてください（CPU Wheels 例）:
  ```bash
  poetry run pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
  poetry install
  ```

## インストール（Poetry）

```bash
# リポジトリをクローン
git clone <repository-url>
cd building-lod2-builder-heat

# Python 3.13 の venv を作成して依存を導入
poetry env use 3.13
poetry install
```

## CLI 使用方法

エントリポイント: extract-roofline（Typer アプリ）

```bash
poetry run extract-roofline run CHECKPOINT_FILE DATA_ROOT \
  [--output-dir PATH] [--byproduct-dir PATH] \
  [--prefer-gpu/--force-cpu] [--skip-exist/--overwrite] \
  [--rich-error/--normal-error] [--exit-on-error]
```

- CHECKPOINT_FILE: HEAT 屋根エッジ検出チェックポイント（.pth）
- DATA_ROOT: 入力データのルートディレクトリ。直下の各サブディレクトリを 1 対象として処理します
- --output-dir PATH: 出力ルート（未指定時は DATA_ROOT 配下に出力）
- --byproduct-dir PATH: 可視化 PNG を保存（デバッグ支援）
- --prefer-gpu/--force-cpu: GPU 利用可なら GPU を使用するか（既定: prefer-gpu）
- --skip-exist/--overwrite: 既存結果をスキップ/上書き（既定: スキップ）
- --rich-error/--normal-error: Typer のリッチトレースバック切替（既定: rich）
- --exit-on-error: 最初の対象エラーで停止

### 実行例（CPU 強制）

```bash
poetry run extract-roofline run roof_edge_detection_parameter.pth data/input_root \
  --output-dir out/output_root \
  --byproduct-dir out/byproduct_root \
  --force-cpu
```

## データレイアウト（DATA_ROOT 配下）

各対象サブディレクトリ直下に以下のファイル名で配置してください（src/.../common/file_names.py 参照）。

必須:
- roofline_extraction_input_rgb.png  （RGB 画像）
- roofline_extraction_input_depth.png（深度画像）

出力（対象ごと）:
- parameters.json
- --byproduct-dir 指定時:
  - roofline_extraction_result_rgb.png
  - roofline_extraction_result_depth.png

例:
```
DATA_ROOT/
  simple/
    roofline_extraction_input_rgb.png
    roofline_extraction_input_depth.png
  complex/
    roofline_extraction_input_rgb.png
    roofline_extraction_input_depth.png
```

## 出力フォーマット（parameters.json）

最低限、以下のキーを出力します。
```json
{
  "roofline_corners": [[10, 20], [30, 40]],
  "roofline_edges": [[0, 1]]
}
```

## HEAT モデル/推論のポイント

- HEAT を HEAT(force_cpu=not prefer_gpu) で初期化し、GPU を優先利用
- HEAT.load_checkpoint の戻り値からキャンバスサイズを推定（None の場合は既定 256）
- 入力は BGR 順で推論: model.infer(input_rgb[:, :, [2, 1, 0]])

## テスト

統合テスト（HEAT/チェックポイントが必要）:
```bash
# 事前に heat/torch が解決できる環境を用意してください
poetry run pytest -q -m integration
# 単一テスト
poetry run pytest -q tests/test_main.py::TestRunIntegration::test_all -m integration
```
前提:
- heat がインストール済み（poetry install 経由）
- チェックポイント roof_edge_detection_parameter.pth がプロジェクトルートまたは test_data/ に存在（無ければテストがダウンロードを試行）

## トラブルシューティング / デバッグ

- 可視化 PNG が必要な場合は --byproduct-dir を指定して確認
- CUDA 無し環境では --force-cpu を使用
- Typer のリッチエラーは --normal-error で無効化でき、その際 _TYPER_STANDARD_TRACEBACK=true が設定されます（標準トレースバック）
- 入力取り込みに失敗する場合は、ファイル名が上記と一致しているか確認

## 既知の注意事項

- 本リポジトリの CLI 名称は extract-roofline（run サブコマンド）です。旧名称 detect-roof-edges は使用しません
- 将来的に OBJ/CRS を取り込む場合は、CRS 文字列（例: "EPSG:6677"）を含む JSON サイドカーの採用を検討

## ライセンス

GNU GPL v3.0 以降（GPL-3.0-or-later）。詳細は [LICENSE](LICENSE) を参照してください。
