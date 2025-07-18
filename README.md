# 建築物LOD2自動モデリングツールのHEAT

建築物のLiDARデータと航空写真から、機械学習を用いて屋根の輪郭線を自動検出し、LOD2（Level of Detail 2）3Dモデリングに必要な屋根エッジ情報を抽出するツールです。

## 概要

このツールは、以下の機能を提供します：

- LASファイルとオルソ画像（航空写真）、OBJファイル（3Dメッシュモデル）との統合処理
- HEATライブラリを使用した深層学習による屋根エッジ検出
- 検出された屋根の角点座標と関連情報の出力

## 必要な環境

- Python 3.13以上
- CUDA対応GPU（推奨、CPUでも動作可能予定）

## インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd building-lod2-builder-heat

# 依存関係をインストール
poetry install
```

## 使用方法

### コマンドライン実行

```bash
poetry run detect-roof-edges [OPTIONS] CHECKPOINT_FILE DSM_DIR OUTPUT_DIR
```

#### 引数

- `CHECKPOINT_FILE`: 学習済みモデルファイルのパス（.pthファイル）
- `DSM_DIR`: LASファイルが格納されているディレクトリ
- `OUTPUT_DIR`: 検出結果を出力するディレクトリ

#### オプション

- `--ortho-dir PATH`: オルソ画像ファイルが格納されているディレクトリ（オプション）
- `--obj-dir PATH`: OBJファイルが格納されているディレクトリ（オプション）
- `--intermediate-dir PATH`: 中間生成物を保存するディレクトリ（オプション）
- `--gpu/--cpu`: GPU使用の有無（デフォルト: GPU使用）
- `--skip-exist/--overwrite`: 既に結果が存在する場合はスキップするかどうか（デフォルト: スキップ）

### 実行例

```bash
# 基本的な実行（DSMのみ）
poetry run detect-roof-edges roof_edge_detection_parameter.pth data/dsm data/output

# オルソ画像を使用した実行
poetry run detect-roof-edges \
  --ortho-dir data/ortho \
  roof_edge_detection_parameter.pth \
  data/dsm \
  data/output

# OBJファイルを使用した実行（外形線情報を利用）
poetry run detect-roof-edges \
  --obj-dir data/obj \
  roof_edge_detection_parameter.pth \
  data/dsm \
  data/output

# 全ての入力データと中間ファイル保存を含む実行
poetry run detect-roof-edges \
  --ortho-dir data/ortho \
  --obj-dir data/obj \
  --intermediate-dir data/intermediate \
  roof_edge_detection_parameter.pth \
  data/dsm \
  data/output

# 既存結果を上書きする実行
poetry run detect-roof-edges \
  --overwrite \
  roof_edge_detection_parameter.pth \
  data/dsm \
  data/output
```

## データ形式

### 入力データ

#### LASファイル
- 形式: `.las`
- 内容: DSM点群データ
- 命名規則: `<任意のID>.las`

#### オルソ画像（オプション）
- 形式: `.tif`
- 内容: 航空写真のオルソ画像
- 命名規則: `<LASファイルと同一のID>.tif`

#### OBJファイル（オプション）
- 形式: `.obj`
- 内容: 3Dメッシュモデル（建物の外形線抽出に使用）
- 命名規則: `<LASファイルと同一のID>.obj`
- 付随ファイル: `<同一のID>.json`（座標系情報を含む）

### 出力データ

#### 屋根エッジ検出結果
- 形式: `.json`
- 内容: 検出された屋根の角点座標と関連情報
- 構造:
```json
{
   "roof_corners": [
      [39, 44],
      [71, 59],
      [19, 65]
   ],
   "roof_edges": [
      [0, 1],
      [1, 2],
      [2, 3]
   ],
   "roof_canvas_size": 256,
   "roof_image_bounds": [64, 32, 192, 224],
   "roof_geo_bounds": [139.123, 35.456, 139.789, 35.987],
   "roof_crs": "EPSG:6677"
}
```

### 中間生成物（オプション）

処理過程で以下のファイルが生成されます：

- `ortho.png`: オルソ画像の入力データ（オルソ画像使用時）
- `dsm_rgb.png`: DSMのRGB可視化画像
- `dsm_depth.png`: DSMの深度画像
- `padded_rgb.png`: パディング処理されたRGB画像
- `padded_depth.png`: パディング処理された深度画像
- `result.png`: 検出された屋根エッジと角点の可視化画像

## 技術仕様

### 主要な依存関係

- **laspy**: LASファイルの読み込み・処理
- **numpy**: 数値計算
- **opencv-python**: 画像処理
- **pyproj**: 座標変換
- **rasterio**: ラスターデータ処理
- **shapely**: 幾何学的図形処理
- **typer**: コマンドラインインターフェース
- **heat**: 屋根エッジ検出用深層学習ライブラリ

### 処理フロー

1. **データ読み込み**: LASファイルからポイントクラウドデータを読み込み
2. **DSM生成**: ポイントクラウドからDSM（Digital Surface Model）を生成
3. **画像前処理**: DSMデータを画像形式に変換し、必要に応じてパディング処理
4. **エッジ検出**: HEATライブラリを使用して屋根エッジを検出
5. **後処理**: 検出結果から角点座標を抽出
6. **出力**: JSON形式で結果を保存

## 開発者情報

- **作者**: fukuchidaisuke
- **所属**: realglobe.jp
- **バージョン**: 0.1.0

## ライセンス

このプロジェクトは GNU General Public License v3.0 またはそれ以降のバージョン（GPL-3.0-or-later）の下でライセンスされています。

依存関係である [HEAT](https://github.com/realglobe-Inc/heat) ライブラリが GPL ライセンスを使用しているため、このプロジェクトも GPL ライセンスを採用しています。

詳細については [LICENSE](LICENSE) ファイルを参照してください。

## トラブルシューティング

### よくある問題

1. **座標系の問題**
   - 入力データの座標系が正しく設定されているか確認
   - pyproj の CRS 設定を確認

2. **ファイル形式エラー**
   - LASファイルの形式とバージョンを確認
   - ファイル名の命名規則を確認

### ログとデバッグ

処理中のログは標準出力に表示されます。詳細なデバッグ情報が必要な場合は、中間ディレクトリを指定して中間生成物を確認してください。
