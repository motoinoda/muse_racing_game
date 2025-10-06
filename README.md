# Muse Racing Game 🧠🏎️

脳波（集中度）でスピードをコントロールするレースゲーム

Muse EEGヘッドバンドから取得した脳波データをリアルタイムで解析し、集中度に応じて車のスピードが変化するインタラクティブなレーシングゲームです。

## 特徴

- **リアルタイム脳波解析**: Muse EEGデバイスから脳波データを取得し、周波数帯域（デルタ、シータ、アルファ、ベータ、ガンマ）を解析
- **集中度計算**: ベータ波とアルファ波の比率から集中度を算出
- **インタラクティブなゲームプレイ**: 集中度が高いほど車のスピードが速くなる
- **視覚的フィードバック**: 脳波の状態、集中度、スピードをリアルタイム表示

## 必要な環境

- Python 3.11
- Muse 2
- macOS（Bluetooth LE対応）

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

### レースゲームの起動

```bash
python muse_race_game.py
```

### 脳波ビューアの起動（データ確認用）

```bash
python muse_viewer.py
```

## レーシングゲームの遊び方

1. Museヘッドバンドを装着し、電源を入れる
2. 「Scan for Muse」ボタンをクリックし，デバイススキャンを開始
3. 「Connect」ボタンをクリックしてMuseデバイスに接続
4. Tutorial Mode をクリック後，Start Game をクリックし，脳波計測をテスト
5. 安静時に Contact Quality が OK もしくは Good になるまで 電極の接触を調整
6. 調整が完了したら，Stop をクリック
7. Difficulty を選択し，Start Game をクリック
8. **集中する**ことで車のスピードが上がる
9. キーボードの矢印キーでレーを移動
10. 障害物を避けながら制限時間内で，どこまで走れるのか挑戦してみよう！
11. 「βパワーの左右差でレーン移動」にチェックを入れると脳波からレーン移動が可能（βパワーの左右差を見ているので，歯ぎしりなどの筋電による操作もできます）


## ファイル構成

```
muse_race_game/
├── muse_race_game.py      # メインのレースゲーム
├── muse_viewer.py         # 脳波データビューア
├── requirements.txt       # 依存パッケージリスト
├── archive/              # 過去のバージョン
│   ├── muse_power_viewer.py
│   └── muse_race_game copy.py
└── README.md             # このファイル
```

## 技術仕様

### 脳波解析

- **サンプリングレート**: 256 Hz
- **周波数帯域**:
  - Theta (θ): 4-8 Hz
  - Alpha (α): 8-13 Hz
  - Beta (β): 13-30 Hz
  - Gamma (γ): 30-50 Hz

### 集中度計算

```
集中度 = Betaパワー / （Alphaパワー + Thetaパワー）
```

スピードは集中度に応じて0-100の範囲で変化します。

## 依存ライブラリ

- `bleak`: Bluetooth Low Energy通信
- `PyQt5`: GUIフレームワーク
- `pyqtgraph`: リアルタイムグラフ描画
- `numpy`: 数値計算
- `scipy`: 信号処理
- `qasync`: 非同期処理
- `bitstring`: バイナリデータ処理

## トラブルシューティング

### デバイスが見つからない

- Museヘッドバンドの電源が入っているか確認
- Bluetoothが有効になっているか確認
- 他のアプリでMuseを使用していないか確認

### 接続が不安定

- Museヘッドバンドを正しく装着し、電極が肌に接触しているか確認
- バッテリー残量を確認
- デバイスを再起動してみる

### 脳波データが取得できない

- Museの電極と肌の接触を確認
- 髪の毛が電極を遮っていないか確認
- 導電性ジェルの使用を検討（必要に応じて）

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します！バグ報告や機能提案はIssueでお願いします。

## 参考

このプロジェクトは以下のライブラリを参考にしています：
- [muse-lsl](https://github.com/alexandrebarachant/muse-lsl)
- [Muse Developer Documentation](https://web.archive.org/web/20181105231756/http://developer.choosemuse.com/tools/windows-tools/available-data-muse-direct)
