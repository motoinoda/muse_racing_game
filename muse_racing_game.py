#!/usr/bin/env python3
"""
Muse Mind Control Racing Game
脳波（集中度）でスピードをコントロールするレースゲーム
"""

import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import sys
import asyncio
import numpy as np
import time
import struct
import bitstring
from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from bleak import BleakScanner, BleakClient
import qasync
from scipy import signal as scipy_signal

# Muse constants
MUSE_SAMPLING_EEG_RATE = 256
MUSE_GATT_ATTR_STREAM_TOGGLE = '273e0001-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_TP9 = '273e0003-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_AF7 = '273e0004-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_AF8 = '273e0005-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_TP10 = '273e0006-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_RIGHTAUX = '273e0007-4c4d-454d-96be-f03bac821358'

class BrainwaveAnalyzer:
    """リアルタイム脳波解析クラス"""

    def __init__(self, window_size=256):
        self.window_size = window_size
        self.eeg_buffer = {
            'TP9': deque(maxlen=window_size),
            'AF7': deque(maxlen=window_size),
            'AF8': deque(maxlen=window_size),
            'TP10': deque(maxlen=window_size)
        }

        # 周波数帯域定義
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 49)
        }

        self.last_powers = {band: 0.0 for band in self.bands.keys()}

    def add_samples(self, channel, samples):
        """サンプルをバッファに追加"""
        for sample in samples:
            self.eeg_buffer[channel].append(sample)

    def compute_band_powers(self):
        """周波数帯域ごとのパワーを計算（左右チャンネルを平均してからパワー計算）"""
        # 全チャンネルのデータが揃っているか確認
        for channel in ['TP9', 'AF7', 'AF8', 'TP10']:
            if len(self.eeg_buffer[channel]) < self.window_size:
                return self.last_powers

        # 左右チャンネルの平均を計算
        # 左側: TP9（左耳後ろ）+ AF7（左前額）の平均
        left_data = (np.array(self.eeg_buffer['TP9']) + np.array(self.eeg_buffer['AF7'])) / 2.0
        # 右側: AF8（右前額）+ TP10（右耳後ろ）の平均
        right_data = (np.array(self.eeg_buffer['AF8']) + np.array(self.eeg_buffer['TP10'])) / 2.0

        # 左チャンネル（平均後）のFFT計算
        left_fft = np.fft.rfft(left_data)
        fft_freq = np.fft.rfftfreq(len(left_data), 1.0 / MUSE_SAMPLING_EEG_RATE)
        left_power = np.abs(left_fft) ** 2

        # 右チャンネル（平均後）のFFT計算
        right_fft = np.fft.rfft(right_data)
        right_power = np.abs(right_fft) ** 2

        # 各周波数帯域のパワーを計算（左右別々）
        powers = {}
        for band_name, (low_freq, high_freq) in self.bands.items():
            idx = np.logical_and(fft_freq >= low_freq, fft_freq <= high_freq)
            powers[f'{band_name}_left'] = np.sum(left_power[idx])
            powers[f'{band_name}_right'] = np.sum(right_power[idx])
            # 平均も保存（集中度計算用）
            powers[band_name] = (powers[f'{band_name}_left'] + powers[f'{band_name}_right']) / 2.0

        self.last_powers = powers
        return powers

    def get_focus_score(self):
        """集中度スコアを計算（0.0-1.0）- 左右平均"""
        powers = self.compute_band_powers()

        # 【旧指標】β波（集中）/ (α波（リラックス）+ θ波（眠気）)
        beta = powers['beta']
        alpha = powers['alpha']
        theta = powers['theta']
        if alpha + theta == 0:
            return 0.0
        focus = beta / (alpha + theta)
        # 0-1の範囲に正規化（経験的な係数）
        focus_normalized = np.clip(focus / 2.0, 0.0, 1.0)
        return focus_normalized

        # # 【新指標】β波 / α波
        # beta = powers['beta']
        # alpha = powers['alpha']

        # if alpha == 0:
        #     return 0.0

        # focus = beta / alpha

        # # 0-1の範囲に正規化（経験的な係数を調整）
        # # すぐに100%になる場合は、この値を大きくする（10.0, 20.0など）
        # focus_normalized = np.clip(focus / 10.0, 0.0, 1.0)

        # return focus_normalized

    def get_focus_scores_lr(self):
        """左右別の集中度スコアを計算（0.0-1.0）"""
        powers = self.compute_band_powers()

        # 【旧指標】β / (α + θ)
        # # 左チャンネルの集中度
        # beta_left = powers.get('beta_left', 0)
        # alpha_left = powers.get('alpha_left', 0)
        # theta_left = powers.get('theta_left', 0)
        # if alpha_left + theta_left == 0:
        #     focus_left = 0.0
        # else:
        #     focus_left = beta_left / (alpha_left + theta_left)
        #     focus_left = np.clip(focus_left / 2.0, 0.0, 1.0)
        # # 右チャンネルの集中度
        # beta_right = powers.get('beta_right', 0)
        # alpha_right = powers.get('alpha_right', 0)
        # theta_right = powers.get('theta_right', 0)
        # if alpha_right + theta_right == 0:
        #     focus_right = 0.0
        # else:
        #     focus_right = beta_right / (alpha_right + theta_right)
        #     focus_right = np.clip(focus_right / 2.0, 0.0, 1.0)
        # return focus_left, focus_right

        # 【新指標】β / α
        # 左チャンネルの集中度
        beta_left = powers.get('beta_left', 0)
        alpha_left = powers.get('alpha_left', 0)

        if alpha_left == 0:
            focus_left = 0.0
        else:
            focus_left = beta_left / alpha_left
            focus_left = np.clip(focus_left / 10.0, 0.0, 1.0)

        # 右チャンネルの集中度
        beta_right = powers.get('beta_right', 0)
        alpha_right = powers.get('alpha_right', 0)

        if alpha_right == 0:
            focus_right = 0.0
        else:
            focus_right = beta_right / alpha_right
            focus_right = np.clip(focus_right / 10.0, 0.0, 1.0)

        return focus_left, focus_right

    def compute_lateral_bias(self):
        """左右チャンネルのβパワー対数比率を計算（チャンネル平均使用）"""
        # 全チャンネルのデータが揃っているか確認
        for channel in ['TP9', 'AF7', 'AF8', 'TP10']:
            if len(self.eeg_buffer[channel]) < self.window_size:
                return 0.0

        # 左右チャンネルの平均を計算
        # 左側: TP9（左耳後ろ）+ AF7（左前額）の平均
        left_data = (np.array(self.eeg_buffer['TP9']) + np.array(self.eeg_buffer['AF7'])) / 2.0
        # 右側: AF8（右前額）+ TP10（右耳後ろ）の平均
        right_data = (np.array(self.eeg_buffer['AF8']) + np.array(self.eeg_buffer['TP10'])) / 2.0

        # 各チャンネル（平均後）のFFT計算
        left_fft = np.fft.rfft(left_data)
        right_fft = np.fft.rfft(right_data)
        fft_freq = np.fft.rfftfreq(len(left_data), 1.0 / MUSE_SAMPLING_EEG_RATE)

        # βバンド（13-30Hz）のパワーを計算
        # beta_idx = np.logical_and(fft_freq >= 30, fft_freq <= 49)
        beta_idx = np.logical_and(fft_freq >= 13, fft_freq <= 30)
        left_beta = np.sum(np.abs(left_fft[beta_idx]) ** 2)
        right_beta = np.sum(np.abs(right_fft[beta_idx]) ** 2)

        # βパワーの対数比率を計算
        # 負の値: 左のβが強い、正の値: 右のβが強い
        if left_beta == 0 or right_beta == 0:
            return 0.0

        # 対数比率: log(右β/左β)
        # 範囲: -∞ 〜 +∞、0 = 等しい
        ratio = right_beta / left_beta
        bias = np.log(ratio)

        # -1.0 〜 +1.0 の範囲にクリップ（しきい値判定用）
        bias = np.clip(bias, -1.0, 1.0)

        return bias

class RaceGame(QtWidgets.QWidget):
    """レースゲーム画面"""

    def __init__(self):
        super().__init__()
        # 3レーンシステム
        self.current_lane = 1  # 0=左、1=中央、2=右
        self.speed = 0.0  # 現在のスピード
        self.distance = 0.0  # 走行距離
        self.obstacles = []  # 障害物リスト（[lane_index, y_pos]の形式）
        self.game_over = False
        self.game_clear = False  # ゲームクリア
        self.score = 0

        # 制限時間関連
        self.time_limit = 30.0  # 30秒
        self.remaining_time = 30.0
        self.start_time = None

        # 難易度設定（障害物の出現頻度）
        self.difficulty_base_prob = 0.015  # デフォルトはNormal
        self.difficulty_level = 'normal'  # 'easy', 'normal', 'hard'

        # 脳波操作用
        self.brain_control_enabled = False
        self.lateral_bias = 0.0
        self.bias_threshold = 0.1  # レーン変更のしきい値
        self.bias_cooldown = 0  # レーン変更のクールダウン

        # 障害物生成管理
        self.last_obstacle_lane = -1  # 最後に障害物を配置したレーン
        self.obstacle_cooldown = 0  # 障害物生成のクールダウン
        self.tutorial_mode = False  # チュートリアルモード

        # 色設定
        self.road_color = QtGui.QColor(80, 80, 80)
        self.car_color = QtGui.QColor(255, 0, 0)
        self.obstacle_color = QtGui.QColor(100, 100, 200)
        self.line_color = QtGui.QColor(255, 255, 255)

        self.setMinimumSize(300, 400)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)  # キーボード入力を受け取る

    def keyPressEvent(self, event):
        """キーボード入力処理"""
        if self.game_over or self.game_clear:
            return

        # 左右の矢印キーでレーンを移動
        if event.key() == QtCore.Qt.Key_Left:
            if self.current_lane > 0:
                self.current_lane -= 1
        elif event.key() == QtCore.Qt.Key_Right:
            if self.current_lane < 2:
                self.current_lane += 1

        self.update()

    def update_game(self, focus_score, lateral_bias=0.0):
        """ゲーム状態を更新"""
        if self.game_over or self.game_clear:
            return

        # チュートリアルモード以外では時間制限を適用
        if not self.tutorial_mode:
            # 開始時刻の記録
            if self.start_time is None:
                self.start_time = time.time()

            # 残り時間の更新
            elapsed = time.time() - self.start_time
            self.remaining_time = self.time_limit - elapsed

            # 時間切れでゲームクリア
            if self.remaining_time <= 0:
                self.remaining_time = 0.0  # 0.0で表示
                self.game_clear = True
                self.update()  # 画面を再描画してクリア画面を表示
                return

        # スピードを集中度に基づいて更新
        target_speed = focus_score * 10.0  # 最大10ピクセル/フレーム
        self.speed = self.speed * 0.9 + target_speed * 0.1  # スムージング

        # 走行距離を更新
        self.distance += self.speed

        # スコア計算
        self.score = int(self.distance + self.speed * 100)

        # 脳波による左右制御
        if self.brain_control_enabled:
            self.lateral_bias = lateral_bias

            # クールダウン減少
            if self.bias_cooldown > 0:
                self.bias_cooldown -= 1

            # しきい値を超えたらレーン変更（クールダウン中でなければ）
            if self.bias_cooldown == 0:
                if lateral_bias < -self.bias_threshold and self.current_lane > 0:
                    # 左に移動
                    self.current_lane -= 1
                    self.bias_cooldown = 20  # 約0.6秒のクールダウン
                elif lateral_bias > self.bias_threshold and self.current_lane < 2:
                    # 右に移動
                    self.current_lane += 1
                    self.bias_cooldown = 20

        # 障害物の生成
        # クールダウン減少
        if self.obstacle_cooldown > 0:
            self.obstacle_cooldown -= 1

        # 現在の画面の高さを取得
        current_height = self.height() if self.height() > 0 else 600

        # 障害物生成確率をスコアに応じて増加
        # 難易度に応じた基本確率、スコアが1000増えるごとに0.005増加
        base_prob = self.difficulty_base_prob
        score_factor = min(0.025, (self.score / 1000) * 0.05)
        obstacle_prob = base_prob + score_factor

        # 障害物生成（チュートリアルモードでも生成するが、当たり判定は無効）
        if self.obstacle_cooldown == 0 and np.random.random() < obstacle_prob:
            # 現在画面上にある障害物のレーンを確認
            # より広い範囲（車2台分程度）のスペースを空ける
            occupied_lanes = set()
            for obs in self.obstacles:
                if -0.1 <= obs[1] <= 0.4:  # 画面上部、車2台分程度のスペース
                    occupied_lanes.add(obs[0])

            # 利用可能なレーンを決定（占有されていないレーン）
            available_lanes = [l for l in [0, 1, 2] if l not in occupied_lanes]

            # 脳波操作モード、Easyモード、またはチュートリアルモードの場合は最大1つの障害物のみ
            if self.brain_control_enabled or self.difficulty_level == 'easy' or self.tutorial_mode:
                if len(available_lanes) > 0:
                    # ランダムに1つのレーンを選択
                    lane = np.random.choice(available_lanes)
                    self.obstacles.append([lane, -0.08])
                    self.obstacle_cooldown = 30
            else:
                # キーボード操作モードの場合は複数配置可能
                # 最低1レーンは必ず空ける（全レーン塞がないようにする）
                if len(available_lanes) == 0:
                    # 全レーン占有されているので、この回は生成しない
                    pass
                elif len(available_lanes) == 3:
                    # 全レーン空いている場合は、最大2レーンに配置
                    num_obstacles = np.random.randint(1, 3)  # 1または2個
                    selected_lanes = np.random.choice(available_lanes, num_obstacles, replace=False)
                    for lane in selected_lanes:
                        self.obstacles.append([lane, -0.08])
                    self.obstacle_cooldown = 30
                else:
                    # 一部のレーンが空いている場合
                    # 必ず1レーンは空けるため、最大で(available_lanes - 1)個まで配置
                    max_new_obstacles = max(1, len(available_lanes) - 1)
                    num_obstacles = np.random.randint(1, max_new_obstacles + 1)
                    selected_lanes = np.random.choice(available_lanes, num_obstacles, replace=False)
                    for lane in selected_lanes:
                        self.obstacles.append([lane, -0.08])
                    self.obstacle_cooldown = 30

        # 障害物を移動と衝突判定
        for obstacle in self.obstacles[:]:
            # Y座標を比率で更新（スピードに応じて移動）
            obstacle[1] += (self.speed * 2) / current_height

            # 画面外に出たら削除（画面下部を超えたら）
            if obstacle[1] > 1.0:
                self.obstacles.remove(obstacle)
                continue

            # 衝突判定：同じレーンにいて、Y座標が近い場合（チュートリアルモードでは無効）
            if not self.tutorial_mode and obstacle[0] == self.current_lane:
                # 車のY座標（画面の80%位置）
                car_y_ratio = 0.8
                obs_y_ratio = obstacle[1]
                distance_ratio = abs(obs_y_ratio - car_y_ratio)

                # デバッグ出力
                print(f"Same lane! Obstacle Y ratio: {obs_y_ratio:.3f}, Car Y ratio: {car_y_ratio:.3f}, Distance ratio: {distance_ratio:.3f}")

                # 衝突判定の閾値（画面の高さに対する比率）
                # 車の高さ60px + 障害物の高さ40px = 100px
                # より厳しい判定にするため、閾値を小さくする
                # 車の中心から±30px程度（合計60px）= 画面高さ600pxなら 0.05
                collision_threshold_ratio = 0.05  # 0.1 → 0.05に変更
                if distance_ratio < collision_threshold_ratio:
                    print("COLLISION!")
                    self.game_over = True

        self.update()

    def paintEvent(self, event):
        """描画処理"""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # ウィジェットの実際のサイズを取得
        width = self.width()
        height = self.height()

        # 背景（道路）
        painter.fillRect(0, 0, width, height, QtGui.QColor(50, 150, 50))

        # 道路の幅を計算（画面の70%）
        road_width = int(width * 0.7)
        road_left = int(width * 0.15)
        painter.fillRect(road_left, 0, road_width, height, self.road_color)

        # レーン区切り線（2本）
        lane1_x = road_left + road_width // 3
        lane2_x = road_left + 2 * road_width // 3
        painter.setPen(QtGui.QPen(QtCore.Qt.white, 2, QtCore.Qt.DashLine))
        painter.drawLine(lane1_x, 0, lane1_x, height)
        painter.drawLine(lane2_x, 0, lane2_x, height)

        # レーン位置を動的に計算
        lane_positions = [
            road_left + road_width // 6,
            road_left + road_width // 2,
            road_left + 5 * road_width // 6
        ]
        car_x = lane_positions[self.current_lane]
        car_y = int(height * 0.8)

        # 車を描画
        painter.setBrush(self.car_color)
        painter.drawRect(int(car_x) - 20, int(car_y) - 30, 40, 60)

        # 障害物を描画
        painter.setBrush(self.obstacle_color)
        for obs in self.obstacles:
            # obs[0]はlane_index（0, 1, 2）
            # obs[1]はy_ratio（0.0〜1.0）
            lane_idx = obs[0]
            obs_x = lane_positions[lane_idx]
            obs_y = obs[1] * height  # 比率から実際のY座標に変換
            painter.drawRect(int(obs_x) - 20, int(obs_y) - 20, 40, 40)

        # チュートリアルモード表示
        if self.tutorial_mode:
            painter.setPen(QtGui.QColor(0, 200, 0))
            painter.setFont(QtGui.QFont('Arial', 20, QtGui.QFont.Bold))
            painter.drawText(int(width * 0.25), 50, 'TUTORIAL MODE')
        else:
            # 残り時間表示（通常モードのみ）
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.setFont(QtGui.QFont('Arial', 28, QtGui.QFont.Bold))
            time_text = f'Time: {self.remaining_time:.1f}s'
            painter.drawText(int(width * 0.35), 40, time_text)

        # ゲームオーバー表示
        if self.game_over:
            painter.setPen(QtCore.Qt.red)
            painter.setFont(QtGui.QFont('Arial', 40, QtGui.QFont.Bold))
            painter.drawText(int(width * 0.15), int(height * 0.5), 'GAME OVER!')

        # ゲームクリア表示
        if self.game_clear:
            # 白い背景を描画
            bg_rect = QtCore.QRect(int(width * 0.1), int(height * 0.3), int(width * 0.8), int(height * 0.3))
            painter.fillRect(bg_rect, QtGui.QColor(255, 255, 255, 230))  # 半透明の白

            # 枠線を描画
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 0), 3))
            painter.drawRect(bg_rect)

            # テキストを描画
            painter.setPen(QtGui.QColor(0, 200, 0))
            painter.setFont(QtGui.QFont('Arial', 40, QtGui.QFont.Bold))
            painter.drawText(int(width * 0.25), int(height * 0.42), 'GAME CLEAR!')
            painter.setFont(QtGui.QFont('Arial', 24))
            painter.drawText(int(width * 0.32), int(height * 0.52), f'Score: {self.score}')

class MuseRaceGame(QtWidgets.QMainWindow):
    """メインアプリケーション"""

    def __init__(self):
        super().__init__()
        self.client = None
        self.device_address = None
        self.is_streaming = False

        # 脳波解析器
        self.analyzer = BrainwaveAnalyzer(window_size=256)

        # muse-lsl互換のデータ処理変数
        self.timestamps = np.full(5, np.nan)
        self.data = np.zeros((5, 12))
        self.last_tm = 0
        self.first_sample = True
        self.sample_index = 0
        self.reg_params = None
        self._P = 1e-4

        # ハンドルとUUIDのマッピング
        self.uuid_to_handle = {
            MUSE_GATT_ATTR_TP9: 32,
            MUSE_GATT_ATTR_AF7: 35,
            MUSE_GATT_ATTR_AF8: 38,
            MUSE_GATT_ATTR_TP10: 41,
            MUSE_GATT_ATTR_RIGHTAUX: 44
        }

        self.handle_to_channel = {
            32: 'TP9',
            35: 'AF7',
            38: 'AF8',
            41: 'TP10',
            44: 'RIGHTAUX'
        }

        # 集中度スコア
        self.focus_score = 0.0
        self.focus_left = 0.0
        self.focus_right = 0.0
        self.lateral_bias = 0.0

        # 接触品質（信号品質）
        self.contact_quality = {
            'TP9': 'Good',
            'AF7': 'Good',
            'AF8': 'Good',
            'TP10': 'Good'
        }

        # UI初期化
        self.init_ui()

        # ゲーム更新タイマー
        self.game_timer = QtCore.QTimer()
        self.game_timer.timeout.connect(self.update_game)

    def init_ui(self):
        """UI初期化"""
        self.setWindowTitle('Muse Mind Control Racing Game')

        # 画面サイズを取得してウィンドウサイズを調整（より控えめに）
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        # タスクバーやメニューバーを考慮して、さらに小さく
        window_width = min(900, int(screen.width() * 0.7))
        window_height = min(550, int(screen.height() * 0.65))
        # 位置も上に配置
        self.setGeometry(50, 30, window_width, window_height)

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout(central_widget)

        # 左側: コントロールパネル（スクロール可能）
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        left_widget = QtWidgets.QWidget()
        left_panel = QtWidgets.QVBoxLayout(left_widget)
        left_scroll.setWidget(left_widget)

        # デバイス接続コントロール
        self.scan_button = QtWidgets.QPushButton('Scan for Muse')
        self.scan_button.clicked.connect(self.scan_devices)
        left_panel.addWidget(self.scan_button)

        self.device_combo = QtWidgets.QComboBox()
        left_panel.addWidget(QtWidgets.QLabel('Device:'))
        left_panel.addWidget(self.device_combo)

        self.connect_button = QtWidgets.QPushButton('Connect')
        self.connect_button.clicked.connect(self.connect_device)
        self.connect_button.setEnabled(False)
        left_panel.addWidget(self.connect_button)

        self.start_button = QtWidgets.QPushButton('Start Game')
        self.start_button.clicked.connect(self.start_game)
        self.start_button.setEnabled(False)
        left_panel.addWidget(self.start_button)

        self.tutorial_button = QtWidgets.QPushButton('Tutorial Mode')
        self.tutorial_button.clicked.connect(self.start_tutorial)
        self.tutorial_button.setEnabled(False)
        self.tutorial_button.setStyleSheet("""
            QPushButton {
                background-color: #4ECDC4;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45B7D1;
            }
        """)
        left_panel.addWidget(self.tutorial_button)

        # 難易度設定
        left_panel.addWidget(QtWidgets.QLabel(''))
        left_panel.addWidget(QtWidgets.QLabel('Difficulty:'))
        difficulty_layout = QtWidgets.QHBoxLayout()

        self.easy_button = QtWidgets.QPushButton('Easy')
        self.easy_button.clicked.connect(lambda: self.set_difficulty('easy'))
        difficulty_layout.addWidget(self.easy_button)

        self.normal_button = QtWidgets.QPushButton('Normal')
        self.normal_button.clicked.connect(lambda: self.set_difficulty('normal'))
        difficulty_layout.addWidget(self.normal_button)

        self.hard_button = QtWidgets.QPushButton('Hard')
        self.hard_button.clicked.connect(lambda: self.set_difficulty('hard'))
        difficulty_layout.addWidget(self.hard_button)

        left_panel.addLayout(difficulty_layout)

        # デフォルトでNormalを選択状態に
        self.current_difficulty = 'normal'
        self.update_difficulty_buttons()

        self.stop_button = QtWidgets.QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_game)
        self.stop_button.setEnabled(False)
        left_panel.addWidget(self.stop_button)

        self.retry_button = QtWidgets.QPushButton('Retry')
        self.retry_button.clicked.connect(self.retry_game)
        self.retry_button.setEnabled(False)
        self.retry_button.setStyleSheet("""
            QPushButton {
                background-color: #FF6B6B;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF5252;
            }
        """)
        left_panel.addWidget(self.retry_button)

        # ステータス
        self.status_label = QtWidgets.QLabel('Status: Ready')
        left_panel.addWidget(self.status_label)

        left_panel.addWidget(QtWidgets.QLabel(''))

        # 集中度メーター
        left_panel.addWidget(QtWidgets.QLabel('Focus Level:'))
        self.focus_bar = QtWidgets.QProgressBar()
        self.focus_bar.setMaximum(100)
        self.focus_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4ECDC4;
            }
        """)
        left_panel.addWidget(self.focus_bar)

        # スピードメーター
        left_panel.addWidget(QtWidgets.QLabel('Speed:'))
        self.speed_label = QtWidgets.QLabel('0.0 km/h')
        self.speed_label.setFont(QtGui.QFont('Arial', 24, QtGui.QFont.Bold))
        left_panel.addWidget(self.speed_label)

        # 距離
        left_panel.addWidget(QtWidgets.QLabel('Distance:'))
        self.distance_label = QtWidgets.QLabel('0 m')
        self.distance_label.setFont(QtGui.QFont('Arial', 18))
        left_panel.addWidget(self.distance_label)

        # スコア
        left_panel.addWidget(QtWidgets.QLabel('Score:'))
        self.score_label = QtWidgets.QLabel('0')
        self.score_label.setFont(QtGui.QFont('Arial', 18))
        left_panel.addWidget(self.score_label)

        left_panel.addWidget(QtWidgets.QLabel(''))

        # 脳波操作モード切替
        left_panel.addWidget(QtWidgets.QLabel(''))
        self.brain_control_checkbox = QtWidgets.QCheckBox('βパワーの左右差でレーン移動')
        self.brain_control_checkbox.stateChanged.connect(self.toggle_brain_control)
        left_panel.addWidget(self.brain_control_checkbox)

        # 左右バイアス表示
        left_panel.addWidget(QtWidgets.QLabel('Left-Right Bias:'))
        self.bias_bar = QtWidgets.QProgressBar()
        self.bias_bar.setMaximum(100)
        self.bias_bar.setMinimum(-100)
        self.bias_bar.setValue(0)
        self.bias_bar.setFormat('%v')
        self.bias_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #FF6B6B;
            }
        """)
        left_panel.addWidget(self.bias_bar)

        self.bias_label = QtWidgets.QLabel('Left: 0% | Right: 0%')
        left_panel.addWidget(self.bias_label)

        left_panel.addWidget(QtWidgets.QLabel(''))

        # 接触品質表示
        left_panel.addWidget(QtWidgets.QLabel('Contact Quality:'))
        self.contact_labels = {}
        channels = ['TP9', 'AF7', 'AF8', 'TP10']
        for channel in channels:
            label = QtWidgets.QLabel(f'{channel}: Good')
            label.setMinimumWidth(100)
            label.setStyleSheet('padding: 5px; background-color: #90EE90; border-radius: 3px; font-weight: bold;')
            self.contact_labels[channel] = label
            left_panel.addWidget(label)

        # 操作説明
        controls_label = QtWidgets.QLabel('【操作方法】\n← → : レーン切替\n      (3レーン)\n集中 : スピードUP\n左右脳 : レーン移動\n      (要ON)')
        controls_label.setStyleSheet('background-color: #f0f0f0; padding: 10px; border-radius: 5px;')
        left_panel.addWidget(controls_label)

        # 脳波説明
        help_label = QtWidgets.QLabel(
            '【脳波について】\n'
            'Theta波(4-8Hz): 眠気\n'
            'Alpha波(8-13Hz): リラックス\n'
            'Beta波(13-30Hz): 集中\n'
            '\n'
            '集中度 = Beta/(Alpha+Theta)'
        )
        help_label.setStyleSheet('background-color: #ECF0F1; padding: 10px; border-radius: 5px; font-size: 10px;')
        left_panel.addWidget(help_label)

        left_panel.addStretch()

        # 右側: ゲーム画面とグラフ
        right_panel = QtWidgets.QHBoxLayout()

        # ゲーム画面
        self.race_game = RaceGame()
        right_panel.addWidget(self.race_game, 2)

        # 脳波パワーグラフ（縦配置）
        graph_widget = pg.GraphicsLayoutWidget()
        right_panel.addWidget(graph_widget, 1)

        # 棒グラフプロット
        self.power_plot = graph_widget.addPlot(title="脳波 & 集中度")
        self.power_plot.setLabel('left', 'Level')
        self.power_plot.setLabel('bottom', '')
        self.power_plot.showGrid(y=True, alpha=0.3)
        self.power_plot.setYRange(0, 100, padding=0)  # Y軸を0-100に固定

        # 棒グラフ用のデータ（左右チャンネル別）
        self.bar_items = {}
        # θL, θR, αL, αR, βL, βR, FocusL, FocusR
        x_positions = [0, 0.5, 1.5, 2, 3, 3.5, 5, 5.5]
        colors_left = ['#FFD700', '#FFA500', '#4ECDC4', '#00CED1', '#FF6B6B', '#DC143C', '#96CEB4', '#5FA777']
        labels = ['Theta_L', 'Theta_R', 'Alpha_L', 'Alpha_R', 'Beta_L', 'Beta_R', 'Focus_L', 'Focus_R']

        for i, (x, label, color) in enumerate(zip(x_positions, labels, colors_left)):
            bar = pg.BarGraphItem(x=[x], height=[0], width=0.4, brush=color)
            self.power_plot.addItem(bar)
            self.bar_items[label] = bar

        # X軸のラベル設定
        x_dict = {0.25: 'θ', 1.75: 'α', 3.25: 'β', 5.25: 'Focus'}
        x_axis = self.power_plot.getAxis('bottom')
        x_axis.setTicks([list(x_dict.items())])

        layout.addWidget(left_scroll, 1)
        layout.addLayout(right_panel, 2)

    @qasync.asyncSlot()
    async def scan_devices(self):
        """Museデバイスをスキャン"""
        self.status_label.setText('Status: Scanning...')
        self.scan_button.setEnabled(False)

        try:
            devices = await BleakScanner.discover(timeout=10.0)
            self.device_combo.clear()

            for device in devices:
                if device.name and "muse" in device.name.lower():
                    self.device_combo.addItem(f"{device.name} ({device.address})", device.address)

            if self.device_combo.count() > 0:
                self.status_label.setText(f'Status: Found {self.device_combo.count()} device(s)')
                self.connect_button.setEnabled(True)
            else:
                self.status_label.setText('Status: No Muse devices found')
        except Exception as e:
            self.status_label.setText(f'Status: Scan error - {str(e)}')

        self.scan_button.setEnabled(True)

    def set_difficulty(self, difficulty):
        """難易度を設定"""
        self.current_difficulty = difficulty

        # 難易度に応じて障害物の出現頻度を設定
        self.race_game.difficulty_level = difficulty  # 難易度レベルを保存
        if difficulty == 'easy':
            self.race_game.difficulty_base_prob = 0.015  # Easy: 操作ONと同じ
            print("🟢 Difficulty: EASY (障害物: 少なめ)")
        elif difficulty == 'normal':
            self.race_game.difficulty_base_prob = 0.1  # Normal: 1.5%
            print("🟡 Difficulty: NORMAL (障害物: 標準)")
        elif difficulty == 'hard':
            self.race_game.difficulty_base_prob = 0.3  # Hard: 2.5%
            print("🔴 Difficulty: HARD (障害物: 多め)")

        self.update_difficulty_buttons()

    def update_difficulty_buttons(self):
        """難易度ボタンの表示を更新"""
        # すべてのボタンをリセット
        for btn in [self.easy_button, self.normal_button, self.hard_button]:
            btn.setStyleSheet("")

        # 選択中のボタンをハイライト
        selected_style = """
            QPushButton {
                background-color: #4ECDC4;
                color: white;
                font-weight: bold;
            }
        """

        if self.current_difficulty == 'easy':
            self.easy_button.setStyleSheet(selected_style)
        elif self.current_difficulty == 'normal':
            self.normal_button.setStyleSheet(selected_style)
        elif self.current_difficulty == 'hard':
            self.hard_button.setStyleSheet(selected_style)

    def toggle_brain_control(self, state):
        """脳波操作モードの切り替え"""
        self.race_game.brain_control_enabled = (state == QtCore.Qt.Checked)
        if self.race_game.brain_control_enabled:
            print("✅ 脳波による左右操作: 有効")
        else:
            print("❌ 脳波による左右操作: 無効（キーボードのみ）")

    @qasync.asyncSlot()
    async def connect_device(self):
        """デバイスに接続"""
        if not self.device_combo.currentData():
            return

        self.device_address = self.device_combo.currentData()
        self.status_label.setText('Status: Connecting...')

        try:
            self.client = BleakClient(self.device_address)
            await self.client.connect()

            if self.client.is_connected:
                self.status_label.setText('Status: Connected')
                self.start_button.setEnabled(True)
                self.tutorial_button.setEnabled(True)
                self.connect_button.setEnabled(False)
        except Exception as e:
            self.status_label.setText(f'Status: Connection error - {str(e)}')

    def _unpack_eeg_channel(self, packet):
        """EEGデータアンパック"""
        aa = bitstring.Bits(bytes=packet)
        pattern = "uint:16,uint:12,uint:12,uint:12,uint:12,uint:12,uint:12, \
                   uint:12,uint:12,uint:12,uint:12,uint:12,uint:12"
        res = aa.unpack(pattern)
        packet_index = res[0]
        data = res[1:]
        data = 0.48828125 * (np.array(data) - 2048)
        return packet_index, data

    def _init_timestamp_correction(self):
        """タイムスタンプ補正初期化"""
        self.sample_index = 0
        self._P = 1e-4
        t0 = time.time()
        self.reg_params = np.array([t0, 1. / MUSE_SAMPLING_EEG_RATE])

    def _update_timestamp_correction(self, t_source, t_receiver):
        """タイムスタンプ補正更新"""
        t_receiver = t_receiver - self.reg_params[0]
        P = self._P
        R = self.reg_params[1]
        P = P - ((P**2) * (t_source**2)) / (1 - (P * (t_source**2)))
        R = R + P * t_source * (t_receiver - t_source * R)
        self.reg_params[1] = R
        self._P = P

    def _handle_eeg(self, sender, data):
        """EEGデータハンドラー"""
        if self.first_sample:
            self._init_timestamp_correction()
            self.first_sample = False

        timestamp = time.time()
        sender_uuid = str(sender.uuid)

        if sender_uuid not in self.uuid_to_handle:
            return

        handle = self.uuid_to_handle[sender_uuid]
        index = int((handle - 32) / 3)
        tm, d = self._unpack_eeg_channel(data)

        if self.last_tm == 0:
            self.last_tm = tm - 1

        self.data[index] = d
        self.timestamps[index] = timestamp

        # 最後のデータを受信したら処理
        if handle == 35:
            if tm != self.last_tm + 1:
                if (tm - self.last_tm) != -65535:
                    self.sample_index += 12 * (tm - self.last_tm + 1)

            self.last_tm = tm
            idxs = np.arange(0, 12) + self.sample_index
            self.sample_index += 12

            self._update_timestamp_correction(idxs[-1], np.nanmin(self.timestamps))

            # データを解析器に追加
            channels = ['TP9', 'AF7', 'AF8', 'TP10']
            for i, channel in enumerate(channels):
                if i < 4:
                    self.analyzer.add_samples(channel, self.data[i])

            # 集中度スコアを更新
            self.focus_score = self.analyzer.get_focus_score()

            # 左右別の集中度スコアを更新
            self.focus_left, self.focus_right = self.analyzer.get_focus_scores_lr()

            # 左右バイアスを更新
            self.lateral_bias = self.analyzer.compute_lateral_bias()

            # 接触品質を評価（信号の標準偏差から）
            self._evaluate_contact_quality()

            self.timestamps = np.full(5, np.nan)
            self.data = np.zeros((5, 12))

    def _evaluate_contact_quality(self):
        """各チャンネルの接触品質を評価"""
        channels = ['TP9', 'AF7', 'AF8', 'TP10']
        for channel in channels:
            if len(self.analyzer.eeg_buffer[channel]) >= 128:  # 0.5秒分のデータ
                std = np.std(list(self.analyzer.eeg_buffer[channel]))

                if std < 20:
                    status_text = 'Good'
                    color = '#90EE90'  # Light green
                elif std < 50:
                    status_text = 'OK'
                    color = '#FFD700'  # Gold
                else:
                    status_text = 'Bad'
                    color = '#FF6B6B'  # Red

                self.contact_quality[channel] = status_text
                self.contact_labels[channel].setText(f'{channel}: {status_text}')
                self.contact_labels[channel].setStyleSheet(
                    f'padding: 5px; background-color: {color}; border-radius: 3px; font-weight: bold;'
                )

    def _write_cmd(self, cmd):
        """コマンド書き込み"""
        async def write_async():
            await self.client.write_gatt_char(MUSE_GATT_ATTR_STREAM_TOGGLE, bytearray(cmd), response=False)
        return asyncio.create_task(write_async())

    def _write_cmd_str(self, cmd):
        """文字列コマンド書き込み"""
        cmd_bytes = [len(cmd) + 1, *(ord(char) for char in cmd), ord('\n')]
        return self._write_cmd(cmd_bytes)

    async def _start_streaming(self, tutorial_mode=False):
        """ストリーミング開始（共通処理）"""
        if not self.client or not self.client.is_connected:
            return

        self.status_label.setText('Status: Starting...')

        # EEG通知設定
        eeg_characteristics = [
            MUSE_GATT_ATTR_TP9,
            MUSE_GATT_ATTR_AF7,
            MUSE_GATT_ATTR_AF8,
            MUSE_GATT_ATTR_TP10,
            MUSE_GATT_ATTR_RIGHTAUX
        ]

        for char_uuid in eeg_characteristics:
            await self.client.start_notify(char_uuid, self._handle_eeg)

        # Museコマンド送信
        preset_cmd = [0x04, 0x70, 0x32, 0x31, 0x0a]
        await self.client.write_gatt_char(MUSE_GATT_ATTR_STREAM_TOGGLE, bytearray(preset_cmd), response=False)
        await asyncio.sleep(1)

        await self._write_cmd_str('d')
        await asyncio.sleep(0.5)
        await self._write_cmd_str('d')

        self.is_streaming = True

        # ゲームリセット
        self.race_game.current_lane = 1
        self.race_game.speed = 0.0
        self.race_game.distance = 0.0
        self.race_game.obstacles = []
        self.race_game.game_over = False
        self.race_game.game_clear = False
        self.race_game.score = 0
        self.race_game.last_obstacle_lane = -1
        self.race_game.obstacle_cooldown = 0
        self.race_game.bias_cooldown = 0
        self.race_game.tutorial_mode = tutorial_mode
        self.race_game.remaining_time = 30.0
        self.race_game.start_time = None

        # UI更新
        self.start_button.setEnabled(False)
        self.tutorial_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.retry_button.setEnabled(False)

        # ゲームループ開始
        self.game_timer.start(33)  # 30 FPS

        if tutorial_mode:
            self.status_label.setText('Status: Tutorial Mode - No obstacles!')
        else:
            self.status_label.setText('Status: Game Running!')

    @qasync.asyncSlot()
    async def start_game(self):
        """ゲーム開始"""
        try:
            await self._start_streaming(tutorial_mode=False)
        except Exception as e:
            self.status_label.setText(f'Status: Start error - {str(e)}')

    @qasync.asyncSlot()
    async def start_tutorial(self):
        """チュートリアルモード開始"""
        try:
            await self._start_streaming(tutorial_mode=True)
        except Exception as e:
            self.status_label.setText(f'Status: Start error - {str(e)}')

    @qasync.asyncSlot()
    async def retry_game(self):
        """ゲームをリトライ"""
        # ゲームリセット
        self.race_game.current_lane = 1
        self.race_game.speed = 0.0
        self.race_game.distance = 0.0
        self.race_game.obstacles = []
        self.race_game.game_over = False
        self.race_game.game_clear = False
        self.race_game.score = 0
        self.race_game.last_obstacle_lane = -1
        self.race_game.obstacle_cooldown = 0
        self.race_game.bias_cooldown = 0
        self.race_game.remaining_time = 30.0
        self.race_game.start_time = None

        # UI更新
        self.retry_button.setEnabled(False)
        self.status_label.setText('Status: Game Running!')

        print("🔄 Game Restarted!")

    @qasync.asyncSlot()
    async def stop_game(self):
        """ゲーム停止"""
        if self.is_streaming:
            try:
                self.game_timer.stop()

                if self.client and self.client.is_connected:
                    await self._write_cmd_str('h')
                    await asyncio.sleep(0.5)

                    eeg_characteristics = [
                        MUSE_GATT_ATTR_TP9,
                        MUSE_GATT_ATTR_AF7,
                        MUSE_GATT_ATTR_AF8,
                        MUSE_GATT_ATTR_TP10,
                        MUSE_GATT_ATTR_RIGHTAUX
                    ]

                    for char_uuid in eeg_characteristics:
                        await self.client.stop_notify(char_uuid)

                self.is_streaming = False
                self.status_label.setText('Status: Game Stopped')

                self.start_button.setEnabled(True)
                self.tutorial_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.retry_button.setEnabled(False)

            except Exception as e:
                self.status_label.setText(f'Status: Stop error - {str(e)}')

    def update_game(self):
        """ゲーム状態を更新"""
        if not self.is_streaming:
            return

        # ゲームを更新（左右バイアスを渡す）
        self.race_game.update_game(self.focus_score, self.lateral_bias)

        # ゲームオーバー時の処理
        if self.race_game.game_over and not self.retry_button.isEnabled():
            self.retry_button.setEnabled(True)
            self.status_label.setText('Status: Game Over! Press Retry to play again')

        # ゲームクリア時の処理
        if self.race_game.game_clear and not self.retry_button.isEnabled():
            self.retry_button.setEnabled(True)
            self.status_label.setText(f'Status: Game Clear! Score: {self.race_game.score}')

        # UI更新
        self.focus_bar.setValue(int(self.focus_score * 100))
        self.speed_label.setText(f'{self.race_game.speed * 10:.1f} km/h')
        self.distance_label.setText(f'{int(self.race_game.distance)} m')
        self.score_label.setText(f'{self.race_game.score}')

        # 左右バイアス表示
        bias_percent = int(self.lateral_bias * 100)
        self.bias_bar.setValue(bias_percent)
        if self.lateral_bias < 0:
            self.bias_label.setText(f'Left: {abs(bias_percent)}% | Right: 0%')
        else:
            self.bias_label.setText(f'Left: 0% | Right: {bias_percent}%')

        # 棒グラフ更新（左右チャンネル別）
        powers = self.analyzer.last_powers
        # 左右すべてのパワー値を取得
        theta_left = powers.get('theta_left', 0)
        theta_right = powers.get('theta_right', 0)
        alpha_left = powers.get('alpha_left', 0)
        alpha_right = powers.get('alpha_right', 0)
        beta_left = powers.get('beta_left', 0)
        beta_right = powers.get('beta_right', 0)

        # 対数スケールでパワー値を変換（0-100スケール）
        # log10を使用し、EEGパワーの典型的な範囲（10^2 〜 10^8）にマッピング
        def power_to_log_scale(power, min_val=1e2, max_val=1e8):
            """パワー値を対数スケールの0-100に変換"""
            if power <= 0:
                power = min_val
            log_power = np.log10(np.clip(power, min_val, max_val))
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            # 0-100スケールに正規化
            normalized = (log_power - log_min) / (log_max - log_min) * 100
            return max(0, min(100, normalized))

        self.bar_items['Theta_L'].setOpts(height=[power_to_log_scale(theta_left)])
        self.bar_items['Theta_R'].setOpts(height=[power_to_log_scale(theta_right)])
        self.bar_items['Alpha_L'].setOpts(height=[power_to_log_scale(alpha_left)])
        self.bar_items['Alpha_R'].setOpts(height=[power_to_log_scale(alpha_right)])
        self.bar_items['Beta_L'].setOpts(height=[power_to_log_scale(beta_left)])
        self.bar_items['Beta_R'].setOpts(height=[power_to_log_scale(beta_right)])
        self.bar_items['Focus_L'].setOpts(height=[self.focus_left * 100])  # 0-100スケール
        self.bar_items['Focus_R'].setOpts(height=[self.focus_right * 100])  # 0-100スケール

    async def disconnect(self):
        """切断"""
        if self.is_streaming:
            await self.stop_game()

        if self.client and self.client.is_connected:
            await self.client.disconnect()

    def closeEvent(self, event):
        """終了処理"""
        if hasattr(self, 'client') and self.client:
            asyncio.create_task(self.disconnect())
        event.accept()

class MuseRaceApp:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.game = MuseRaceGame()

    def run(self):
        """アプリケーション実行"""
        self.game.show()

        loop = qasync.QEventLoop(self.app)
        asyncio.set_event_loop(loop)

        with loop:
            loop.run_forever()

def main():
    print("Muse Mind Control Racing Game")
    print("=" * 40)
    print("集中すると車が速くなります！")
    print()
    print("【操作方法】")
    print("← → : レーン切替（3レーン）")
    print("集中度 : スピードをコントロール")
    print()
    print("【ゲームの流れ】")
    print("1. Scan for Muse devices")
    print("2. Connect to your device")
    print("3. Tutorial Mode で練習（障害物なし）")
    print("4. Start Game で本番プレイ")
    print("5. ← →キーでレーンを切り替えて障害物を回避")
    print()

    app = MuseRaceApp()
    app.run()

if __name__ == "__main__":
    main()
