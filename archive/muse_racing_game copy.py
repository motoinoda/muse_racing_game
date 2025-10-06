#!/usr/bin/env python3
"""
Muse Mind Control Racing Game
脳波（集中度）でスピードをコントロールするレースゲーム
"""

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
            'gamma': (30, 50)
        }

        self.last_powers = {band: 0.0 for band in self.bands.keys()}

    def add_samples(self, channel, samples):
        """サンプルをバッファに追加"""
        for sample in samples:
            self.eeg_buffer[channel].append(sample)

    def compute_band_powers(self):
        """周波数帯域ごとのパワーを計算"""
        if len(self.eeg_buffer['AF7']) < self.window_size:
            return self.last_powers

        # 前額部（AF7, AF8）の平均を使用
        data_af7 = np.array(self.eeg_buffer['AF7'])
        data_af8 = np.array(self.eeg_buffer['AF8'])
        data = (data_af7 + data_af8) / 2.0

        # FFT計算
        fft_vals = np.fft.rfft(data)
        fft_freq = np.fft.rfftfreq(len(data), 1.0 / MUSE_SAMPLING_EEG_RATE)
        fft_power = np.abs(fft_vals) ** 2

        # 各周波数帯域のパワーを計算
        powers = {}
        for band_name, (low_freq, high_freq) in self.bands.items():
            idx = np.logical_and(fft_freq >= low_freq, fft_freq <= high_freq)
            powers[band_name] = np.sum(fft_power[idx])

        self.last_powers = powers
        return powers

    def get_focus_score(self):
        """集中度スコアを計算（0.0-1.0）"""
        powers = self.compute_band_powers()

        # β波（集中）/ (α波（リラックス）+ θ波（眠気）)
        beta = powers['beta']
        alpha = powers['alpha']
        theta = powers['theta']

        if alpha + theta == 0:
            return 0.0

        focus = beta / (alpha + theta)

        # 0-1の範囲に正規化（経験的な係数）
        focus_normalized = np.clip(focus / 2.0, 0.0, 1.0)

        return focus_normalized

class RaceGame(QtWidgets.QWidget):
    """レースゲーム画面"""

    def __init__(self):
        super().__init__()
        # 3レーンシステム
        self.lanes = [120, 200, 280]  # 左、中央、右のレーン位置
        self.current_lane = 1  # 0=左、1=中央、2=右
        self.car_x = self.lanes[self.current_lane]  # 車のX位置
        self.car_y = 500  # 車のY位置（固定）
        self.speed = 0.0  # 現在のスピード
        self.distance = 0.0  # 走行距離
        self.obstacles = []  # 障害物リスト
        self.game_over = False
        self.score = 0

        # 色設定
        self.road_color = QtGui.QColor(80, 80, 80)
        self.car_color = QtGui.QColor(255, 0, 0)
        self.obstacle_color = QtGui.QColor(100, 100, 200)
        self.line_color = QtGui.QColor(255, 255, 255)

        self.setMinimumSize(400, 600)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)  # キーボード入力を受け取る

    def keyPressEvent(self, event):
        """キーボード入力処理"""
        if self.game_over:
            return

        # 左右の矢印キーでレーンを移動
        if event.key() == QtCore.Qt.Key_Left:
            if self.current_lane > 0:
                self.current_lane -= 1
                self.car_x = self.lanes[self.current_lane]
        elif event.key() == QtCore.Qt.Key_Right:
            if self.current_lane < 2:
                self.current_lane += 1
                self.car_x = self.lanes[self.current_lane]

        self.update()

    def update_game(self, focus_score):
        """ゲーム状態を更新"""
        if self.game_over:
            return

        # スピードを集中度に基づいて更新
        target_speed = focus_score * 10.0  # 最大10ピクセル/フレーム
        self.speed = self.speed * 0.9 + target_speed * 0.1  # スムージング

        # 走行距離を更新
        self.distance += self.speed

        # スコア計算
        self.score = int(self.distance + self.speed * 100)

        # 障害物の生成（ランダムなレーンに配置）
        if np.random.random() < 0.02:
            lane = np.random.randint(0, 3)
            x_pos = self.lanes[lane]
            self.obstacles.append([x_pos, -50])

        # 障害物を移動
        for obstacle in self.obstacles[:]:
            obstacle[1] += self.speed * 2

            # 画面外に出たら削除
            if obstacle[1] > 650:
                self.obstacles.remove(obstacle)

            # 衝突判定
            if abs(obstacle[0] - self.car_x) < 40 and abs(obstacle[1] - self.car_y) < 40:
                self.game_over = True

        self.update()

    def paintEvent(self, event):
        """描画処理"""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # 背景（道路）
        painter.fillRect(0, 0, 400, 600, QtGui.QColor(50, 150, 50))
        painter.fillRect(60, 0, 280, 600, self.road_color)

        # レーン区切り線（2本）
        painter.setPen(QtGui.QPen(QtCore.Qt.white, 2, QtCore.Qt.DashLine))
        painter.drawLine(160, 0, 160, 600)  # 左と中央の境界
        painter.drawLine(240, 0, 240, 600)  # 中央と右の境界

        # 車を描画
        painter.setBrush(self.car_color)
        painter.drawRect(int(self.car_x) - 20, int(self.car_y) - 30, 40, 60)

        # 障害物を描画
        painter.setBrush(self.obstacle_color)
        for obs in self.obstacles:
            painter.drawRect(int(obs[0]) - 20, int(obs[1]) - 20, 40, 40)

        # ゲームオーバー表示
        if self.game_over:
            painter.setPen(QtCore.Qt.red)
            painter.setFont(QtGui.QFont('Arial', 40, QtGui.QFont.Bold))
            painter.drawText(50, 300, 'GAME OVER!')

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

        # UI初期化
        self.init_ui()

        # ゲーム更新タイマー
        self.game_timer = QtCore.QTimer()
        self.game_timer.timeout.connect(self.update_game)

    def init_ui(self):
        """UI初期化"""
        self.setWindowTitle('Muse Mind Control Racing Game')
        self.setGeometry(100, 100, 1000, 700)

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout(central_widget)

        # 左側: コントロールパネル
        left_panel = QtWidgets.QVBoxLayout()

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

        self.stop_button = QtWidgets.QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_game)
        self.stop_button.setEnabled(False)
        left_panel.addWidget(self.stop_button)

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

        # 脳波パワー表示
        left_panel.addWidget(QtWidgets.QLabel('Brain Waves:'))
        self.wave_labels = {}
        for band in ['alpha', 'beta', 'theta']:
            label = QtWidgets.QLabel(f'{band.capitalize()}: 0')
            self.wave_labels[band] = label
            left_panel.addWidget(label)

        left_panel.addWidget(QtWidgets.QLabel(''))

        # 操作説明
        controls_label = QtWidgets.QLabel('【操作方法】\n← → : レーン切替\n      (3レーン)\n集中 : スピードUP')
        controls_label.setStyleSheet('background-color: #f0f0f0; padding: 10px; border-radius: 5px;')
        left_panel.addWidget(controls_label)

        left_panel.addStretch()

        # 右側: ゲーム画面とグラフ
        right_panel = QtWidgets.QVBoxLayout()

        # ゲーム画面
        self.race_game = RaceGame()
        right_panel.addWidget(self.race_game, 3)

        # 脳波パワーグラフ
        graph_widget = pg.GraphicsLayoutWidget()
        graph_widget.setMaximumHeight(200)
        right_panel.addWidget(graph_widget, 1)

        # 棒グラフプロット
        self.power_plot = graph_widget.addPlot(title="脳波パワー & 集中度")
        self.power_plot.setLabel('left', 'パワー / スコア')
        self.power_plot.setLabel('bottom', '指標')
        self.power_plot.showGrid(y=True, alpha=0.3)

        # 棒グラフ用のデータ
        self.bar_items = {}
        x_positions = [0, 1, 2, 3]  # θ, α, β, Focus
        colors = ['#FFD700', '#4ECDC4', '#FF6B6B', '#96CEB4']  # 金、シアン、赤、緑
        labels = ['Theta', 'Alpha', 'Beta', 'Focus']

        for i, (x, color, label) in enumerate(zip(x_positions, colors, labels)):
            bar = pg.BarGraphItem(x=[x], height=[0], width=0.6, brush=color)
            self.power_plot.addItem(bar)
            self.bar_items[label] = bar

        # X軸のラベル設定
        x_dict = dict(enumerate(labels))
        x_axis = self.power_plot.getAxis('bottom')
        x_axis.setTicks([list(x_dict.items())])

        layout.addLayout(left_panel, 1)
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

            self.timestamps = np.full(5, np.nan)
            self.data = np.zeros((5, 12))

    def _write_cmd(self, cmd):
        """コマンド書き込み"""
        async def write_async():
            await self.client.write_gatt_char(MUSE_GATT_ATTR_STREAM_TOGGLE, bytearray(cmd), response=False)
        return asyncio.create_task(write_async())

    def _write_cmd_str(self, cmd):
        """文字列コマンド書き込み"""
        cmd_bytes = [len(cmd) + 1, *(ord(char) for char in cmd), ord('\n')]
        return self._write_cmd(cmd_bytes)

    @qasync.asyncSlot()
    async def start_game(self):
        """ゲーム開始"""
        if not self.client or not self.client.is_connected:
            return

        try:
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
            self.race_game.car_x = self.race_game.lanes[1]
            self.race_game.car_y = 500
            self.race_game.speed = 0.0
            self.race_game.distance = 0.0
            self.race_game.obstacles = []
            self.race_game.game_over = False
            self.race_game.score = 0

            # UI更新
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            # ゲームループ開始
            self.game_timer.start(33)  # 30 FPS

            self.status_label.setText('Status: Game Running!')

        except Exception as e:
            self.status_label.setText(f'Status: Start error - {str(e)}')

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
                self.stop_button.setEnabled(False)

            except Exception as e:
                self.status_label.setText(f'Status: Stop error - {str(e)}')

    def update_game(self):
        """ゲーム状態を更新"""
        if not self.is_streaming:
            return

        # ゲームを更新
        self.race_game.update_game(self.focus_score)

        # UI更新
        self.focus_bar.setValue(int(self.focus_score * 100))
        self.speed_label.setText(f'{self.race_game.speed * 10:.1f} km/h')
        self.distance_label.setText(f'{int(self.race_game.distance)} m')
        self.score_label.setText(f'{self.race_game.score}')

        # 脳波パワー表示
        powers = self.analyzer.last_powers
        theta_power = powers.get('theta', 0)
        alpha_power = powers.get('alpha', 0)
        beta_power = powers.get('beta', 0)

        for band in ['alpha', 'beta', 'theta']:
            power = powers.get(band, 0)
            self.wave_labels[band].setText(f'{band.capitalize()}: {int(power)}')

        # 棒グラフ更新
        # パワー値を正規化して表示（最大値を基準に）
        max_power = max(theta_power, alpha_power, beta_power, 1)

        self.bar_items['Theta'].setOpts(height=[theta_power / max_power * 100])
        self.bar_items['Alpha'].setOpts(height=[alpha_power / max_power * 100])
        self.bar_items['Beta'].setOpts(height=[beta_power / max_power * 100])
        self.bar_items['Focus'].setOpts(height=[self.focus_score * 100])  # 0-100スケール

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
    print("3. Start Game!")
    print("4. ← →キーでレーンを切り替えて障害物を回避")
    print()

    app = MuseRaceApp()
    app.run()

if __name__ == "__main__":
    main()
