#!/usr/bin/env python3
"""
Muse Real-time EEG Power Viewer
リアルタイムでθ（4-8Hz）、α（8-13Hz）、β（13-30Hz）パワーを可視化
"""

import sys
import asyncio
import numpy as np
import time
import bitstring
from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from bleak import BleakScanner, BleakClient
import qasync
from scipy.fft import fft, fftfreq
from scipy.signal import welch

# Muse constants
MUSE_SAMPLING_EEG_RATE = 256
MUSE_GATT_ATTR_STREAM_TOGGLE = '273e0001-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_TP9 = '273e0003-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_AF7 = '273e0004-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_AF8 = '273e0005-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_TP10 = '273e0006-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_RIGHTAUX = '273e0007-4c4d-454d-96be-f03bac821358'

class PowerAnalyzer:
    """パワースペクトラム解析クラス"""
    def __init__(self, sample_rate=256, window_size=1.0):  # 1秒に短縮
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_size)
        
        # 周波数帯域定義
        self.bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30)
        }
        
        # デバッグ用カウンター
        self.calculation_count = 0
        
    def calculate_band_power(self, data):
        """周波数帯域別パワー計算"""
        self.calculation_count += 1
        
        if len(data) < self.window_samples:
            print(f"Debug: Not enough data - have {len(data)}, need {self.window_samples}")
            return {'theta': 0, 'alpha': 0, 'beta': 0}
        
        # 最新のwindow_samples分のデータを使用
        signal = np.array(data[-self.window_samples:])
        
        # データの基本統計
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        signal_range = np.ptp(signal)
        
        # DC成分除去
        signal = signal - signal_mean
        
        # Welch法でパワースペクトル密度計算（パラメータ調整）
        try:
            # npersegを調整してより適切な周波数分解能を得る
            nperseg = min(self.window_samples // 2, 128)  # より小さなセグメント
            freqs, psd = welch(signal, 
                             fs=self.sample_rate, 
                             nperseg=nperseg,
                             noverlap=nperseg//2,
                             window='hann')
            
            # デバッグ情報（最初の数回のみ）
            if self.calculation_count <= 3:
                print(f"Debug calc #{self.calculation_count}:")
                print(f"  Signal: mean={signal_mean:.2f}, std={signal_std:.2f}, range={signal_range:.2f}")
                print(f"  Freqs range: {freqs[0]:.2f}-{freqs[-1]:.2f} Hz")
                print(f"  PSD range: {np.min(psd):.2e}-{np.max(psd):.2e}")
                
        except Exception as e:
            print(f"Welch calculation error: {e}")
            return {'theta': 0, 'alpha': 0, 'beta': 0}
        
        # 各帯域のパワー計算
        powers = {}
        for band_name, (low_freq, high_freq) in self.bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                powers[band_name] = band_power
                
                # デバッグ情報（最初の数回のみ）
                if self.calculation_count <= 3:
                    band_freqs = freqs[band_mask]
                    band_psd = psd[band_mask]
                    print(f"  {band_name}: {len(band_freqs)} freq points, power={band_power:.2e}")
            else:
                powers[band_name] = 0
                
        return powers
    
    def calculate_relative_power(self, powers):
        """相対パワー計算"""
        total_power = sum(powers.values())
        if total_power == 0:
            return {k: 0 for k in powers.keys()}
        return {k: v/total_power for k, v in powers.items()}

class MusePowerViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.client = None
        self.device_address = None
        self.is_streaming = False
        
        # EEGデータバッファ（長めに設定してパワー計算に使用）
        self.buffer_size = 1024  # 4秒分程度
        self.eeg_data = {
            'TP9': deque(maxlen=self.buffer_size),
            'AF7': deque(maxlen=self.buffer_size),
            'AF8': deque(maxlen=self.buffer_size),
            'TP10': deque(maxlen=self.buffer_size)
        }
        
        # パワー成分データ
        self.power_history_size = 60  # 30秒分（0.5秒毎）
        self.power_data = {
            'TP9': {'theta': deque(maxlen=self.power_history_size),
                   'alpha': deque(maxlen=self.power_history_size),
                   'beta': deque(maxlen=self.power_history_size)},
            'AF7': {'theta': deque(maxlen=self.power_history_size),
                   'alpha': deque(maxlen=self.power_history_size),
                   'beta': deque(maxlen=self.power_history_size)},
            'AF8': {'theta': deque(maxlen=self.power_history_size),
                   'alpha': deque(maxlen=self.power_history_size),
                   'beta': deque(maxlen=self.power_history_size)},
            'TP10': {'theta': deque(maxlen=self.power_history_size),
                    'alpha': deque(maxlen=self.power_history_size),
                    'beta': deque(maxlen=self.power_history_size)}
        }
        
        # パワー解析器
        self.power_analyzer = PowerAnalyzer()
        
        # muse-lsl互換のデータ処理変数
        self.timestamps = np.full(5, np.nan)
        self.data = np.zeros((5, 12))
        self.last_tm = 0
        self.first_sample = True
        self.sample_index = 0
        self.reg_params = None
        self._P = 1e-4
        
        # ハンドルマッピング
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
        
        # UI色設定
        self.band_colors = {
            'theta': '#FF6B6B',  # 赤系
            'alpha': '#4ECDC4',  # 緑系
            'beta': '#45B7D1'    # 青系
        }
        
        # 統計
        self.sample_count = 0
        self.start_time = None
        self.last_power_calc = 0
        
        # UI初期化
        self.init_ui()
        
        # タイマー設定
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        
        self.power_timer = QtCore.QTimer()
        self.power_timer.timeout.connect(self.calculate_powers)
    
    def init_ui(self):
        """UI初期化"""
        self.setWindowTitle('Muse EEG Power Viewer - θ/α/β Bands')
        self.setGeometry(50, 50, 1400, 900)
        
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Control panel
        control_layout = QtWidgets.QHBoxLayout()
        
        self.scan_button = QtWidgets.QPushButton('Scan for Muse')
        self.scan_button.clicked.connect(self.scan_devices)
        control_layout.addWidget(self.scan_button)
        
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.setMinimumWidth(300)
        control_layout.addWidget(QtWidgets.QLabel('Device:'))
        control_layout.addWidget(self.device_combo)
        
        self.connect_button = QtWidgets.QPushButton('Connect')
        self.connect_button.clicked.connect(self.connect_device)
        self.connect_button.setEnabled(False)
        control_layout.addWidget(self.connect_button)
        
        self.start_button = QtWidgets.QPushButton('Start Streaming')
        self.start_button.clicked.connect(self.start_streaming)
        self.start_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QtWidgets.QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_streaming)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        # Status
        self.status_label = QtWidgets.QLabel('Status: Ready')
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # Main plot area - 分割して上下に配置
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(splitter)
        
        # 上部: リアルタイム波形（縮小版）
        wave_widget = pg.GraphicsLayoutWidget()
        wave_widget.setMaximumHeight(300)
        splitter.addWidget(wave_widget)
        
        # 下部: パワースペクトラム
        power_widget = pg.GraphicsLayoutWidget()
        splitter.addWidget(power_widget)
        
        # リアルタイム波形プロット（簡略版）
        self.wave_plots = {}
        self.wave_curves = {}
        channels = ['TP9', 'AF7', 'AF8', 'TP10']
        
        for i, channel in enumerate(channels):
            plot = wave_widget.addPlot(title=f'Raw EEG - {channel}')
            plot.setLabel('left', 'μV')
            plot.setLabel('bottom', 'Samples')
            plot.showGrid(True, alpha=0.3)
            plot.setYRange(-100, 100)
            
            curve = plot.plot(pen=pg.mkPen(color='gray', width=1))
            
            self.wave_plots[channel] = plot
            self.wave_curves[channel] = curve
            
            if i % 2 == 1:
                wave_widget.nextRow()
        
        # パワースペクトラムプロット
        self.power_plots = {}
        self.power_curves = {}
        
        for i, channel in enumerate(channels):
            plot = power_widget.addPlot(title=f'EEG Power Bands - {channel}')
            plot.setLabel('left', 'Power', units='μV²')
            plot.setLabel('bottom', 'Time (0.5s intervals)')
            plot.showGrid(True, alpha=0.3)
            plot.addLegend()
            
            curves = {}
            for band, color in self.band_colors.items():
                curve = plot.plot(pen=pg.mkPen(color=color, width=2), name=f'{band.capitalize()} ({self.power_analyzer.bands[band][0]}-{self.power_analyzer.bands[band][1]}Hz)')
                curves[band] = curve
            
            self.power_plots[channel] = plot
            self.power_curves[channel] = curves
            
            if i % 2 == 1:
                power_widget.nextRow()
        
        # Statistics panel
        stats_layout = QtWidgets.QHBoxLayout()
        self.stats_label = QtWidgets.QLabel('Samples: 0 | Rate: 0 Hz | Power Updates: 0')
        stats_layout.addWidget(self.stats_label)
        
        # Current power values display
        self.power_display = QtWidgets.QLabel('Current Powers: Calculating...')
        self.power_display.setStyleSheet('font-family: monospace; background-color: #f0f0f0; padding: 5px;')
        stats_layout.addWidget(self.power_display)
        
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
    
    @qasync.asyncSlot()
    async def scan_devices(self):
        """Museデバイスをスキャン"""
        self.status_label.setText('Status: Scanning for devices...')
        self.scan_button.setEnabled(False)
        
        try:
            print("Scanning for Muse devices...")
            devices = await BleakScanner.discover(timeout=10.0)
            
            self.device_combo.clear()
            muse_devices = []
            
            for device in devices:
                if device.name and "muse" in device.name.lower():
                    muse_devices.append(device)
                    display_name = f"{device.name} ({device.address})"
                    self.device_combo.addItem(display_name, device.address)
                    print(f"Found: {device.name} ({device.address})")
            
            if muse_devices:
                self.status_label.setText(f'Status: Found {len(muse_devices)} Muse device(s)')
                self.connect_button.setEnabled(True)
            else:
                self.status_label.setText('Status: No Muse devices found')
                
        except Exception as e:
            self.status_label.setText(f'Status: Scan error - {str(e)}')
            print(f"Scan error: {e}")
        
        self.scan_button.setEnabled(True)
    
    @qasync.asyncSlot()
    async def connect_device(self):
        """デバイスに接続"""
        if not self.device_combo.currentData():
            self.status_label.setText('Status: No device selected')
            return
        
        self.device_address = self.device_combo.currentData()
        self.status_label.setText('Status: Connecting...')
        
        try:
            print(f"Connecting to {self.device_address}...")
            self.client = BleakClient(self.device_address)
            await self.client.connect()
            
            if self.client.is_connected:
                self.status_label.setText('Status: Connected')
                self.start_button.setEnabled(True)
                self.connect_button.setEnabled(False)
                print("Connected to Muse!")
            else:
                self.status_label.setText('Status: Connection failed')
                
        except Exception as e:
            self.status_label.setText(f'Status: Connection error - {str(e)}')
            print(f"Connection error: {e}")
    
    def _unpack_eeg_channel(self, packet):
        """EEGデータアンパック処理"""
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
        
        # AF7（handle 35）を最後に受信したら処理
        if handle == 35:
            if tm != self.last_tm + 1:
                if (tm - self.last_tm) != -65535:
                    self.sample_index += 12 * (tm - self.last_tm + 1)
            
            self.last_tm = tm
            
            idxs = np.arange(0, 12) + self.sample_index
            self.sample_index += 12
            
            self._update_timestamp_correction(idxs[-1], np.nanmin(self.timestamps))
            
            # データをバッファに追加
            channels = ['TP9', 'AF7', 'AF8', 'TP10']
            for i, channel in enumerate(channels):
                if i < 4:
                    for sample in self.data[i]:
                        self.eeg_data[channel].append(sample)
            
            self.sample_count += 12
            
            # データリセット
            self.timestamps = np.full(5, np.nan)
            self.data = np.zeros((5, 12))
    
    def _write_cmd_str(self, cmd):
        """コマンド送信"""
        async def write_async():
            cmd_bytes = [len(cmd) + 1, *(ord(char) for char in cmd), ord('\n')]
            await self.client.write_gatt_char(MUSE_GATT_ATTR_STREAM_TOGGLE, bytearray(cmd_bytes), response=False)
        return asyncio.create_task(write_async())
    
    @qasync.asyncSlot()
    async def start_streaming(self):
        """ストリーミング開始"""
        if not self.client or not self.client.is_connected:
            self.status_label.setText('Status: Not connected')
            return
        
        try:
            self.status_label.setText('Status: Starting streaming...')
            
            # EEG通知設定
            eeg_characteristics = [
                MUSE_GATT_ATTR_TP9,
                MUSE_GATT_ATTR_AF7,
                MUSE_GATT_ATTR_AF8,
                MUSE_GATT_ATTR_TP10,
                MUSE_GATT_ATTR_RIGHTAUX
            ]
            
            for char_uuid in eeg_characteristics:
                try:
                    await self.client.start_notify(char_uuid, self._handle_eeg)
                except Exception as e:
                    print(f"Failed to start notifications for {char_uuid}: {e}")
            
            # Museコマンド送信
            preset_cmd = [0x04, 0x70, 0x32, 0x31, 0x0a]  # p21
            await self.client.write_gatt_char(MUSE_GATT_ATTR_STREAM_TOGGLE, bytearray(preset_cmd), response=False)
            await asyncio.sleep(1)
            
            await self._write_cmd_str('d')
            await asyncio.sleep(0.5)
            await self._write_cmd_str('d')
            
            self.is_streaming = True
            self.start_time = time.time()
            self.sample_count = 0
            self.last_power_calc = 0
            
            # UI更新
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
            # タイマー開始
            self.plot_timer.start(100)    # 10 FPS for plots
            self.power_timer.start(500)   # 0.5秒毎にパワー計算
            
            self.status_label.setText('Status: Streaming active')
            print("✅ EEG power analysis started")
            
        except Exception as e:
            self.status_label.setText(f'Status: Streaming error - {str(e)}')
            print(f"❌ Streaming error: {e}")
    
    @qasync.asyncSlot()
    async def stop_streaming(self):
        """ストリーミング停止"""
        if self.is_streaming:
            try:
                # タイマー停止
                self.plot_timer.stop()
                self.power_timer.stop()
                
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
                        try:
                            await self.client.stop_notify(char_uuid)
                        except:
                            pass
                
                self.is_streaming = False
                self.status_label.setText('Status: Streaming stopped')
                
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                
                print("✅ Streaming stopped")
                
            except Exception as e:
                self.status_label.setText(f'Status: Stop error - {str(e)}')
                print(f"❌ Stop error: {e}")
    
    def calculate_powers(self):
        """パワー成分計算（0.5秒毎）"""
        if not self.is_streaming:
            return
        
        channels = ['TP9', 'AF7', 'AF8', 'TP10']
        current_powers = {}
        
        for channel in channels:
            data_length = len(self.eeg_data[channel])
            
            # 最低1秒分のデータが必要（256サンプル）
            if data_length >= self.power_analyzer.window_samples:
                # データの統計情報を表示（デバッグ）
                if self.last_power_calc < 3:
                    recent_data = list(self.eeg_data[channel])[-100:]  # 最新100サンプル
                    print(f"Channel {channel}: {data_length} samples, recent range: [{np.min(recent_data):.1f}, {np.max(recent_data):.1f}]")
                
                powers = self.power_analyzer.calculate_band_power(self.eeg_data[channel])
                
                # パワーデータに追加
                for band in ['theta', 'alpha', 'beta']:
                    self.power_data[channel][band].append(powers[band])
                
                current_powers[channel] = powers
            else:
                # データが不足している場合の情報表示
                if self.last_power_calc < 5:
                    print(f"Channel {channel}: Not enough data - {data_length}/{self.power_analyzer.window_samples}")
        
        # 現在のパワー値を表示（より読みやすい形式）
        if current_powers:
            power_lines = []
            for channel in channels:
                if channel in current_powers:
                    powers = current_powers[channel]
                    # 対数スケールで表示を改善
                    theta_log = np.log10(max(powers['theta'], 1e-10))
                    alpha_log = np.log10(max(powers['alpha'], 1e-10))
                    beta_log = np.log10(max(powers['beta'], 1e-10))
                    
                    power_lines.append(f"{channel}: θ:{theta_log:.1f} α:{alpha_log:.1f} β:{beta_log:.1f} (log10)")
            
            self.power_display.setText(" | ".join(power_lines))
        
        self.last_power_calc += 1
    
    def update_plots(self):
        """プロット更新"""
        if not self.is_streaming:
            return
        
        # 統計更新
        if self.start_time:
            elapsed = time.time() - self.start_time
            rate = self.sample_count / elapsed if elapsed > 0 else 0
            self.stats_label.setText(f'Samples: {self.sample_count} | Rate: {rate:.1f} Hz | Power Updates: {self.last_power_calc}')
        
        channels = ['TP9', 'AF7', 'AF8', 'TP10']
        
        # 波形プロット更新（簡略版）
        for channel in channels:
            data = np.array(self.eeg_data[channel])
            if len(data) > 0:
                # 最新256サンプル（1秒分）のみ表示
                display_data = data[-256:] if len(data) >= 256 else data
                x = np.arange(len(display_data))
                self.wave_curves[channel].setData(x, display_data)
        
        # パワープロット更新
        for channel in channels:
            for band in ['theta', 'alpha', 'beta']:
                power_values = np.array(self.power_data[channel][band])
                if len(power_values) > 0:
                    x = np.arange(len(power_values))
                    self.power_curves[channel][band].setData(x, power_values)
    
    async def disconnect(self):
        """デバイス切断"""
        if self.is_streaming:
            await self.stop_streaming()
        
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            self.status_label.setText('Status: Disconnected')
            print("Disconnected")
    
    def closeEvent(self, event):
        """アプリケーション終了時の処理"""
        if hasattr(self, 'client') and self.client:
            asyncio.create_task(self.disconnect())
        event.accept()

class MusePowerApp:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        
        # pyqtgraph設定
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        self.viewer = MusePowerViewer()
        
    def run(self):
        """アプリケーション実行"""
        self.viewer.show()
        
        # qasyncイベントループ
        loop = qasync.QEventLoop(self.app)
        asyncio.set_event_loop(loop)
        
        with loop:
            loop.run_forever()

def main():
    print("Muse EEG Power Viewer")
    print("=" * 40)
    print("Real-time θ/α/β band power analysis")
    print("θ (Theta): 4-8Hz   - Deep relaxation, meditation")
    print("α (Alpha): 8-13Hz  - Relaxed awareness, creativity")
    print("β (Beta):  13-30Hz - Active thinking, concentration")
    print()
    print("Power is calculated every 0.5 seconds using 2-second windows")
    print("1. Scan for Muse devices")
    print("2. Connect to your device") 
    print("3. Start streaming to view real-time power analysis")
    print()
    
    app = MusePowerApp()
    app.run()

if __name__ == "__main__":
    main()