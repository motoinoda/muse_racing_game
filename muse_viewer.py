#!/usr/bin/env python3
"""
Muse Fixed Real-time Viewer
muse-lslの実装に基づいて修正したバージョン
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
from scipy import signal

# Muse constants
MUSE_SAMPLING_EEG_RATE = 256
MUSE_GATT_ATTR_STREAM_TOGGLE = '273e0001-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_TP9 = '273e0003-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_AF7 = '273e0004-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_AF8 = '273e0005-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_TP10 = '273e0006-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_RIGHTAUX = '273e0007-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_TELEMETRY = '273e000b-4c4d-454d-96be-f03bac821358'
MUSE_GATT_ATTR_ACCELEROMETER = '273e000a-4c4d-454d-96be-f03bac821358'

class MuseFixedViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.client = None
        self.device_address = None
        self.is_streaming = False
        
        # EEGデータバッファ
        self.buffer_size = 1000
        self.eeg_data = {
            'TP9': deque(maxlen=self.buffer_size),
            'AF7': deque(maxlen=self.buffer_size),
            'AF8': deque(maxlen=self.buffer_size),
            'TP10': deque(maxlen=self.buffer_size)
        }
        
        # muse-lsl互換のデータ処理変数
        self.timestamps = np.full(5, np.nan)
        self.data = np.zeros((5, 12))
        self.last_tm = 0
        self.first_sample = True
        self.sample_index = 0
        self.reg_params = None
        self._P = 1e-4
        
        # ハンドルとUUIDのマッピング（muse-lsl方式）
        self.uuid_to_handle = {
            MUSE_GATT_ATTR_TP9: 32,      # 0x20
            MUSE_GATT_ATTR_AF7: 35,      # 0x23  
            MUSE_GATT_ATTR_AF8: 38,      # 0x26
            MUSE_GATT_ATTR_TP10: 41,     # 0x29
            MUSE_GATT_ATTR_RIGHTAUX: 44  # 0x2c
        }
        
        self.handle_to_channel = {
            32: 'TP9',
            35: 'AF7', 
            38: 'AF8',
            41: 'TP10',
            44: 'RIGHTAUX'
        }
        
        # UI色設定
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # 統計
        self.sample_count = 0
        self.start_time = None

        # バンドパスフィルタ設定
        self.filter_enabled = False
        self.lowcut = 1.0
        self.highcut = 50.0
        self.filter_order = 4
        self.sos = None
        self.zi = {}  # フィルタの初期状態を各チャンネルごとに保存

        # 接触品質データ（信号品質から推定）
        self.contact_quality = {
            'TP9': 'Good',
            'AF7': 'Good',
            'AF8': 'Good',
            'TP10': 'Good'
        }
        self.signal_std = {
            'TP9': deque(maxlen=256),  # 1秒分のデータ
            'AF7': deque(maxlen=256),
            'AF8': deque(maxlen=256),
            'TP10': deque(maxlen=256)
        }

        # テレメトリデータ
        self.battery_level = 0.0
        self.temperature = 0.0

        # UI初期化
        self.init_ui()
        
        # プロット更新タイマー
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
    
    def init_ui(self):
        """UI初期化"""
        self.setWindowTitle('Muse Fixed Real-time Viewer')
        self.setGeometry(100, 100, 1200, 800)
        
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

        # Filter control panel
        filter_layout = QtWidgets.QHBoxLayout()

        self.filter_checkbox = QtWidgets.QCheckBox('Enable Bandpass Filter')
        self.filter_checkbox.stateChanged.connect(self.toggle_filter)
        filter_layout.addWidget(self.filter_checkbox)

        filter_layout.addWidget(QtWidgets.QLabel('Low Cut (Hz):'))
        self.lowcut_spinbox = QtWidgets.QDoubleSpinBox()
        self.lowcut_spinbox.setRange(0.1, 100.0)
        self.lowcut_spinbox.setValue(1.0)
        self.lowcut_spinbox.setSingleStep(0.5)
        self.lowcut_spinbox.valueChanged.connect(self.update_filter_params)
        filter_layout.addWidget(self.lowcut_spinbox)

        filter_layout.addWidget(QtWidgets.QLabel('High Cut (Hz):'))
        self.highcut_spinbox = QtWidgets.QDoubleSpinBox()
        self.highcut_spinbox.setRange(1.0, 128.0)
        self.highcut_spinbox.setValue(50.0)
        self.highcut_spinbox.setSingleStep(1.0)
        self.highcut_spinbox.valueChanged.connect(self.update_filter_params)
        filter_layout.addWidget(self.highcut_spinbox)

        filter_layout.addWidget(QtWidgets.QLabel('Order:'))
        self.order_spinbox = QtWidgets.QSpinBox()
        self.order_spinbox.setRange(2, 8)
        self.order_spinbox.setValue(4)
        self.order_spinbox.valueChanged.connect(self.update_filter_params)
        filter_layout.addWidget(self.order_spinbox)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Contact quality panel
        contact_layout = QtWidgets.QHBoxLayout()
        contact_layout.addWidget(QtWidgets.QLabel('Contact Quality:'))

        self.contact_labels = {}
        channels = ['TP9', 'AF7', 'AF8', 'TP10']
        for channel in channels:
            label = QtWidgets.QLabel(f'{channel}: Good')
            label.setMinimumWidth(100)
            label.setStyleSheet('padding: 5px; background-color: #90EE90; border-radius: 3px;')
            self.contact_labels[channel] = label
            contact_layout.addWidget(label)

        contact_layout.addStretch()
        layout.addLayout(contact_layout)

        # Plot area
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)
        
        # Create plots for EEG channels
        self.plots = {}
        self.curves = {}
        channels = ['TP9', 'AF7', 'AF8', 'TP10']
        channel_names = ['Left Ear (TP9)', 'Left Forehead (AF7)', 
                        'Right Forehead (AF8)', 'Right Ear (TP10)']
        
        for i, (channel, name) in enumerate(zip(channels, channel_names)):
            plot = self.plot_widget.addPlot(title=name)
            plot.setLabel('left', 'Amplitude', units='μV')
            plot.setLabel('bottom', 'Samples')
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setYRange(-200, 200)
            
            curve = plot.plot(pen=pg.mkPen(color=self.colors[i], width=2))
            
            self.plots[channel] = plot
            self.curves[channel] = curve
            
            if i % 2 == 1:
                self.plot_widget.nextRow()
        
        # Statistics
        stats_layout = QtWidgets.QHBoxLayout()
        self.stats_label = QtWidgets.QLabel('Samples: 0 | Rate: 0 Hz')
        stats_layout.addWidget(self.stats_label)

        self.battery_label = QtWidgets.QLabel('Battery: --% | Temp: --°C')
        stats_layout.addWidget(self.battery_label)

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

    def toggle_filter(self, state):
        """フィルタのオン/オフ切り替え"""
        self.filter_enabled = (state == QtCore.Qt.Checked)
        if self.filter_enabled:
            self.design_filter()
            print(f"✅ Bandpass filter enabled: {self.lowcut}-{self.highcut} Hz, Order: {self.filter_order}")
        else:
            print("❌ Bandpass filter disabled")

    def update_filter_params(self):
        """フィルタパラメータ更新"""
        self.lowcut = self.lowcut_spinbox.value()
        self.highcut = self.highcut_spinbox.value()
        self.filter_order = self.order_spinbox.value()

        if self.filter_enabled:
            self.design_filter()
            print(f"Filter updated: {self.lowcut}-{self.highcut} Hz, Order: {self.filter_order}")

    def design_filter(self):
        """バンドパスフィルタを設計"""
        try:
            nyq = 0.5 * MUSE_SAMPLING_EEG_RATE
            low = self.lowcut / nyq
            high = self.highcut / nyq

            if low >= high:
                print("❌ Error: Low cut must be less than high cut")
                self.filter_enabled = False
                self.filter_checkbox.setChecked(False)
                return

            # Butterworthバンドパスフィルタを設計
            self.sos = signal.butter(self.filter_order, [low, high], btype='band', output='sos')

            # 各チャンネルのフィルタ初期状態を初期化
            for channel in ['TP9', 'AF7', 'AF8', 'TP10']:
                self.zi[channel] = signal.sosfilt_zi(self.sos)

            print(f"Filter designed: {self.lowcut}-{self.highcut} Hz")
        except Exception as e:
            print(f"❌ Filter design error: {e}")
            self.filter_enabled = False
            self.filter_checkbox.setChecked(False)

    def apply_filter(self, data, channel):
        """バンドパスフィルタを適用"""
        if not self.filter_enabled or self.sos is None:
            return data

        try:
            if channel not in self.zi:
                self.zi[channel] = signal.sosfilt_zi(self.sos)

            # フィルタを適用
            filtered_data, self.zi[channel] = signal.sosfilt(self.sos, data, zi=self.zi[channel])
            return filtered_data
        except Exception as e:
            print(f"❌ Filter application error: {e}")
            return data

    def _handle_telemetry(self, sender, data):
        """テレメトリデータハンドラー（バッテリー、温度など）"""
        try:
            if len(data) < 10:
                return

            # muse-jsのparseTelemetryに基づく解析
            # データはビッグエンディアン（>）の16ビット整数
            import struct

            sequence_id = struct.unpack('>H', data[0:2])[0]
            battery_raw = struct.unpack('>H', data[2:4])[0]
            fuel_gauge_raw = struct.unpack('>H', data[4:6])[0]
            temperature_raw = struct.unpack('>H', data[8:10])[0]

            # 変換
            self.battery_level = battery_raw / 512.0 * 100  # パーセント
            fuel_gauge_voltage = fuel_gauge_raw * 2.2  # ミリボルト
            self.temperature = temperature_raw  # 生の値（単位不明）

            # UI更新
            self.battery_label.setText(f'Battery: {self.battery_level:.1f}% | Temp: {self.temperature}')

            print(f"📊 Telemetry - Battery: {self.battery_level:.1f}%, Voltage: {fuel_gauge_voltage:.0f}mV, Temp: {self.temperature}")
        except Exception as e:
            print(f"❌ Telemetry handler error: {e}")

    def _evaluate_contact_quality(self):
        """信号の標準偏差から接触品質を評価"""
        channels = ['TP9', 'AF7', 'AF8', 'TP10']
        for channel in channels:
            if len(self.signal_std[channel]) >= 128:  # 0.5秒分のデータ
                std = np.std(list(self.signal_std[channel]))

                # 標準偏差に基づく評価（muse-lslの推奨値）
                if std < 20:
                    status_text = 'Good'
                    color = '#90EE90'  # 薄緑
                elif std < 50:
                    status_text = 'OK'
                    color = '#FFD700'  # 金色
                else:
                    status_text = 'Bad'
                    color = '#FF6B6B'  # 赤

                self.contact_quality[channel] = status_text

                # UI更新
                self.contact_labels[channel].setText(f'{channel}: {status_text}')
                self.contact_labels[channel].setStyleSheet(
                    f'padding: 5px; background-color: {color}; border-radius: 3px; font-weight: bold;'
                )

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
        """muse-lslのEEGデータアンパック処理"""
        aa = bitstring.Bits(bytes=packet)
        pattern = "uint:16,uint:12,uint:12,uint:12,uint:12,uint:12,uint:12, \
                   uint:12,uint:12,uint:12,uint:12,uint:12,uint:12"
        
        res = aa.unpack(pattern)
        packet_index = res[0]
        data = res[1:]
        # 12 bits on a 2 mVpp range
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
        """EEGデータハンドラー（muse-lsl方式）"""
        if self.first_sample:
            self._init_timestamp_correction()
            self.first_sample = False
        
        timestamp = time.time()
        
        # ハンドルを取得（muse-lsl方式）
        sender_uuid = str(sender.uuid)
        if sender_uuid not in self.uuid_to_handle:
            return
        
        handle = self.uuid_to_handle[sender_uuid]
        
        # samples are received in this order : 44, 41, 38, 32, 35
        # wait until we get 35 and call the data callback
        index = int((handle - 32) / 3)
        tm, d = self._unpack_eeg_channel(data)
        
        if self.last_tm == 0:
            self.last_tm = tm - 1
        
        self.data[index] = d
        self.timestamps[index] = timestamp
        
        print(f"Received EEG data from {self.handle_to_channel[handle]} (handle {handle}): {len(data)} bytes, tm={tm}")
        
        # 最後のデータ（handle == 35, AF7）を受信したらコールバック実行
        if handle == 35:
            if tm != self.last_tm + 1:
                if (tm - self.last_tm) != -65535:  # カウンターリセット
                    print(f"Missing sample {tm} : {self.last_tm}")
                    self.sample_index += 12 * (tm - self.last_tm + 1)
            
            self.last_tm = tm
            
            # タイムスタンプインデックス計算
            idxs = np.arange(0, 12) + self.sample_index
            self.sample_index += 12
            
            # タイムスタンプ補正更新
            self._update_timestamp_correction(idxs[-1], np.nanmin(self.timestamps))
            
            # タイムスタンプを外挿
            timestamps = self.reg_params[1] * idxs + self.reg_params[0]
            
            # データをバッファに追加（最初の4チャンネル）
            channels = ['TP9', 'AF7', 'AF8', 'TP10']
            for i, channel in enumerate(channels):
                if i < 4:
                    # フィルタを適用
                    samples = self.data[i]
                    if self.filter_enabled:
                        samples = self.apply_filter(samples, channel)

                    for sample in samples:
                        self.eeg_data[channel].append(sample)
                        # 接触品質評価用に標準偏差計算用データを保存
                        self.signal_std[channel].append(sample)
            
            self.sample_count += 12
            print(f"Processed complete EEG sample set, total samples: {self.sample_count}")
            
            # データをリセット
            self.timestamps = np.full(5, np.nan)
            self.data = np.zeros((5, 12))
    
    def _write_cmd(self, cmd):
        """コマンド書き込み（muse-lsl方式）"""
        async def write_async():
            await self.client.write_gatt_char(MUSE_GATT_ATTR_STREAM_TOGGLE, bytearray(cmd), response=False)
        return asyncio.create_task(write_async())
    
    def _write_cmd_str(self, cmd):
        """文字列コマンド書き込み（muse-lsl方式）"""
        cmd_bytes = [len(cmd) + 1, *(ord(char) for char in cmd), ord('\n')]
        return self._write_cmd(cmd_bytes)
    
    @qasync.asyncSlot()
    async def start_streaming(self):
        """ストリーミング開始（muse-lsl方式）"""
        if not self.client or not self.client.is_connected:
            self.status_label.setText('Status: Not connected')
            return
        
        try:
            self.status_label.setText('Status: Starting streaming...')
            
            print("Setting up EEG subscriptions...")

            # テレメトリ（バッテリー、温度）の通知を設定
            try:
                await self.client.start_notify(MUSE_GATT_ATTR_TELEMETRY, self._handle_telemetry)
                print(f"✅ Started notifications for Telemetry (battery, temperature)")
            except Exception as e:
                print(f"⚠️ Failed to start Telemetry notifications: {e}")

            # EEG特性に通知を設定（muse-lsl順序）
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
                    print(f"✅ Started notifications for {char_uuid}")
                except Exception as e:
                    print(f"❌ Failed to start notifications for {char_uuid}: {e}")
            
            print("Sending Muse commands...")
            
            # プリセット選択（muse-lsl方式）
            print("Setting preset p21...")
            preset_cmd = [0x04, 0x70, 0x32, 0x31, 0x0a]  # p21
            await self.client.write_gatt_char(MUSE_GATT_ATTR_STREAM_TOGGLE, bytearray(preset_cmd), response=False)
            await asyncio.sleep(1)
            
            # 初期化コマンド（muse-lsl muse.pyから）
            print("Sending start command 'd'...")
            await self._write_cmd_str('d')
            await asyncio.sleep(0.5)
            
            print("Sending resume command...")  
            await self._write_cmd_str('d')
            
            self.is_streaming = True
            self.start_time = time.time()
            self.sample_count = 0
            
            # UI更新
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
            # プロット更新開始
            self.timer.start(50)  # 20 FPS
            
            self.status_label.setText('Status: Streaming active')
            print("✅ EEG streaming started")
            
        except Exception as e:
            self.status_label.setText(f'Status: Streaming error - {str(e)}')
            print(f"❌ Streaming error: {e}")
            import traceback
            traceback.print_exc()
    
    @qasync.asyncSlot()
    async def stop_streaming(self):
        """ストリーミング停止"""
        if self.is_streaming:
            try:
                print("Stopping streaming...")
                
                # タイマー停止
                self.timer.stop()
                
                # ストリーミング停止コマンド（muse-lsl方式）
                if self.client and self.client.is_connected:
                    print("Sending stop command 'h'...")
                    await self._write_cmd_str('h')
                    await asyncio.sleep(0.5)

                    # テレメトリ通知停止
                    try:
                        await self.client.stop_notify(MUSE_GATT_ATTR_TELEMETRY)
                        print(f"Stopped notifications for Telemetry")
                    except Exception as e:
                        print(f"Error stopping Telemetry: {e}")

                    # EEG通知停止
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
                            print(f"Stopped notifications for {char_uuid}")
                        except Exception as e:
                            print(f"Error stopping {char_uuid}: {e}")
                
                self.is_streaming = False
                self.status_label.setText('Status: Streaming stopped')
                
                # UI更新
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                
                print("✅ Streaming stopped")
                
            except Exception as e:
                self.status_label.setText(f'Status: Stop error - {str(e)}')
                print(f"❌ Stop error: {e}")
    
    def update_plots(self):
        """プロット更新"""
        if not self.is_streaming:
            return

        # 統計更新
        if self.start_time:
            elapsed = time.time() - self.start_time
            rate = self.sample_count / elapsed if elapsed > 0 else 0
            self.stats_label.setText(f'Samples: {self.sample_count} | Rate: {rate:.1f} Hz')

        # 接触品質評価
        self._evaluate_contact_quality()

        # プロット更新
        for channel, curve in self.curves.items():
            data = np.array(self.eeg_data[channel])
            if len(data) > 0:
                x = np.arange(len(data))
                curve.setData(x, data)

                # Y軸の自動調整
                if len(data) > 50:
                    recent_data = data[-200:]
                    mean_val = np.mean(recent_data)
                    std_val = np.std(recent_data)
                    if std_val > 0:
                        y_range = max(50, 3 * std_val)
                        self.plots[channel].setYRange(mean_val - y_range, mean_val + y_range)
    
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

class MuseApp:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        
        # pyqtgraph設定
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        self.viewer = MuseFixedViewer()
        
    def run(self):
        """アプリケーション実行"""
        self.viewer.show()
        
        # qasyncイベントループ
        loop = qasync.QEventLoop(self.app)
        asyncio.set_event_loop(loop)
        
        with loop:
            loop.run_forever()

def main():
    print("Muse Fixed Real-time Viewer")
    print("=" * 40)
    print("Fixed version based on muse-lsl implementation")
    print("1. Scan for Muse devices")
    print("2. Connect to your device")
    print("3. Start streaming to view real-time EEG")
    print()
    
    app = MuseApp()
    app.run()

if __name__ == "__main__":
    main()