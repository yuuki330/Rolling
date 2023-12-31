## 必要なモジュールのインポート
import pyrtools as pt
import cv2
from scipy import signal
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
import pandas as pd
from time import time
from scipy.io.wavfile import write
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
# import librosa
import resampy
# import scipy.stats

class VM_rolling:
    ## コンストラクタの定義
    def __init__(self, video_path):
        self.video_object = cv2.VideoCapture(video_path) # ビデオオブジェクトを作成
        self.nframes = int(self.video_object.get(cv2.CAP_PROP_FRAME_COUNT)) # 動画のフレーム数を取得
        self.height = int(self.video_object.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 動画の高さを取得
        self.width = int(self.video_object.get(cv2.CAP_PROP_FRAME_WIDTH)) # 動画の幅を取得
        self.fps = int(self.video_object.get(cv2.CAP_PROP_FPS)) # 動画のfpsを取得
        ## 動画の情報が取得できなかった場合はエラーを出力
        if self.width == 0 or self.height == 0 or self.fps == 0: 
            raise Exception("Invalid video") 
        ## 動画の情報を出力
        print(self.width, self.height, self.fps) 
    
    ## 信号の畳み込み合成結果が最も大きくなるようにシフトして合成する関数
    def align(self, x, y):
        ## 信号xと信号yの畳み込みを計算
        tshift = np.argmax(signal.fftconvolve(x, np.flip(y))) + 1
        ## シフト幅を計算
        shift = y.size - tshift
        ## 元信号をシフトして返す
        return np.roll(x,shift)
    
    def plot_spectrogram(self, x, name, fs=2000):
        # 入力波形をリサンプリング
        y = resampy.resample(x, sr_orig=61920, sr_new=2000)
        plt.figure()
        plt.specgram(y, Fs=fs, cmap=plt.get_cmap('jet'))
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        # plt.ylim(0, 1100)
        plt.colorbar().set_label('PSD(dB)')
        plt.savefig('./result/spectrogram_'+name)
        plt.close()

    def get_scaled_sound(self, sound):
        maxs = np.max(sound)
        mins = np.min(sound)
        if maxs!=1.0 or mins!=-1.0:
            rangev = maxs - mins
            sound = 2 * sound / rangev
            newmax = np.max(sound)
            offset = newmax - 1.0
            sound -= offset
        return sound
    
    def save_audio(self, file_name, x, sr):
        y = resampy.resample(x, sr_orig=61920, sr_new=2000)
        if sr==0:
            raise Exception("Invalid sampling rate")
        write(file_name, sr, y)
    
    def sound_from_video(self, nscale, norientation, downsample_factor):
        ## Parameters
        nframes = self.nframes
        video = self.video_object
        first_coeff = dict()
        recovered_signal = dict()
        N_gap = 312
        Num_signal = (self.height+N_gap) * self.nframes

        ## 結果用のパラメータ
        stop_iter = 20
        start_idx = 0
        end_idx = (stop_iter-1)*(self.height+N_gap) # -1するのは初期フレームと差分を取るところを除外するため

        ## 時間とiterの初期化
        iter = 0
        time_start = time()
        time_present = time()
        print(f'Steerable Pyramid 計算中...')

        ## 1フレーム目の読み出し
        success, frame = video.read()

        ## ダウンサンプリング処理(画像サイズを小さくする)
        if downsample_factor < 1:
            scaled_frame = cv2.resize(frame, (0, 0), fx=downsample_factor, fy=downsample_factor)
        else:
            scaled_frame = frame
        
        ## gray画像にし、正規化する
        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        norm_frame = cv2.normalize(gray_frame.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        ## Steerable Pyramidの計算
        first_pyr = pt.pyramids.SteerablePyramidFreq(norm_frame, nscale, norientation-1, is_complex=True)
        first_pyr = first_pyr.pyr_coeffs

        for band, coefficient in first_pyr.items():
            if band == 'residual_lowpass':
                continue
            if band == 'residual_highpass':
                continue
            first_coeff[band] = coefficient
            recovered_signal[band] = [0] * Num_signal
        
        ## 2フレーム目以降の処理
        while success:
            if downsample_factor < 1:
                frame = cv2.resize(frame, (0, 0), fx=downsample_factor, fy=downsample_factor)
            else:
                scaled_frame = frame
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            norm_frame = cv2.normalize(gray_frame.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            pyramid = pt.pyramids.SteerablePyramidFreq(norm_frame, nscale, norientation-1, is_complex=True)
            pyramid = pyramid.pyr_coeffs

            amp_pyr = dict()
            phase_diff_pyr = dict()

            for band, coefficient in pyramid.items():
                if band == 'residual_lowpass':
                    continue
                if band == 'residual_highpass':
                    continue
                amp_pyr[band] = np.abs(coefficient)
            
            ## 初期フレームとの位相差を計算
            for band, coefficient in pyramid.items():
                if band == 'residual_lowpass':
                    continue
                if band == 'residual_highpass':
                    continue
                phase_diff_pyr[band] = np.mod(math.pi + np.angle(coefficient)-np.angle(first_coeff[band]),2*math.pi) - math.pi

            for band in recovered_signal:
                amp = amp_pyr[band]
                phase_diff = phase_diff_pyr[band]

                idx_offset = 0

                for y in range(len(amp[:, 0])):
                    lms = np.multiply(phase_diff[:, y], np.multiply(amp[:, y], amp[:, y]))
                    amp_net = np.mean(lms) / np.sum(amp[:, y])

                    repeat_count = 2 ** band[0]

                    for i in range(1, repeat_count+1):
                        index = y + 1032 * (iter - 2) + idx_offset + (i - 1)
                        recovered_signal[band][index] = amp_net
                    idx_offset += repeat_count-1

            ## 次のフレームの読み込み
            success, frame = video.read()
            time_present = time()
            ## 進捗の表示
            if(iter%50==0):
                print(f'進捗: {iter}/{nframes}, 経過時間: {time_present-time_start:.2f}秒')
            if iter == stop_iter:
                break
            iter += 1

        ## グラフを表示
        df = pd.DataFrame()
        for band in recovered_signal:
            plt.figure(figsize=[20, 5])
            plt.plot(range(start_idx, end_idx), recovered_signal[band][start_idx:end_idx])
            plt.savefig('./result/補間前_'+str(band))
            plt.close()
            df[str(band)] = recovered_signal[band]
        df.to_csv('補間前.csv')
        
        jmax = 10
        df = pd.DataFrame()
        for j in range(jmax):
            ## 繰り返し一括補完
            for band in recovered_signal:
                recovered_signal[band] = recovered_signal[band][start_idx:end_idx]
                forward_pred = []
                ar_model = AutoReg(recovered_signal[band], lags=700)
                ar_model = ar_model.fit()
                for i in range(0, len(recovered_signal[band]), self.height+N_gap):
                    if stop_iter == i//(self.height+N_gap):
                        break
                    forward_pred.append(ar_model.predict(start=(self.height+i)+1, end=i+self.height+N_gap))
                    recovered_signal[band][(i+self.height):i+self.height+N_gap] = forward_pred[(i//(self.height+N_gap))]

                    ## predの最終値と次のフレームの最初の値を揃えるために、以降の値にpredの最終値と次のフレームの差分を足す
                    if len(recovered_signal[band]) > i+self.height+N_gap+1:
                        diff = recovered_signal[band][i+self.height+N_gap+1] - recovered_signal[band][i+self.height+N_gap]
                        recovered_signal[band][i+self.height+N_gap:] += diff
                        # print(recovered_signal[band][i+self.height+N_gap])
                        # print(recovered_signal[band][i+self.height+N_gap+1])
                        # print()

                    if(i%100==0):
                        time_present = time()
                        print(f'{j}_{band}_進捗: {i}/{len(recovered_signal[band])}, 経過時間: {time_present-time_start:.2f}秒')
                if j == jmax-1:
                    df[str(band)] = recovered_signal[band]

            ## グラフとcsvを保存
            if((j+1)%10==0):
                for band in recovered_signal:
                    plt.figure(figsize=[20, 5])
                    plt.plot(range(start_idx, end_idx), recovered_signal[band][start_idx:end_idx])
                    # plt.plot(range(len(recovered_signal[band])), recovered_signal[band])
                    plt.savefig(f'./result/補間後_{j+1}'+str(band))
                    plt.close()
        print('補間処理完了')
        df.to_csv('補間後.csv')
    
        ## 復元音声変数の初期化
        recov_sound = np.zeros(len(recovered_signal[(0,0)]))
        j = 0
        df = pd.DataFrame()
        for band in recovered_signal:
            recov_sound += self.align(np.array(recovered_signal[band]), np.array(recov_sound))
            plt.figure(figsize=[20, 5])
            plt.plot(range(start_idx, end_idx), recov_sound[start_idx:end_idx])
            plt.savefig('./result/合成後_'+str(band))
            plt.close()
            df[str(band)] = recov_sound

            ## 復元音声のスペクトログラムを保存
            self.plot_spectrogram(recov_sound, str(band))
        df.to_csv('合成後.csv')

        ## 特定の位相方向のみを合成
        # for band in recovered_signal:
        #     if j == 3:
        #         recov_sound += self.align(np.array(recovered_signal[band]), np.array(recov_sound))
        #     j += 1
            ## グラフを保存
            # plt.figure(figsize=[20, 5])
            # plt.plot(range(start_idx, end_idx), self.align(np.array(recovered_signal[band]), np.array(recov_sound))[start_idx:end_idx])
            # # plt.plot(range(len(recov_sound)), recov_sound)
            # plt.savefig('./result/align後_'+str(band))
            # plt.close()
            # plt.figure(figsize=[20, 5])
            # plt.plot(range(start_idx, end_idx), recov_sound[start_idx:end_idx])
            # plt.plot(range(len(recov_sound)), recov_sound)
            # plt.savefig('./result/合成後_'+str(band))
            # plt.close()
            # self.plot_spectrogram(recov_sound, str(band))
    
        return recov_sound


video_path = './data/KitKat-60Hz-RollingShutter-Mary_MIDI-input.avi'
video = VM_rolling(video_path)
# x = video.sound_from_video(1, 1, 1)
x = video.sound_from_video(1, 1, 1)
# video.save_audio('test.wav', x, 2000)