import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import resampy

# 波形信号の読み出し
df = pd.read_csv("result/繰り返し_下駄_1のみ_iter10/合成後.csv", index_col=0)

# 波形信号のリサンプリング
x = df.iloc[:, 0].values
y = resampy.resample(x, sr_orig=61920, sr_new=2000)

# リサンプル前の波形を保存
plt.figure(figsize=(10, 5))
# plt.plot(x)
plt.plot(range(3000), x[:3000])
plt.savefig("result/繰り返し_下駄_1のみ_iter10/original.png")

# リサンプル後の波形を保存
plt.figure(figsize=(10, 5))
# plt.plot(y)
plt.plot(range(200), y[:200])
plt.savefig("result/繰り返し_下駄_1のみ_iter10/resampled.png")