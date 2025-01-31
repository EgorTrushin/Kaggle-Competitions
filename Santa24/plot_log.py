#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('log.csv')

fig, axs = plt.subplots(1, 2, figsize=(12,4))

axs[0].plot(data["score"], color='dodgerblue', label='Perplexity')
axs[0].axhline(y = data["score"].min(), color='black', linestyle='--', label=str(round(data['score'].min(),3)))
axs[1].plot(data["temperature"], color='orangered', label='Temperature')

axs[0].legend()
axs[1].legend()

plt.show()
