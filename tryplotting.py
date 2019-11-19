import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('UFC_stats.csv')

print(df.columns)

plt.plot(df.F1_Height, df.F1_Weight)
plt.show()
