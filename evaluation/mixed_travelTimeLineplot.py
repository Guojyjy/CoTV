import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib.ticker as ticker

def one_inter():
	tt = [50.18, 49.26, 49.07, 48.71, 48.34]
	pene_rate = ['0%', '25%', '50%', '75%', '100%']

	dataframe = pd.DataFrame({'Penetration rate': pene_rate, 'Travel time (s)': tt})
	ax = sns.lineplot(x='Penetration rate', y='Travel time (s)', data=dataframe, marker="s", ms=8, linewidth=4)

	ax.axhline(y=59.76, color='r', label='Baseline', linewidth=4)
	ax.axhline(y=51.38, color='g', label='PressLight', linewidth=4)
	ax.text(y=58.56, x=3.2, s='Baseline', fontsize=18)
	ax.text(y=51.88, x=3.05, s='PressLight', fontsize=18)
	ax.text(y=49.18, x=3.45, s='CoTV', fontsize=18)
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	ax.set_xlabel('Penetration rate', fontsize=20)
	ax.set_ylabel('Travel time (s)', fontsize=22)
	plt.show()

def six_inter():
	tt = [68.61, 65.34, 65.96, 66.42, 65.16]
	pene_rate = ['0%', '25%', '50%', '75%', '100%']

	dataframe = pd.DataFrame({'Penetration rate': pene_rate, 'Travel time (s)': tt})
	ax = sns.lineplot(x='Penetration rate', y='Travel time (s)', data=dataframe, marker="s", ms=8, linewidth=4)

	ax.axhline(y=92.80, color='r', label='Baseline', linewidth=4)
	ax.axhline(y=79.91, color='g', label='PressLight', linewidth=4)
	ax.text(y=89.83, x=3.2, s='Baseline', fontsize=18)
	ax.text(y=80.84, x=3.05, s='PressLight', fontsize=18)
	ax.text(y=69.18, x=3.45, s='CoTV', fontsize=18)
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	ax.set_xlabel('Penetration rate', fontsize=20)
	ax.set_ylabel('Travel time (s)', fontsize=22)
	plt.show()

def dublin():
	tt = [47.25, 43.91, 43.38, 44.07, 42.98]
	pene_rate = ['0%', '25%', '50%', '75%', '100%']

	dataframe = pd.DataFrame({'Penetration rate': pene_rate, 'Travel time (s)': tt})
	ax = sns.lineplot(x='Penetration rate', y='Travel time (s)', data=dataframe, marker="s", ms=8, linewidth=4)

	ax.axhline(y=60.86, color='r', label='Baseline', linewidth=4)
	ax.axhline(y=47.30, color='g', label='PressLight', linewidth=4)
	ax.text(y=59.06, x=3.2, s='Baseline', fontsize=18)
	ax.text(y=47.90, x=3.05, s='PressLight', fontsize=18)
	ax.text(y=44.18, x=3.45, s='CoTV', fontsize=18)
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	for axis in [ax.xaxis, ax.yaxis]:
		axis.set_major_locator(ticker.MaxNLocator(integer=True))
	ax.set_xlabel('Penetration rate', fontsize=20)
	ax.set_ylabel('Travel time (s)', fontsize=22)
	plt.show()


if __name__ == "__main__":
	one_inter()
	six_inter()
	dublin()
