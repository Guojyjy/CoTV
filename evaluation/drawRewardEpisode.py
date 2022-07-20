import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def toFloat(list_):
	list = []
	for item in list_:
		item = float(item)
		list.append(item)
	return list

def read_file(file, agent_type, legend_tag):
	interation = []
	avg = []
	tag_this = []

	data = pd.read_csv(file)
	index = -1
	tt = toFloat(list(data['training_iteration']))
	index = len(tt)
	interation += tt * 3

	avg += toFloat(list(data['policy_reward_mean/'+agent_type]))
	avg_values = np.array(toFloat(list(data['policy_reward_mean/'+agent_type])))
	std_values = np.array(toFloat(list(data['policy_reward_std/'+agent_type])))  # add policy_reward_std in python3.7/site-packages/ray/rllib/evaluation/metrics.py
	avg += list(avg_values + std_values)
	avg += list(avg_values - std_values)
	tag_this += [legend_tag]*len(avg_values)*3
	return interation, avg, tag_this

def draw_plot(interation_num, avg_reward, tagg):
	tag_convert = []
	for each in tagg:
	    if each == 'tl':
	        tag_convert.append('TL')
	    else:
	        tag_convert.append('CAV')

	dataframe = pd.DataFrame({'Training iteration': interation_num, 'Average reward': avg_reward, ' ':tagg})
	print(dataframe)

	fig, ax = plt.subplots()
	sns.lineplot(x=dataframe["Training iteration"], y=dataframe["Average reward"], ax=ax , hue=dataframe[" "])
	sns.color_palette('bright')
	plt.legend(title='', loc='lower right', fontsize='13')
	plt.xticks(fontsize=13)
	plt.yticks(fontsize=13)
	plt.xlim(-5, 150)
	ax.set_xlabel("Training Iteration", fontsize=14)
	ax.set_ylabel("Average Episode Reward", fontsize=14)
	plt.grid()
	plt.show()


def CoTV_Dublin():
	TL_scenarios_list = ['Dublin_CoTV/PPO...']  # ray_results 
	TL_scenarios_legend = ['TL']

	CAV_scenarios_list = ['Dublin_CoTV/PPO...']
	CAV_scenarios_legend = ['CAV']

	return TL_scenarios_list, TL_scenarios_legend, CAV_scenarios_list, CAV_scenarios_legend


def main_CoTV():
	path = '~/ray_results/'
	prog = '/progress.csv'
	TL_scenarios_list, TL_scenarios_legend, CAV_scenarios_list, CAV_scenarios_legend = CoTV_Dublin()

	# average reward for agents
	interation_num = []
	avg_all = []
	tag_all = []

	for each in TL_scenarios_list:
		interation, avg, tagg = read_file(path+each+prog, 'tl', TL_scenarios_legend[TL_scenarios_list.index(each)])
		interation_num += interation
		avg_all += avg
		tag_all += tagg

	for each in CAV_scenarios_list:
		interation, avg, tagg = read_file(path+each+prog, 'cav', CAV_scenarios_legend[CAV_scenarios_list.index(each)])
		interation_num += interation
		avg_all += avg
		tag_all += tagg
	draw_plot(interation_num, avg_all, tag_all)

if __name__ == '__main__':
	main_CoTV()