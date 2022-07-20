import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib


def collect_data_random(path, exp_list, legend_pair, num_episode, tag, tag_list):
    
    travel_time = []
    exp = []

    for scen in exp_list:
        print(scen)
        file_list = []
        whole_path = os.path.join(path, scen)
        for episode_file in os.listdir(whole_path):
            if os.path.isfile(whole_path+"/"+episode_file):
                this_episode_path = os.path.join(whole_path+"/"+episode_file)
                if episode_file.endswith('tripinfo.xml'):
                    for trip in sumolib.output.parse(this_episode_path, ['tripinfo']):
                        travel_time.append(float(trip.duration))
                        exp.append(legend_pair[scen])
                        tag_list.append(tag)

    return exp, travel_time, tag_list  

def draw_box(exp, dataset, tag_list):
    dataframe = pd.DataFrame({'Method':exp,'Travel time (sec)':dataset,'Scenario':tag_list})
    ax = sns.boxplot(x=dataframe["Method"], y=dataframe["Travel time (sec)"], hue=dataframe["Scenario"])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_xlabel("Method", fontsize=18)
    ax.set_ylabel("Travel time (sec)", fontsize=18)
    # plt.setp(ax.get_legend().get_texts(), fontsize=17)
    plt.legend(fontsize='xx-large')
    plt.legend(title='', loc='best', fontsize='14')
    plt.show()

if __name__ == '__main__':

    num_episode = 8
    tag_list = []

    path = ''  # output directory 
    exp_list = [] # scenario name
    legend_list = ['FixedTime', 'FlowCAV', 'PressLight', 'GLOSA', 'CoTV']
    legend_pair = {exp_list[i]: legend_list[i] for i in range(0, len(exp_list))}
    exp1, dataset1, tag_list1 = collect_data_random(path, exp_list, legend_pair, num_episode, '1x1 grid', tag_list)

    path = ''  # output directory 
    exp_list = [] # scenario name
    legend_list = ['FixedTime', 'FlowCAV', 'PressLight', 'GLOSA', 'CoTV']
    legend_pair = {exp_list[i]: legend_list[i] for i in range(0, len(exp_list))}

    tag_list = []
    exp2, dataset2, tag_list2 = collect_data_random(path, exp_list, legend_pair, num_episode, '1x6 grid', tag_list)

    path = ''  # output directory 
    exp_list = [] # scenario name
    legend_list = ['FixedTime', 'FlowCAV', 'PressLight', 'GLOSA', 'CoTV']
    legend_pair = {exp_list[i]: legend_list[i] for i in range(0, len(exp_list))}

    tag_list = []
    exp3, dataset3, tag_list3 = collect_data_random(path, exp_list, legend_pair, num_episode, 'Dublin', tag_list)
    draw_box(exp1+exp2+exp3, dataset1+dataset2+dataset3, tag_list1+tag_list2+tag_list3)

