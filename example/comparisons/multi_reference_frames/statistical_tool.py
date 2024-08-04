from scipy import stats
import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import kruskal
import scikit_posthocs as sp
import pandas as pd
import heapq

def statistical_test(data_frame):
    lowest_column = None
    lowest_significance = 0.05#float('inf')
    for column1 in data_frame.columns:
        for column2 in data_frame.columns:
            if column1 != column2:
                # Perform statistical test
                # print(data_frame[column2])
                c_1=np.array(data_frame[column1])
                c_2=np.array(data_frame[column2])
                c_1=c_1[~np.isnan(c_1)]
                c_2=c_2[~np.isnan(c_2)]
                # t_statistic, p_value = stats.ttest_ind(c_1, c_2, equal_var=False)
                statistic, p_value = mannwhitneyu(c_1, c_2, alternative='less')
                print("P-Value:")
                print(p_value)
                # print(p_value)
                # Check if current column has lower value and lower significance
                # print("Statistical Test:")
                # print(statistic)
        if  p_value < lowest_significance: #and statistic < 0 :
            lowest_column = column1
            lowest_significance = p_value

    # Print the lowest column and its statistical significance
    print("Best Score:", lowest_column)
    print("Significance:", lowest_significance)

    return lowest_column, lowest_significance

def generate_ranking(df):
    groups = df.columns.tolist()
    rankings = pd.DataFrame({'group': groups, 'rank': len(groups)})

    for i in range(len(groups)):
        for j in range(len(groups)):
            group1 = groups[i]
            group2 = groups[j]
            samples1 = df[group1].dropna()
            samples2 = df[group2].dropna()
            #t_statistic, p_value=ttest_ind(samples1, samples2, alternative='less') 
            p_value = stats.mannwhitneyu(samples1, samples2, alternative='less')[1]  # Perform one-sided Mann-Whitney U test for lower distribution

            if p_value < 0.05:
                rankings.loc[rankings['group'] == group1, 'rank'] -= 1

    min_val=heapq.nsmallest(len(groups), set(rankings['rank']))
    for i in range(len(min_val)):
        rankings.loc[rankings['rank'] == min_val[i], 'rank'] = i+1
    rank_sort=rankings.sort_values('rank')    
    return rank_sort
