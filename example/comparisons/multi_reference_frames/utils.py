import numpy as np
import pandas as pd
def load_in_table(filename):
    data =np.load(filename, allow_pickle=True)
    results_df=list(data['results_df'])
    results_area=list(data['results_area']) 
    results_dtw=list(data['results_dtw'])
    results_fde=list(data['results_fde'])
    results_fda=list(data['results_fad'])
    name= list(data['name'])
    df = pd.DataFrame(results_df)
    df = df.set_axis(name, axis=0) 
    df = df.transpose()

    area = pd.DataFrame(results_area)
    area = area.set_axis(name, axis=0) 
    area = area.transpose()

    dtw = pd.DataFrame(results_dtw)
    dtw = dtw.set_axis(name, axis=0) 
    dtw = dtw.transpose()

    fde = pd.DataFrame(results_fde)
    fde = fde.set_axis(name, axis=0) 
    fde = fde.transpose()

    fda = pd.DataFrame(results_fda)
    fda = fda.set_axis(name, axis=0)
    fda = fda.transpose()
    return df, area, dtw, fde, fda


def load_in_table_ood(filename):
    data =np.load(filename, allow_pickle=True)
    results_fde=list(data['results_fde'])
    results_fda=list(data['results_fad'])
    name= list(data['name'])

    fde = pd.DataFrame(results_fde)
    fde = fde.set_axis(name, axis=0) 
    fde = fde.transpose()

    fda = pd.DataFrame(results_fda)
    fda = fda.set_axis(name, axis=0)
    fda = fda.transpose()
    return fde, fda