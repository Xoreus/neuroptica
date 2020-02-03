import pandas as pd

def list_to_df(phases):
    phases_flat = [item for sublist in phases for item in sublist]
    df = pd.DataFrame(phases_flat, columns=['Theta','Phi'])
    return df

def to_farhad_phases(df_phases, topo):
    if topo == 'diamond':
        df_far = pd.DataFrame([(df_phases.loc[:, "Theta"][0], df_phases.loc[:, "Phi"][0]),
                   (df_phases.loc[:,"Theta"][2], df_phases.loc[:, "Phi"][2]),
                   (df_phases.loc[:,"Theta"][4], df_phases.loc[:, "Phi"][4]),
                   (df_phases.loc[:,"Theta"][5], df_phases.loc[:, "Phi"][5]),
                   (df_phases.loc[:,"Theta"][7], df_phases.loc[:, "Phi"][7]),
                   (df_phases.loc[:,"Theta"][8], df_phases.loc[:, "Phi"][8]),
                   (df_phases.loc[:,"Theta"][1], df_phases.loc[:, "Phi"][1]),
                   (df_phases.loc[:,"Theta"][3], df_phases.loc[:, "Phi"][3]),
                   (df_phases.loc[:,"Theta"][6], df_phases.loc[:, "Phi"][6])], columns=["Thetas","Phis"])
    elif topo == 'reck':
        df_far = df_phases

    return df_far

def to_neuro_phases(farhad_phases, topo):
    if topo == 'diamond':
        phases = [farhad_phases[0], farhad_phases[6], farhad_phases[1], farhad_phases[7], farhad_phases[2], farhad_phases[3], farhad_phases[8], farhad_phases[4], farhad_phases[5]]
    elif topo == 'reck':
        phases = farhad_phases

    return phases

