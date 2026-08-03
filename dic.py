from leadlad_ccf import leadlag.py


#crée un dico qui regroupe toutes les variables à tester comme predictrices. ATTENTION : fait pour GDP, modifier le in range(2,33) pour les autres#
def build_data():
    dic={}
    for i in range(2,33):
        df=dict_of_df[f'df{i}']
        df_b=df.columns.str.lower().str.replace(" ","_").tolist()
        for j in range (1,df.shape[1]+1):
            dic[df_b[j-1]]=df[df.columns[j-1]]
    return dic

#applique le prewhiten test de toutes les variables sur la variable testée#
def test_prewhiten(variable):
  for i in range(1,33):
    df=dict_of_df[f'df{i}']
    for j in range (1,df.shape[1]+1):
        r=leadlag_ccf(df[df.columns.tolist()[j-1]],variable)
        print(r.summary())
        print(r.plot())
        
    
