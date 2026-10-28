from indicateurs import Indicateurs 
from portefeuille_1 import ParamsPF,construire_panels
from bootstrap import ParamsBS
from features import features_completes
from data_prep import load_macro ,prepare_dataset
from windows import make_loaders
from train import TrainConfig,train,predict
from resultats import rapport 
from decision import rendements_journaliers, choisir_seuils,backtest,rapport_backtest,decisions_du_jour
from Portefeuille import choisir_seuils_portefeuille,backtest_portefeuille,rapport_portefeuille,decisions_portefeuille_du_jour
from prevision import sauver_preparation,figer_seance

import pandas as pd
import yfinance as yf


if __name__=="__main__":
    TICKERS = ["XLK"
]

    DATE='2005-01-01'

    score=load_macro('score_updated.csv',date_col='periode_M')
    score=score['score_niveau']

    brut = yf.download(
        tickers=TICKERS,
        start=DATE,
        group_by='Ticker',
        auto_adjust=True,
        progress=False,
    )

    P = ParamsBS(
        capital=1_000_000,
        n_long=40,
        stop_atr=3,
        trail_atr=0,            # 5.0 pour un trailing chandelier
        seuil_entree=0.3,
        seuil_sortie=0.0,
        sensibilite=0.0,           # sortie asymétrique. 0 = désactivée
        cost_bps=10.0,
        min_titres=1,
    )

    prices = {}

    for ticker in TICKERS:
        if ticker not in brut.columns.get_level_values("Ticker"):
                print(f"{ticker} absent du téléchargement")
                continue
        df = brut[ticker].copy()

        # On s'assure que les colonnes sont simples
        df.columns.name = None

        # Suppression des dates entièrement vides pour ce ticker
        df = df.dropna(
            how="all",
            subset=["Open", "High", "Low", "Close", "Volume"]
        )

        if df.empty:
            print(f"{ticker} ne contient aucune donnée")
            continue

        prices[ticker] = df

    cours_now={"XLK":182.5}
    prices_jour=figer_seance(prices,cours_now)

    panels = construire_panels(
        prices_jour,
        Indicateurs,
        ParamsPF(
            min_titres=P.min_titres
        ),
        generateur=features_completes,
    )
    OPTIONS=dict(
     couts_bps=5,
     entree='cloture',
     stop_atr=2,
     atr=panels['atr'],
     max_positions=10,
     max_entrees_jour=2,
     duree_max=30)
    
    panels={'rsi':panels['rsi'],
        'bb_pct':panels['bollinger_pct_b'],
        'rev_5':panels['rev_5'],
        'stoch':panels['stoch'],
        'er':panels['er'],
        'hist':panels['hist'],
        'score_niveau': score
        }
    
    reglages=dict(train_start='2005-01-01',train_end='2016-12-31',val_end='2019-12-31',target='direction',
              signaux=True)


    indicateurs=pd.concat(panels,axis=1,names=['Indicator','Value'])

    rets=rendements_journaliers(prices)


    final_df=prepare_dataset(prices_jour,indicateurs,score,horizon=22,
                             indicator_field_level=0,**reglages)

    loaders=make_loaders(final_df,window=20,batch_size=256)
    cfg=TrainConfig(task="proba",d_model=16,dropout=0.2,patience=8)
    model,history=train(loaders,cfg,save_path="modele_proba.pt")
    
    tables=rapport("modele_proba.pt",loaders,split='val',out_dir="resultats_proba",data=final_df)

    grille=choisir_seuils_portefeuille(predict(model,loaders['val']),rets,horizon=22,**OPTIONS)
    print(grille.head(10))
    seuil_achat=float(grille.iloc[0]['seuil_achat'])
    seuil_sortie=None if grille.iloc[0]['seuil_sortie']=="aucun (temps seul)" else float(grille.iloc[0]['seuil_sortie'])
    print(seuil_achat, seuil_sortie)

    bt=backtest_portefeuille(predict(model,loaders['test']),prices,seuil_achat=seuil_achat,seuil_sortie=seuil_sortie,horizon=22,couts_bps=5,max_positions=2)
    print(rapport_portefeuille(bt,out_dir='backtest_etf_tech'))

    mon_portefeuille= {}
    dec=decisions_portefeuille_du_jour(model,final_df,mon_portefeuille,seuil_achat,seuil_sortie,prices=prices_jour,**OPTIONS)

    print("A vendre :\n", dec['vendre'].round(3).to_string())
    print('a garder:\n',dec['garder'].round(3).to_string())
    print('a acheter:\n',dec["acheter"].round(3).to_string())
