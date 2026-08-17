import pandas as pd
import numpy as np
import numpy.random as rd
import yfinance as yf
from datetime import date,datetime,timedelta

class indicateurs:

    def __init__(self,
                df:pd.DataFrame,
                seed:int,
                burnin:bool=True,
                ):
        self.seed=seed
        self.df=df
        self.price=df["Close"].astype(float)
        self._cache={}
        self.burnin=burnin

    def ema(self,n:int):
        key=("ema",n)
        if key not in self._cache:
            out=self._ema(self.price,n)
            if self.burnin:
                out.iloc[:n*5]=np.nan
            self._cache[key]=out.rename(f"ema{n}")
        return self._cache[key]
    def sma(self,n:int):
        key=("sma",n)
        if key not in self._cache:
            self._cache[key]=out.rename(f"ema{n}")
        return self._cache[key]

    @staticmethod

    def _ema(s:pd.Series,
             n:int,
             ):
        alpha=2.0/(n+1)
        vals=s.to_numpy()
        out=np.full(len(vals),np.nan)
        ema=None
        for i,p in enumerate(vals):
            if np.isnan(p):
                out[i]=ema if ema is not None else np.nan
                continue
            ema=p if ema is None else ema+alpha*(p-ema)
            out[i]=ema
        return pd.Series(out,index=s.index)

    def TR(m=14):
        TR=[1]
        for i in range(m):
                TR.append(max(self.p_h[len(self.p_h)-14+i]-self.p_b[len(self.p_b)-14+i],
                        abs(self.p_h[len(self.p_h)-14+i]-self.cloture[len(self.cloture)-14+i]),
                        abs(self.p_b[len(self.p_b)-14+i]-self.cloture[len(self.cloture)-14+i])))
        return TR

    def ATR(m=14):
        ATR=[]
        ATR.append(TR.mean())
        ATR.append(np.mean(self.TR()))
        for i in range(1,m+1):
            ATR.append((ATR[i-1]*13+self.TR([i]))/14)
        return ATR

    def ROC(n:int):
        return ((float(self.df.loc[date.today().strftime("%Y-%m-%d"),"Close"])-float(self.df.loc[(date.today()-timedelta(days=14)).strftime("%Y-%m-%d"),"Close"]))*100/(float(self.df.loc[(date.today()-timedelta(days=14)).strftime("%Y-%m-%d")])))
    def ROVL(n=50):
        return float(df.loc[date.today().strftime("%Y-%m-%d"),"Volume"])/float(df["Volume"].tail(n).mean())

    def VWAP(ancrage:str):   # acrage = "Y-m-d"
        vwap=0
        volume=0
        seuil=df["Volume"].loc[ancrage:].count()
        for i in range(1,seuil+1):
            vwap=(df["High"].iloc[-i]+df["Low"].iloc[-i]+df["Close"].iloc[-i])*df["Volume"].iloc[-i]/3
            volume+=df["Volume"].iloc[-i]
        return vwap/volume

    def adx(n=14):
        mvt_h=[]
        mvt_b=[]
        dmp=[]
        dmn=[]
        ADX=[]
        TR14=[]
        DM14_p=[]
        DM14_n=[]
        for i in range(n):
            mvt_h.append(df["High"].iloc[-1-i]-df["High"].iloc[-2-i])
            mvt_b.append(df["Low"].iloc[-1-i]-df["Low"].iloc[-2-i])
            dmp.append(max(0,mvt_h[i]-mvt_b[i]))
            dmn.append(max(0,mvt_b[i]-mvt_h[i]))
        TR14.append(sum(self.TR()))
        DM14_p.append(sum(mvt_h))
        DM14_n.append(sum(dm_b))              
        for i in range(1,n+1):
            TR14.append(TR14[i-1]-(TR14[i-1]/14)+self.TR()[i])
            DM14_p.append(DM14_p[i-1]-(DM14_p[i-1]/14)+dmp[i])
            DM14_n.append(DM14_n[i-1]-(DM14_n[i-1]/14)+dmn[i])
        DI14_p=100*(DM14_p/TR14)
        DI14_n=100*(DM14_n/TR14)
        DX=100*abs(DI14_p-DI14_n)/(DI14_n+DI14_p)
        ADX.append(np.mean(DX))
        for i in range(1,n):
            ADX.append((ADX[i-1]*13+DX[i])/14)
        return ADX








    
        


            


                
        
