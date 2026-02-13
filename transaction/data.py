# %%

import pandas as pd 
df = pd.read_csv("./input/20250530.csv.gz", names=[
    "체결번호",
    "보드ID",
    "세션ID",
    "시장ID",
    "정규시간외구분코드",
    "대량매매구분코드",
    "체결가격",
    "체결수량",
    "체결유형코드",
    "체결시각",
    "시장구분코드",
    "종가",
    "누적거래대금",
    "매도1단계우선호가가격",
    "매도1단계우선호가잔량",
    "매수1단계우선호가가격",
    "매수1단계우선호가잔량",
    "누적매도미결제약정수량",
    "누적매수미결제약정수량",
    "매도호가가격",
    "매도국가코드",
    "매도투자자구분코드",
    "매도위탁자기구분코드",
    "매도외국인투자자구분코드",
    "매수호가가격",
    "매수국가코드",
    "매수투자자구분코드",
    "매수위탁자기구분코드",
    "매수외국인투자자구분코드",
    "매수회원사주문시각",
    "매도회원사주문시각",
    "매도호가체결수량",
    "매수호가체결수량",
    "매도PT구분코드",
    "매수PT구분코드",])
df.index.set_names(["체결일자","종목코드"], inplace=True)
df.head()
datestr = "20250530"
df = df.loc[int(datestr)].loc["KR4101W60000"].query("보드ID == 'G1' and 세션ID == 40")

# %%

datestr + ("0" + df["체결시각"].astype(str))[-9:]
# %%

df["time"] = pd.to_datetime(datestr + df["체결시각"].astype(str).str.rjust(9, "0"), format="%Y%m%d%H%M%S%f")

def chg_10s(df, colname):
    df_time = df.set_index("time")
    assert df_time.index.is_monotonic_increasing
    v = (df_time[colname] - df_time[colname].rolling("10s").mean()).values
    return pd.Series(v, index=df.index)


# Features
df["buy_dt"] = pd.to_datetime(df["매수회원사주문시각"].astype(str).str.rjust(9, "0"), format="%H%M%S%f").diff().dt.total_seconds()
df["sell_dt"] = pd.to_datetime(df["매도회원사주문시각"].astype(str).str.rjust(9, "0"), format="%H%M%S%f").diff().dt.total_seconds()
df["sell_amt"] = df['매도호가체결수량']
df["buy_amt"] = df['매수호가체결수량']
df["dt"] = df["time"].diff().dt.total_seconds()
df["prc_chg_10s"] = chg_10s(df,"체결가격")
df["prc_chg"] = df["체결가격"].diff()
df["trd_krw"] = df["누적거래대금"].diff()
df["bid_chg_10s"] = chg_10s(df,"매수호가가격")
df["ask_chg_10s"] = chg_10s(df,"매도호가가격")
df["bid_amt"] = df["매수1단계우선호가잔량"]
df["ask_amt"] = df["매도1단계우선호가잔량"]
df["open_short_count_chg_10s"] = chg_10s(df,"누적매도미결제약정수량")
df["open_long_count_chg_10s"] = chg_10s(df,"누적매수미결제약정수량")

# Labels
df["buy_country"] = df['매수국가코드']
df["buy_investor"] = df['매수투자자구분코드']
df["buy_broker"] = df['매수위탁자기구분코드']
df["buy_foreign"] = df['매수외국인투자자구분코드']
df["sell_country"] = df['매도국가코드']
df["sell_investor"] = df['매도투자자구분코드']
df["sell_broker"] = df['매도위탁자기구분코드']
df["sell_foreign"] = df['매도외국인투자자구분코드']

# %%

features = ["buy_dt","sell_dt","sell_amt","buy_amt","dt","prc_chg_10s","prc_chg","trd_krw","bid_chg_10s","ask_chg_10s","bid_amt","ask_amt","open_short_count_chg_10s","open_long_count_chg_10s"]
labels = ["buy_country","buy_investor","buy_broker","buy_foreign","sell_country","sell_investor","sell_broker","sell_foreign"]

# %%

df_ = df[features + labels].dropna(axis=0)
X = df_[features].values
# y = df_[labels].values
# y = df_["buy_investor"].replace({7100:0,1000:1,8000:2,5000:3,3000:4,4000:5,2000:6,7000:7,6000:8}).values
y = df_["buy_investor"].replace({7100:0,1000:0,8000:1,5000:0,3000:0,4000:0,2000:0,7000:0,6000:0}).values

# %%

