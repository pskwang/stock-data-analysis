import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. MySQL 연결
# ============================================================

engine = create_engine(
    "mysql+mysqlconnector://root@localhost:3306/stock_db",
    connect_args={"password": "your_password"}
)

# ============================================================
# 2. 주식 데이터 수집
# ============================================================

stocks = {
    'samsung': '005930.KS',
    'kakao':   '035720.KS',
    'skhynix': '000660.KS',
    'naver':   '035420.KS',
}

name_kor = {
    'samsung': '삼성전자',
    'kakao': '카카오',
    'skhynix': 'SK하이닉스',
    'naver': 'NAVER'
}

print("📥 주식 데이터 수집 중...")

df_all = {}
for name, ticker in stocks.items():
    df = yf.download(ticker, start='2020-01-01', end='2024-12-31', progress=False)
    df_all[name] = df
    print(f"  ✅ {name_kor[name]} ({len(df)}일치 데이터)")

# ============================================================
# 3. MySQL에 저장
# ============================================================

print("\n📤 MySQL에 저장 중...")
for name, df in df_all.items():
    df_save = df.copy()
    df_save.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_save.columns]
    df_save.to_sql(f'stock_{name}', con=engine, if_exists='replace', index=True)
    print(f"  ✅ {name_kor[name]} 저장 완료!")

# ============================================================
# 4. 주가 트렌드 시각화
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('국내 주요 종목 주가 트렌드 (2020~2024)', fontsize=16, fontweight='bold')
fig.patch.set_facecolor('white')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, (name, df) in enumerate(df_all.items()):
    ax = axes[idx // 2][idx % 2]
    close = df['Close']
    ax.plot(close.index, close.values, color=colors[idx], linewidth=1.5)
    ax.set_title(name_kor[name], fontsize=13, fontweight='bold')
    ax.set_xlabel('날짜')
    ax.set_ylabel('주가 (원)')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('stock_trend.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 시각화 완료!")

# ============================================================
# 5. 머신러닝 주가 예측 (Random Forest)
# ============================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

def add_features(df):
    """특성 엔지니어링"""
    d = df.copy()
    close = d['Close'].squeeze()
    d['MA5']  = close.rolling(5).mean()
    d['MA20'] = close.rolling(20).mean()
    d['MA60'] = close.rolling(60).mean()
    d['STD20'] = close.rolling(20).std()
    d['Return'] = close.pct_change()
    d['Target'] = close.shift(-1)  # 다음날 종가 예측
    d.dropna(inplace=True)
    return d

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
fig2.suptitle('국내 주요 종목 주가 예측 (Random Forest)', fontsize=16, fontweight='bold')
fig2.patch.set_facecolor('white')

for idx, (name, df) in enumerate(df_all.items()):
    print(f"\n📊 {name_kor[name]} 예측 중...")

    # 특성 추가
    df_feat = add_features(df)

    features = ['Close', 'MA5', 'MA20', 'MA60', 'STD20', 'Return']
    X = df_feat[features].values
    y = df_feat['Target'].values.ravel()

    # 학습/테스트 분리 (80:20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 스케일링
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 모델 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 성능 평가
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"  MAE: {mae:,.0f}원  |  R²: {r2:.4f}")

    # 시각화
    ax = axes2[idx // 2][idx % 2]
    test_index = df_feat.index[split:]
    ax.plot(test_index, y_test, color='gray', linewidth=1.2, label='실제 주가', alpha=0.8)
    ax.plot(test_index, y_pred, color='red',  linewidth=1.2, label='예측 주가', alpha=0.8)
    ax.set_title(f"{name_kor[name]}  |  R²: {r2:.4f}", fontsize=12, fontweight='bold')
    ax.set_xlabel('날짜')
    ax.set_ylabel('주가 (원)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('stock_prediction.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 주가 예측 완료!")
# ============================================================
# 6. 내일 주가 예측
# ============================================================

print("\n📅 내일 주가 예측")
print("=" * 40)

for name, df in df_all.items():
    df_feat = add_features(df)

    features = ['Close', 'MA5', 'MA20', 'MA60', 'STD20', 'Return']
    X = df_feat[features].values
    y = df_feat['Target'].values.ravel()

    # 전체 데이터로 학습
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # 오늘 마지막 데이터로 내일 예측
    today = X[-1].reshape(1, -1)
    today_scaled = scaler.transform(today)
    tomorrow = model.predict(today_scaled)[0]
    today_close = df['Close'].values[-1][0]

    diff = tomorrow - today_close
    sign = "📈" if diff > 0 else "📉"

    print(f"{sign} {name_kor[name]}")
    print(f"   오늘 종가:  {today_close:,.0f}원")
    print(f"   내일 예측:  {tomorrow:,.0f}원")
    print(f"   예상 변동:  {diff:+,.0f}원")
    print()
