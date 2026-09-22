import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ==============================================================================
# BƯỚC 1: BUSINESS UNDERSTANDING (Đã định nghĩa trong Báo cáo PDF)
# Target metric: RMSLE (Root Mean Squared Logarithmic Error)
# ==============================================================================

print("=== BƯỚC 2: DATA UNDERSTANDING & OUTLIER REMOVAL ===")
# Giả định file train.csv và test.csv nằm cùng thư mục
# Bạn có thể tải từ https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
try:
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print(f"Kích thước tập Train ban đầu: {train.shape}")
    print(f"Kích thước tập Test ban đầu: {test.shape}")
except FileNotFoundError:
    print("Không tìm thấy file dataset! Đang khởi tạo dữ liệu giả định để minh họa code...")
    # Tạo dữ liệu giả lập đúng cấu trúc nếu chạy không có file Kaggle local
    np.random.seed(42)
    train = pd.DataFrame({
        'Id': range(1, 1461),
        'MSSubClass': np.random.choice([20, 60, 70, 120], 1460),
        'MSZoning': np.random.choice(['RL', 'RM', 'FV'], 1460),
        'LotFrontage': np.random.normal(70, 20, 1460),
        'LotArea': np.random.normal(10000, 3000, 1460),
        'Neighborhood': np.random.choice(['NAmes', 'CollgCr', 'OldTown'], 1460),
        'OverallQual': np.random.randint(1, 10, 1460),
        'OverallCond': np.random.randint(1, 10, 1460),
        'YearBuilt': np.random.randint(1950, 2010, 1460),
        'YearRemodAdd': np.random.randint(1980, 2010, 1460),
        'TotalBsmtSF': np.random.normal(1000, 300, 1460),
        '1stFlrSF': np.random.normal(1000, 300, 1460),
        '2ndFlrSF': np.random.normal(500, 200, 1460),
        'GrLivArea': np.random.normal(1500, 500, 1460),
        'FullBath': np.random.randint(1, 3, 1460),
        'HalfBath': np.random.randint(0, 2, 1460),
        'BsmtFullBath': np.random.randint(0, 2, 1460),
        'BsmtHalfBath': np.random.randint(0, 1, 1460),
        'PoolArea': [0]*1450 + [500]*10,
        'YrSold': np.random.choice([2006, 2007, 2008, 2009, 2010], 1460),
        'SalePrice': np.random.normal(180000, 50000, 1460)
    })
    test = train.drop(columns=['SalePrice']).copy()
    test['Id'] = range(1461, 2920)

# 1. Loại bỏ Outliers theo nghiên cứu của Dean De Cock (2011)
# Tác giả khuyến nghị xóa các nhà có GrLivArea > 4000 sqft
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
print(f"Kích thước tập Train sau khi xóa Outlier: {train.shape}")

# Tách biến mục tiêu & Log-Transform (RMSLE Target)
train['SalePrice'] = np.log1p(train['SalePrice'])
y_train = train['SalePrice'].values
train_ids = train['Id']
test_ids = test['Id']

train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

# Gộp Train + Test để thực hiện Feature Engineering đồng bộ
all_data = pd.concat((train, test)).reset_index(drop=True)

# ==============================================================================
# BƯỚC 3: DATA PREPARATION (XỬ LÝ KHUYẾT, FEATURE ENGINEERING)
# ==============================================================================
print("\n=== BƯỚC 3: DATA PREPARATION & FEATURE ENGINEERING ===")

# 1. Điền giá trị khuyết (Imputation Rules)
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
for col in none_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

zero_cols = ['GarageArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
             'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath']
for col in zero_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(0)

# LotFrontage: Điền bằng Median theo từng Neighborhood
if 'LotFrontage' in all_data.columns and 'Neighborhood' in all_data.columns:
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Các biến phân loại còn lại: Điền Mode
mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
for col in mode_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# 2. Tạo đặc trưng mới (Feature Creation)
print("Tạo các đặc trưng tổng hợp...")
all_data['TotalSF'] = all_data.get('TotalBsmtSF', 0) + all_data.get('1stFlrSF', 0) + all_data.get('2ndFlrSF', 0)
all_data['TotalBath'] = all_data.get('FullBath', 0) + 0.5*all_data.get('HalfBath', 0) + all_data.get('BsmtFullBath', 0) + 0.5*all_data.get('BsmtHalfBath', 0)
all_data['HouseAge'] = all_data.get('YrSold', 2010) - all_data.get('YearBuilt', 1970)
all_data['RemodAge'] = all_data.get('YrSold', 2010) - all_data.get('YearRemodAdd', 1970)
all_data['IsNew'] = (all_data.get('YearBuilt', 0) == all_data.get('YrSold', 0)).astype(int)
all_data['HasPool'] = (all_data.get('PoolArea', 0) > 0).astype(int)

# 3. Ordinal Encoding cho các thuộc tính chất lượng
qual_dict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
for col in qual_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].map(qual_dict).fillna(0)

# 4. Giảm độ lệch (Skewness Transformation) cho biến số liên tục
numeric_feats = all_data.select_dtypes(include=[np.number]).columns
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.75]
print(f"Số lượng biến số bị lệch cao (>0.75): {len(high_skew)}")

for feat in high_skew.index:
    all_data[feat] = np.log1p(all_data[feat])

# 5. One-Hot Encoding cho các biến phân loại Nominal
all_data = pd.get_dummies(all_data)
print(f"Tổng số thuộc tính sau khi mã hóa One-Hot: {all_data.shape[1]}")

# Tách lại tập Train và Test
X_train = all_data.iloc[:len(y_train)].copy()
X_test = all_data.iloc[len(y_train):].copy()

# Standard Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# BƯỚC 4 & 5: MODELING & CROSS-VALIDATION EVALUATION
# ==========================================
print("\n--- BƯỚC 4 & 5: MODELING & CROSS-VALIDATION EVALUATION ---")

n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

def rmsle_cv(model, X, y):
    """Hàm tính trung bình RMSLE qua 10-fold CV chuẩn hóa"""
    # Xử lý triệt để giá trị NaN/Inf nếu có
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scores = cross_val_score(
        model, 
        X_clean, 
        y, 
        scoring="neg_root_mean_squared_error", 
        cv=kf
    )
    return -scores

# Khởi tạo các mô hình ứng viên
models = {
    'Ridge': RidgeCV(alphas=[10.0, 14.5, 15.0]),
    'Lasso': LassoCV(alphas=[0.0001, 0.0005, 0.001], max_iter=10000, random_state=42),
    'ElasticNet': ElasticNetCV(alphas=[0.0001, 0.0008, 0.001], l1_ratio=[0.8, 0.9], max_iter=10000, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=3, random_state=42, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=5, random_state=42, n_jobs=-1, verbose=-1),
    'CatBoost': CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=3, random_seed=42, verbose=0)
}

# Đánh giá hiệu năng từng mô hình
results = {}
print("\nĐang kiểm thử và tính điểm RMSLE trên 10-Fold CV:")
for name, model in models.items():
    score = rmsle_cv(model, X_train_scaled, y_train)
    results[name] = score.mean()
    print(f" -> Mô hình {name:10s} - RMSLE: {score.mean():.4f} (Std: {score.std():.4f})")

# ==========================================
# STACKING, HUẤN LUYỆN & XUẤT SUBMISSION
# ==========================================
print("\n--- HUẤN LUYỆN MÔ HÌNH HOÀN CHỈNH & DỰ ĐOÁN ---")

# Xử lý triệt để giá trị NaN/Inf nếu có
X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

predictions = []
weights = [0.15, 0.20, 0.15, 0.15, 0.15, 0.20] # Trọng số Blending

for (name, model), w in zip(models.items(), weights):
    model.fit(X_train_scaled, y_train)
    pred_log = model.predict(X_test_scaled)
    # Chuyển ngược log1p về giá trị thực ($ USD)
    predictions.append(np.expm1(pred_log) * w) 

# Tổng hợp dự đoán từ các mô hình
final_predictions = np.sum(predictions, axis=0)

# ==========================================
# BƯỚC 6: DEPLOYMENT (XUẤT SUBMISSION.CSV)
# ==========================================
print("\n--- BƯỚC 6: DEPLOYMENT & SUBMISSION FILE CREATION ---")

test_orig = pd.read_csv('test.csv')

submission = pd.DataFrame({
    'Id': test_orig['Id'],
    'SalePrice': final_predictions
})

submission.to_csv('submission.csv', index=False)
