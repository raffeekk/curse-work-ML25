import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')  # Исправляем устаревший стиль
sns.set_style('whitegrid')

# Функции визуализации (перемещаем в начало)
def save_plot(fig, filename):
    plt.tight_layout()
    plt.savefig(f'figures/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--', lw=2)
    plt.xlabel('Реальные значения (log scale)')
    plt.ylabel('Предсказанные значения (log scale)')
    plt.title(f'Сравнение предсказаний с реальными значениями\n{model_name}')
    save_plot(plt.gcf(), f'predictions_{model_name.lower().replace(" ", "_")}')

def plot_feature_importance(model, feature_names, model_name):
    if model_name == 'Random Forest':
        importance = model.feature_importances_
    else:  # XGBoost
        importance = model.feature_importances_
    
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feat_imp.head(20), x='importance', y='feature')
    plt.title(f'Топ-20 важных признаков ({model_name})')
    plt.xlabel('Важность признака')
    save_plot(plt.gcf(), f'feature_importance_{model_name.lower().replace(" ", "_")}')

# Загрузка данных
df = pd.read_csv('data/train.csv')
print('Размер датасета:', df.shape)
print('\nИнформация о датасете:')
df.info()

# Определяем числовые признаки
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns

# Визуализация данных
plt.figure(figsize=(20, 16))  # Увеличиваем размер графика
sns.heatmap(df[numeric_features].corr(), 
            annot=True,  # Показываем значения
            cmap='coolwarm',
            center=0,
            fmt='.2f',  # Округляем значения до 2 знаков
            annot_kws={'size': 8},  # Уменьшаем размер шрифта значений
            square=True)  # Делаем ячейки квадратными
plt.xticks(rotation=45, ha='right')  # Поворачиваем подписи осей
plt.yticks(rotation=0)
plt.title('Тепловая карта корреляций числовых признаков', pad=20)  # Добавляем отступ для заголовка
save_plot(plt.gcf(), 'correlation_heatmap')

# 2.2 Анализ распределения важных числовых признаков
important_features = ['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'LotArea']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for idx, feature in enumerate(important_features):
    row = idx // 2
    col = idx % 2
    sns.scatterplot(data=df, x=feature, y='SalePrice', ax=axes[row, col])
    axes[row, col].set_title(f'Зависимость цены от {feature}')
save_plot(plt.gcf(), 'important_features_scatter')

# 2.3 Boxplot для категориальных признаков
categorical_features = ['OverallQual', 'Neighborhood', 'HouseStyle', 'SaleCondition']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for idx, feature in enumerate(categorical_features):
    row = idx // 2
    col = idx % 2
    sns.boxplot(data=df, x=feature, y='SalePrice', ax=axes[row, col])
    axes[row, col].set_xticklabels(axes[row, col].get_xticklabels(), rotation=45)
    axes[row, col].set_title(f'Распределение цен по {feature}')
save_plot(plt.gcf(), 'categorical_features_boxplot')

# 3. Анализ пропущенных значений
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

plt.figure(figsize=(12, 6))
missing_values.plot(kind='bar')
plt.title('Количество пропущенных значений по признакам')
plt.xlabel('Признаки')
plt.ylabel('Количество пропущенных значений')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/missing_values.png')
plt.close()

# 4. Анализ целевой переменной
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['SalePrice'], kde=True)
plt.title('Распределение цен продажи')
plt.xlabel('Цена продажи')

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['SalePrice']), kde=True)
plt.title('Распределение логарифма цен продажи')
plt.xlabel('Log(Цена продажи)')

plt.tight_layout()
plt.savefig('figures/price_distribution.png')
plt.close()

print('\nОсновные статистики цен продажи:')
print(df['SalePrice'].describe())

# 5. Корреляционный анализ
correlations = df[numeric_features].corr()['SalePrice'].sort_values(ascending=False)

plt.figure(figsize=(12, 6))
correlations[1:16].plot(kind='bar')
plt.title('Топ-15 признаков по корреляции с ценой продажи')
plt.xlabel('Признаки')
plt.ylabel('Корреляция с ценой продажи')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/correlations.png')
plt.close()

# 5. Подготовка данных
def prepare_data(df):
    df_prep = df.copy()
    
    numeric_features = df_prep.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df_prep.select_dtypes(include=['object']).columns
    
    for feature in numeric_features:
        df_prep[feature].fillna(df_prep[feature].median(), inplace=True)
    
    for feature in categorical_features:
        df_prep[feature].fillna(df_prep[feature].mode()[0], inplace=True)
    
    df_prep = pd.get_dummies(df_prep, columns=categorical_features)
    
    return df_prep

# Подготовка данных
df_prepared = prepare_data(df)

X = df_prepared.drop(['SalePrice', 'Id'], axis=1)
y = np.log1p(df_prepared['SalePrice'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('\nРазмеры обучающей выборки:', X_train_scaled.shape)
print('Размеры тестовой выборки:', X_test_scaled.shape)

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Сначала вычислим метрики для логарифмированных значений
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_log = r2_score(y_test, y_pred)
    
    # Затем аккуратно выполним обратное преобразование
    try:
        y_true_exp = np.expm1(y_test)
        y_pred_exp = np.expm1(y_pred)
        
        # Обрезаем слишком большие значения
        max_price = 1000000  # Максимальная цена в $1M
        y_pred_exp = np.clip(y_pred_exp, 0, max_price)
        
        rmse = np.sqrt(mean_squared_error(y_true_exp, y_pred_exp))
        r2 = r2_score(y_true_exp, y_pred_exp)
    except:
        print(f"Предупреждение: Не удалось вычислить метрики в исходном масштабе для {model_name}")
        rmse = rmse_log
        r2 = r2_log
    
    print(f'\nРезультаты для модели {model_name}:')
    print(f'RMSE (log scale): {rmse_log:.4f}')
    print(f'R2 Score (log scale): {r2_log:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R2 Score: {r2:.4f}')
    
    # Визуализация в логарифмическом масштабе
    plot_predictions(y_test, y_pred, model_name)
    
    return rmse, r2

# Обучение моделей
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    rmse, r2 = train_and_evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
    results[name] = {'RMSE': rmse, 'R2': r2}
    
    if name in ['Random Forest', 'XGBoost']:
        plot_feature_importance(model, X.columns, name)

# Визуализация результатов моделей
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
rmse_values = [results[model]['RMSE'] for model in results]
plt.bar(results.keys(), rmse_values)
plt.title('Сравнение моделей по RMSE')
plt.xticks(rotation=45)
plt.ylabel('RMSE')

plt.subplot(1, 2, 2)
r2_values = [results[model]['R2'] for model in results]
plt.bar(results.keys(), r2_values)
plt.title('Сравнение моделей по R2')
plt.xticks(rotation=45)
plt.ylabel('R2 Score')

save_plot(plt.gcf(), 'model_comparison') 