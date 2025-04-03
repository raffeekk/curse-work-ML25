import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def load_data(train_path, test_path):
    """
    Загрузка тренировочного и тестового наборов данных
    
    Parameters:
    -----------
    train_path : str
        Путь к файлу с тренировочными данными
    test_path : str
        Путь к файлу с тестовыми данными
        
    Returns:
    --------
    tuple
        (train_df, test_df) - загруженные датафреймы
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def analyze_missing_values(df):
    """
    Анализ пропущенных значений в датафрейме
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Датафрейм для анализа
        
    Returns:
    --------
    pandas.DataFrame
        Датафрейм с информацией о пропущенных значениях
    """
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Количество пропущенных': missing_values,
        'Процент пропущенных': missing_percentages
    })
    
    return missing_df[missing_df['Количество пропущенных'] > 0].sort_values('Количество пропущенных', ascending=False)

def handle_missing_values(df, strategy='mean'):
    """
    Обработка пропущенных значений
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Датафрейм для обработки
    strategy : str
        Стратегия заполнения пропущенных значений ('mean', 'median', 'most_frequent')
        
    Returns:
    --------
    pandas.DataFrame
        Обработанный датафрейм
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Обработка числовых признаков
    if len(numeric_columns) > 0:
        numeric_imputer = SimpleImputer(strategy=strategy)
        df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    
    # Обработка категориальных признаков
    if len(categorical_columns) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    
    return df

def encode_categorical_features(df):
    """
    Кодирование категориальных признаков
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Датафрейм для кодирования
        
    Returns:
    --------
    pandas.DataFrame
        Кодированный датафрейм
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column].astype(str))
    
    return df, label_encoders

def prepare_data(train_df, test_df):
    """
    Подготовка данных для обучения
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Тренировочный датафрейм
    test_df : pandas.DataFrame
        Тестовый датафрейм
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test) - подготовленные данные
    """
    # Сохраняем целевую переменную
    y_train = train_df['SalePrice']
    
    # Удаляем целевую переменную из тренировочных данных
    train_df = train_df.drop('SalePrice', axis=1)
    
    # Обработка пропущенных значений
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)
    
    # Кодирование категориальных признаков
    train_df, label_encoders = encode_categorical_features(train_df)
    test_df, _ = encode_categorical_features(test_df)
    
    return train_df, y_train, test_df 