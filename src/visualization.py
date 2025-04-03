import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_target_distribution(y, title='Распределение целевой переменной'):
    """
    Построение графика распределения целевой переменной
    
    Parameters:
    -----------
    y : pandas.Series
        Целевая переменная
    title : str
        Заголовок графика
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=y, bins=50)
    plt.title(title)
    plt.xlabel('Значение')
    plt.ylabel('Количество')
    plt.show()

def plot_correlation_matrix(df, title='Тепловая карта корреляций'):
    """
    Построение тепловой карты корреляций
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Датафрейм с данными
    title : str
        Заголовок графика
    """
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), cmap='coolwarm', center=0)
    plt.title(title)
    plt.show()

def plot_feature_vs_target(df, feature, target='SalePrice', title=None):
    """
    Построение графика зависимости признака от целевой переменной
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Датафрейм с данными
    feature : str
        Название признака
    target : str
        Название целевой переменной
    title : str
        Заголовок графика
    """
    if title is None:
        title = f'Зависимость {target} от {feature}'
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=feature, y=target)
    plt.title(title)
    plt.show()

def plot_categorical_feature_vs_target(df, feature, target='SalePrice', title=None):
    """
    Построение box plot для категориального признака
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Датафрейм с данными
    feature : str
        Название категориального признака
    target : str
        Название целевой переменной
    title : str
        Заголовок графика
    """
    if title is None:
        title = f'Распределение {target} по {feature}'
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=feature, y=target)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def plot_missing_values(df, title='Анализ пропущенных значений'):
    """
    Построение графика пропущенных значений
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Датафрейм с данными
    title : str
        Заголовок графика
    """
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100
    
    plt.figure(figsize=(15, 6))
    missing_percentages.plot(kind='bar')
    plt.title(title)
    plt.xlabel('Признаки')
    plt.ylabel('Процент пропущенных значений')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_model_predictions(y_true, y_pred, title='Сравнение предсказаний с реальными значениями'):
    """
    Построение графика сравнения предсказаний с реальными значениями
    
    Parameters:
    -----------
    y_true : array-like
        Реальные значения
    y_pred : array-like
        Предсказанные значения
    title : str
        Заголовок графика
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(title)
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.show()

def plot_feature_importance(model, feature_names, title='Важность признаков'):
    """
    Построение графика важности признаков
    
    Parameters:
    -----------
    model : sklearn estimator
        Обученная модель с атрибутом feature_importances_
    feature_names : list
        Список названий признаков
    title : str
        Заголовок графика
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show() 