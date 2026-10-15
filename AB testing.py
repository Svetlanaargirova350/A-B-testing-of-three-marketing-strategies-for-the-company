# Кейс
# (Kaggle: https://www.kaggle.com/datasets/chebotinaa/fast-food-marketing-campaign-ab-test)
# Сеть ресторанов быстрого питания планирует добавить в меню новый продукт. 
# Однако компания еще не определилась с выбором из трех возможных маркетинговых кампаний 
# для продвижения новинки. Чтобы выяснить, какая из них окажет наибольшее влияние на продажи, 
# новый продукт был представлен в нескольких случайно выбранных точках на разных рынках. 
# В каждой точке использовалась своя рекламная кампания, и в течение первых четырех 
# недель велся учет еженедельных продаж нового продукта.

# Этап 1
# Формирование гипотез
# H0: Распределения продаж одинаковы во всех трех группах
# H1: Между маркетинговыми стратегиями существует статистически значимый эффект влияния на продажи
# Определение MDE (так как тест уже запустили и собрали данные, здесь невозможно определить MDE)

# Этап 2: сбор данных и разбиение случайным способом клиентов на взаимодействие с маркетинговыми 
# компаниями в течение 4ех недель

# Этап 3: первичная обработка данных
import pandas as pd
data = pd.read_csv("/Users/Desktop/Marketing_Campaign.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data.columns
data.head(5)
# Колонки: 'MarketID', 'MarketSize', 'LocationID', 'AgeOfStore', 'Promotion', 'week', 'SalesInThousands'
data.describe()
# Итого 548 наблюдений
data.dtypes
# Данные уже приведены в нормальном формате
data.isna().sum()
# В данных отсутствуют пропуски
data.nunique()
# Promotion: 3
# LocationID: 137

promo_statistics = data.groupby('Promotion')['SalesInThousands'].agg(['sum', 'count', 'mean'])
promo_statistics
# Выводы: в 1-ом Promotion самая высокая средняя величина SalesInThousands
# В 3-ем Promotion самая высокая сумма SalesInThousands
# По трем группам примерно одинаковое количество поситителей. Для Promotion 1 - 172, 
# Для Promotion 2 - 188, для Promotion 3 - 188

market_statistics = data.groupby('MarketSize')['SalesInThousands'].describe()
market_statistics
# Выводы: на крупном рынке самая высокая средняя величина SalesInThousands
# На Medium рынке высокая сумма SalesInThousands

week_statistics = data.groupby(['week', 'Promotion'])['SalesInThousands'].mean()
week_statistics
# Выводы: общая сумма продаж и среднее значене по продажам колеблется у одного значения

# Этап 4: визуализация данных
import matplotlib.pyplot as plt
import seaborn as sns

# Визуализация Promotion
sns.histplot(data=data, x='SalesInThousands' , bins='auto', color='blue')

# Визуализация по группам Promotion
promotion1 = data[data['Promotion'] == 1]
sns.histplot(data=promotion1, x='SalesInThousands' , bins='auto', kde= True, color ='pink')
promotion2 = data[data['Promotion'] == 2]
sns.histplot(data=promotion2, x='SalesInThousands' , bins='auto', kde= True, color ='orange')
promotion3 = data[data['Promotion'] == 3]
sns.histplot(data=promotion3, x='SalesInThousands' , bins='auto', kde= True, color='green')

# Анализ по группам
plt.figure(figsize=(8,6))
sns.stripplot(data=data, x='Promotion', y='SalesInThousands')
plt.show()

plt.figure(figsize=(8,6))
sns.boxenplot(data=data, x='MarketSize', y='SalesInThousands')
plt.show()
# Визуально все распределения имеют ненормальное распределение
plt.figure(figsize=(8,6))
sns.stripplot(data=data, x='week', y='SalesInThousands')
plt.show()

# Этап 4: проверка A/A теста
# Всего 137 ресторанов, поэтому данные могут быть зависимыми
# Необходимо преобразовать данные: объединение по ресторанам
new_data = data.groupby(['LocationID','Promotion', 'MarketSize'], as_index=False).agg(
    sales = ('SalesInThousands', 'mean'),
    age = ('AgeOfStore', 'first')
    )

# Три вида рынков: Small, Medium, Large
# Необходимо проверить, что группы изначально сбалансированы
# Для этого применим Хи-квадрат
from scipy import stats
cont_table = pd.crosstab(new_data['Promotion'], new_data['MarketSize'])
cont_table
p_chi2 = stats.chi2_contingency(cont_table)[1]
print(p_chi2) # 0.879 > 0.05
# Применение стратегий по трем рынкам равномерно и сбалансировано

# Этап 5: проверка тестов на однородность и нормальность
# Ключевые статистические условия (допущения)

# 1. Однородность (равенство) дисперсий
group_1 = new_data[new_data['Promotion'] == 1]['sales']
group_1
group_2 = new_data[new_data['Promotion'] == 2]['sales']
group_2
group_3 = new_data[new_data['Promotion'] == 3]['sales']
group_3
stat, p_val = stats.levene(group_1, group_2, group_3)

stat
p_val # 0.6271730417231256
# p-value > 0.05, дисперсии равны

# 2. Нормальность распределения
# Визуальный анализ показал, что распределение не нормальное
# Проверка по тесту Шапиро: ни одна из трех групп не имеет нормального распределения
p1 = stats.shapiro(group_1)[1]
p2 = stats.shapiro(group_2)[1]
p3 = stats.shapiro(group_3)[1]
print(p1) # 0.00015705579426139593
print(p2) # 2.3925502318888903e-05
print(p3) # 0.00013465984375216067
# Классический ANOVA применять нельзя

# Этап 6: статистический анализ
statistics, p_value = stats.kruskal(group_1, group_2, group_3)
statistics
p_value #0.00017

# Результат значим: 0.00017 < 0.05
# Как минимум одна маркетинговая стратегия работает значимо лучше/хуже остальных

# Теперь можно узнать, какая стратегия наилучшая
# тест Манна-Уитни (Mann-Whitney U-test) с поправкой Бонферрони: попарные тесты

p_1_2 = stats.mannwhitneyu(group_1, group_2)[1]
p_1_3 = stats.mannwhitneyu(group_1, group_3)[1]
p_2_3 = stats.mannwhitneyu(group_2, group_3)[1]

# Применяем поправку Бонферрони (перемножаем P-value на количество значений)
corr_p_1_2 = min(p_1_2 * 3, 1.0)
corr_p_1_3 = min(p_1_3 * 3, 1.0)
corr_p_2_3 = min(p_2_3 * 3, 1.0)
print(corr_p_1_2) # 0.00033
print(corr_p_1_3) # 0.53203
print(corr_p_2_3) # 0.00746
# Стратегия 2 работает хуже, чем стратегии 1 и 3

# Анализ медиан
print(group_1.median()) # 56.075
print(group_2.median()) # 46.655
print(group_3.median()) # 49.8525

# Стратегия 2 показала худший результат
# Оценка Effect size
med1 = group_1.median()
med2 = group_2.median()
med3 = group_3.median()
effect1 = ((med1 - med2)/med2)*100
effect1 # 20.19
# Медианные продажи в группе 1 выше, чем в группе 2, на 20.19%
effect2 = ((med3 - med2)/med2)*100
effect2 # 6.85
# Медианные продажи в группе 3 выше, чем в группе 2, на 6.85%

# 95% бутстреп-доверительный интервал для разности медиан между двумя группами
import numpy as np
rng = np.random.default_rng(42) # для генерации случайных чисел

# Матрицы псевдовыборок
random1 = rng.choice(group_1, size=(5000, len(group_1)), replace = True)
random2 = rng.choice(group_2, size=(5000, len(group_2)), replace = True)
random3 = rng.choice(group_3, size=(5000, len(group_3)), replace = True)

# Разность медиан
diff1 = np.median(random1, axis = 1) - np.median(random2, axis = 1)
ci_lower1, ci_upper1 = np.percentile(diff1, [2.5, 97.5])
ci_lower1, ci_upper1
# Доверительный интервал: (4.009999999999998, 13.505062500000001)
# Различие между медианами двух групп является статистически значимым на уровне 5%.

diff2 = np.median(random3, axis = 1) - np.median(random2, axis = 1)
ci_lower2, ci_upper2 = np.percentile(diff2, [2.5, 97.5])
ci_lower2, ci_upper2
# Доверительный интервал: (1.0949999999999989, 9.147499999999994)
# Различие между медианами двух групп является статистически значимым на уровне 5%.

# Ключевой вывод: рекомендуется рассматривать стратегии 1 и 3
# Ограничения: 137 ресторанов и 4 недели сбора данных, отсутствие информации о затратах компании




