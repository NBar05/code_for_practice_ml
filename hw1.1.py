import numpy as np
import pandas as pd

# импотрт данных
data = pd.read_csv("data/titanic.csv", header=0)

# Считаем:
# размеченных мужчин и женщин
print(dict(data['Sex'].value_counts()))
# и процент выживших
print(round(np.mean(data['Survived']) * 100, 2))
# и процент пассажиров первого класса
print(round(data['Pclass'][data['Pclass']==1].count() / data['Pclass'].count() * 100, 2))
# и средний возраст и медианное значение
print([round(np.mean(data['Age']), 2), data['Age'].median()])
# и корреляцию двух признаков
print(round(data[['SibSp', 'Parch']].corr(method='pearson').iloc[1, 0], 2))
# и самое частое женское имя
a = data['Name'][data['Sex'] == 'female']
print(a.str.split(' ', expand=True).groupby([2]).count().sort_values([1], ascending=False).head())
