from google.colab import drive
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

import pandas as pd
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Datasets/salaries.csv')

inputs = df.drop('salary_more_then_100k', axis='columns')

target = df['salary_more_then_100k']


# no es necesario hacer one hot encoding ni dummy variables

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

print(inputs)

inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')

model = tree.DecisionTreeClassifier()

model.fit(inputs_n, target)
model.score(inputs_n, target)

# salario de google, ingeniero, con bacherlor > 100k

model.predict([[2, 1, 0]])