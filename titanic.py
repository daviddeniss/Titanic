import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar o dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Visualizar as primeiras linhas
print(df.head())

# Informações básicas sobre o dataset
print(df.info())

# Estatísticas descritivas
print(df.describe())

# Verificar valores faltantes
print(df.isnull().sum())

# Preencher valores faltantes na coluna 'Age' com a média
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Preencher valores faltantes na coluna 'Embarked' com o valor mais frequente
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Remover a coluna 'Cabin' pois tem muitos valores faltantes
df.drop('Cabin', axis=1, inplace=True)

# Converter 'Sex' para valores numéricos (male: 0, female: 1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Visualizar o dataset após a limpeza
print(df.head())

# Taxa de sobrevivência por sexo
survival_by_sex = df.groupby('Sex')['Survived'].mean()
print("Taxa de sobrevivência por sexo:\n", survival_by_sex)

# Taxa de sobrevivência por classe
survival_by_class = df.groupby('Pclass')['Survived'].mean()
print("Taxa de sobrevivência por classe:\n", survival_by_class)

# Média de idade dos sobreviventes vs não sobreviventes
age_survival = df.groupby('Survived')['Age'].mean()
print("Média de idade dos sobreviventes vs não sobreviventes:\n", age_survival)

survival_by_sex.plot(kind='bar', color=['blue', 'pink'])
plt.title('Taxa de Sobrevivência por Sexo')
plt.xlabel('Sexo (0: Masculino, 1: Feminino)')
plt.ylabel('Taxa de Sobrevivência')
plt.show()

# Gráfico de barras da taxa de sobrevivência por classe
survival_by_class.plot(kind='bar', color=['red', 'green', 'blue'])
plt.title('Taxa de Sobrevivência por Classe')
plt.xlabel('Classe')
plt.ylabel('Taxa de Sobrevivência')
plt.show()

# Histograma da idade dos passageiros
df['Age'].plot(kind='hist', bins=20, color='purple')
plt.title('Distribuição de Idade dos Passageiros')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

df.to_csv('titanic_cleaned.csv', index=False)