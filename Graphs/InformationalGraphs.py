import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

surprise = pd.read_csv('../Graphs/surprise_data.csv')
terrain = pd.read_csv('../Graphs/terrain_data.csv')
weather = pd.read_csv('../Graphs/weather_data.csv')
aerialSuper = pd.read_csv('../Graphs/aerialSuper_data.csv')

plt.figure(figsize=(8, 6))
sns.set_style('whitegrid')
sns.countplot(x='wina', hue='surpa', data=surprise, palette='RdBu_r')
plt.title('Element of Surprise')
plt.show()

# Plot for terrain1
plt.figure(figsize=(8, 6))
sns.set_style('whitegrid')
sns.countplot(x='wina', hue='terra1', data=terrain, palette='RdBu_r',)
plt.title('Terrain 1')
plt.show()

# Plot for terrain2
plt.figure(figsize=(8, 6))
sns.set_style('whitegrid')
sns.countplot(x='wina', hue='terra2', data=terrain, palette='RdBu_r')
plt.title('Terrain 2')
plt.show()

# Plot for terrain3
plt.figure(figsize=(8, 6))
sns.set_style('whitegrid')
sns.countplot(x='wina', hue='terra3', data=terrain, palette='RdBu_r')
plt.title('Terrain 3')
plt.show()

# Loop for weather plots (wx1 to wx5)
for j in range(1, 6):
    plt.figure(figsize=(8, 6))
    sns.set_style('whitegrid')
    sns.countplot(x='wina', hue='wx{}'.format(j), data=weather, palette='RdBu_r')
    plt.title('Weather {}'.format(j))
    plt.show()

plt.figure(figsize=(8, 6))
sns.set_style('whitegrid')
sns.countplot(x='wina', hue='aeroa', data=aerialSuper, palette='RdBu_r')
plt.title('Aerial superiority')
plt.show()
