import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('stacking_synchronization_experiment/results.csv')

# Stacking accuracy by Dataset, Model Set and Setting
df_stacking = df[df['method'] == 'Stacking'].copy()
df_stacking['Setting'] = df_stacking.apply(lambda x: f"Sync={x['sync']}, Avg={x['avg']}", axis=1)

g = sns.catplot(data=df_stacking, x='dataset', y='accuracy', hue='Setting', col='model_set', kind='bar', height=5, aspect=1.2)
g.set_axis_labels("Dataset", "Accuracy")
g.set_titles("Model Set: {col_name}")
plt.ylim(0.9, 1.0)
plt.tight_layout()
plt.savefig('stacking_synchronization_experiment/stacking_accuracy_by_set.png')

# Overall comparison
df['Group'] = df.apply(lambda x: f"{x['method']} (Sync={x['sync']})" if x['method']=='Bagging' else f"{x['method']} (Sync={x['sync']}, Avg={x['avg']})", axis=1)

plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='Group', y='accuracy')
plt.xticks(rotation=45, ha='right')
plt.title('Accuracy Distribution across all datasets and model sets')
plt.tight_layout()
plt.savefig('stacking_synchronization_experiment/overall_accuracy_box.png')

print("Updated plots generated successfully.")
