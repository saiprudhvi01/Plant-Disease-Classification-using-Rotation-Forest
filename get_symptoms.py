import pandas as pd

df = pd.read_csv('plant_disease_symptom_observations_18000.csv')
symptoms = df['Symptom_Name'].unique()
print('Total unique symptoms:', len(symptoms))
print('\nSymptoms:')
for i, s in enumerate(symptoms, 1):
    print(f'{i}. {s}')
