import pandas as pd

df = pd.read_csv(r'c:\Users\Shraddha\Desktop\MCA\ML\Mini Project\data\ir_train.csv')
print('Full Train shape:', df.shape)
print('\nData types:')
print(df.dtypes.to_string())
print('\n--- Null counts ---')
print(df.isnull().sum().to_string())
print('\n--- Target distribution ---')
print(df['is_delayed'].value_counts())
print(f'\nDelay rate: {df["is_delayed"].mean():.4f}')
print('\n--- Numerical summary ---')
print(df.describe().to_string())
print('\n--- Categorical columns unique values ---')
for col in df.select_dtypes(include='object').columns:
    print(f'{col}: {df[col].nunique()} unique values')
    if df[col].nunique() <= 20:
        print(f'  Values: {df[col].value_counts().to_dict()}')
