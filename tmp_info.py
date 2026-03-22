import pandas as pd 
from collections import Counter 
df=pd.read_csv('spending_patterns_detailed.csv') 
print(df.head().to_string()) 
print('\nColumns:', df.columns.tolist()) 
print('\nCategory counts (top 10):', Counter(df['Category']).most_common(10)) 
print('Total rows', len(df)) 
