import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv('data.csv')
df['theme'] = df['theme'].apply(lambda x: x.split('_'))

# Step 3: Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Step 4: One-hot encode the themes
encoded_labels = mlb.fit_transform(df['theme'])

# Step 5: Create a DataFrame for the encoded labels
df_encoded = pd.DataFrame(encoded_labels, columns=mlb.classes_)
df_encoded = df_encoded.astype(float)

# Step 6: Convert the one-hot encoded columns back to a list of labels
df['labels'] = df_encoded.values.tolist()

# Step 6: Combine the original dataframe with the encoded labels
df = pd.concat([df, df_encoded], axis=1)

# Step 7: Print the processed DataFrame
print(df.columns)

df.to_csv('data_encoded.csv', index=False)