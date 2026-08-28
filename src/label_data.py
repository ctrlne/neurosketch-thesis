import pandas as pd

def label_dataset():
    print("Cross-referencing patient IDs with the PaHaW master clinical file...")
    
    # 1. Load your augmented features and the PaHaW master Excel file
    features_path = r"data\processed\kinematic_features_master_augmented.csv"
    excel_path = r"data\raw\data\PaHaW\PaHaW_files\corpus_PaHaW.xlsx"
    
    df_features = pd.read_csv(features_path)
    
    # Using openpyxl engine to read the modern .xlsx file
    df_labels = pd.read_excel(excel_path, engine='openpyxl')
    
    # 2. Clean the Master ID column 
    # Pandas sometimes reads '00001' as the number 1. We force it back to a 5-digit string.
    df_labels['ID'] = df_labels['ID'].astype(str).str.zfill(5)
    
    # 3. Extract the base ID from your generated filenames
    # Turns "00001_1_1_aug1" into just "00001" so it matches the Excel sheet
    df_features['base_id'] = df_features['patient_id'].apply(lambda x: str(x).split('_')[0])
    
    # 4. Merge the datasets based on the ID
    merged_df = pd.merge(df_features, df_labels[['ID', 'Disease']], 
                         left_on='base_id', right_on='ID', how='left')
    
    # 5. Convert PD / H into Machine Learning binary labels (1 = PD, 0 = Healthy)
    merged_df['target'] = merged_df['Disease'].apply(lambda x: 1 if x == 'PD' else 0)
    
    # 6. Clean up temporary columns and save
    merged_df.drop(columns=['base_id', 'ID', 'Disease'], inplace=True)
    final_output_path = r"data\processed\final_labeled_kinematics.csv"
    merged_df.to_csv(final_output_path, index=False)
    
    print(f"Success! Final labeled dataset saved to: {final_output_path}")
    # print(merged_df[['patient_id', 'target']].head(10)) # Preview the first 10 rows
    print(merged_df['target'].value_counts())

if __name__ == "__main__":
    label_dataset()