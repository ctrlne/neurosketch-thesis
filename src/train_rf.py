import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_kinematic_model():
    # Load the fully labeled and augmented dataset
    df = pd.read_csv(r"data\processed\final_labeled_kinematics.csv")
    
    # isolate the kinematic features (X) and the target label (y)
    features = ['mean_velocity', 'pressure_variance', 'jerk_variance', 'tremor_power_4_to_6_hz']
    X = df[features]
    y = df['target']
    
    # Stratified split ensures the 80/20 ratio of PD to HC is maintained in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # initialize and train the rf classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # predict unseen test set/data
    predictions = rf_model.predict(X_test)
    
    # Output evaluation metrics
    print("=== Random Forest Kinematic Stream Results ===")
    print(f"Overall Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=["Healthy (0)", "Parkinson's (1)"]))

if __name__ == "__main__":
    train_kinematic_model()