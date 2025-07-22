import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Function to load and preprocess data
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    
    data = data.drop(['id'], axis=1)

    # Fill missing values
    data['bmi'].fillna(data['bmi'].mean(), inplace=True)
    data['smoking_status'].fillna('Missing', inplace=True)
    
    # Handle outliers for BMI
    Q1 = data['bmi'].quantile(0.25)
    Q3 = data['bmi'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data['bmi'] = np.where(data['bmi'] > upper_bound, upper_bound, data['bmi'])
    data['bmi'] = np.where(data['bmi'] < lower_bound, lower_bound, data['bmi'])
    
    # Handle outliers for avg_glucose_level
    Q1 = data['avg_glucose_level'].quantile(0.25)
    Q3 = data['avg_glucose_level'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data['avg_glucose_level'] = np.where(data['avg_glucose_level'] < lower_bound, lower_bound, data['avg_glucose_level'])
    data['avg_glucose_level'] = np.where(data['avg_glucose_level'] > upper_bound, upper_bound, data['avg_glucose_level'])
    
    # Encode categorical variables
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    return data


def create_model(data):
    X = data.drop(['stroke'], axis=1)
    y = data['stroke']

    # Oversampling using SMOTE
    smote = SMOTE(random_state=42)
    X_sampled, y_sampled = smote.fit_resample(X, y)
    new_data = pd.DataFrame(X_sampled, columns=X.columns)
    new_data['stroke'] = y_sampled

    # Scale the data
    numerical_cols = ['age', 'bmi', 'avg_glucose_level']
    scaler = StandardScaler()
    X_sampled[numerical_cols] = scaler.fit_transform(X_sampled[numerical_cols])

    # Identify skewed features
    skewness = X_sampled[numerical_cols].apply(lambda x: x.skew())
    skewed_features = skewness[skewness > 0.75].index

    # Apply log transformation to reduce skewness
    for feature in skewed_features:
        X_sampled[feature] = np.log1p(X_sampled[feature] + 1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)
  
    # Train the model 


    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # y_pred_prob = model.predict_proba(X_test)[:, 1] 

    y_test = y_test.astype(int)
    
    # Test the model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))

    print("Classification report: \n", classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
  
    return model, scaler, X.columns

def main():
    data = get_clean_data()
    model, scaler, feature_names = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('model/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

if __name__ == '__main__':
    main()
