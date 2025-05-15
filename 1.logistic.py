import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    GENDER = st.sidebar.selectbox('Gender', ('male', 'female'))
    PCLASS = st.sidebar.selectbox('Pclass', (1, 2, 3))
    AGE = st.sidebar.number_input('Age', min_value=0, max_value=100, value=30)
    SIBSP = st.sidebar.number_input('Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
    PARCH = st.sidebar.number_input('Parents/Children Aboard', min_value=0, max_value=10, value=0)

    gender_encoded = 1 if GENDER == 'male' else 0

    data = {
        'Sex': gender_encoded,
        'Pclass': PCLASS,
        'Age': AGE,
        'SibSp': SIBSP,
        'Parch': PARCH
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Load and prepare training data
TITANIC_TRAIN = pd.read_csv("Titanic_train.csv")
TITANIC_TRAIN.dropna(inplace=True)

# Encode 'Sex' column
TITANIC_TRAIN['Sex'] = TITANIC_TRAIN['Sex'].map({'male': 1, 'female': 0})

X = TITANIC_TRAIN[['Sex', 'Pclass', 'Age', 'SibSp', 'Parch']]
Y = TITANIC_TRAIN['Survived']

# Train the model
clf = LogisticRegression()
clf.fit(X, Y)

# Make prediction
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Predicted Probability')
st.write(prediction_proba)

