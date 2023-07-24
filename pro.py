import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
#Load the ANN model
model=tf.keras.models.load_model(r'/content/breast_cancer_model.h5')

#Function to process the input data
def preprocess_data(data):
    #perform any necessary preprocessing steps
    scaler=MinMaxScaler()
    scaled_data=scaler.fit_transform(data)
    return scaled_data[:, :30] #Keep only the first 30 features
#Function to make predicttions
def make_prediction(data):
    preprocessed_data=preprocess_data(data)
    prediction=model.predict(preprocessed_data)
    return prediction
#Main function to run the Streamlit app
def main():
    st.title("Breast Cancer Prediction")

    #Define feature names
    feature_names=[
        'mean radius','mean texture','mean perimeter','mean area',
        'mean smoothness','mean compactness','mean concavity','mean concave points','mean symmetry',
        'mean fractal dimension','radius error','texture error','perimeter error','area error',
        'smoothness error','compactness error','concavity error','concave points error','symmetry error',
        'fractal dimension error','worst radius','worst texture','worst perimeter','worst area','worst smoothness',
        'worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension'
    ]
    st.write('Enter the values for the features below:')
    #Create input fields for each feature
    input_data=[]
    for feature_name in feature_names:
        value=st.number_input(feature_name,value=0.000000)
        input_data.append(value)
    #Create a dataframe with the input data
    df=pd.DataFrame([input_data],columns=feature_names)
    #Make a prediction when the 'Predict' button is clicked
    if st.button('Predict'):
        prediction=make_prediction(df)
        if prediction[0]==0:
            st.write('You have Malignant(M) tumor')
        else:
            st.write('You have a Benign(B) tumor')
if __name__ == '__main__':
    main()
    


