########## cmd - streamlit run streamlit_api.py #############

import pickle
import streamlit as st

pickle_in = open("classifier.pkl",'rb')
classifier = pickle.load(pickle_in)

# Testing URL
# http://127.0.0.1:5000/predict?variance=3&skewness=8&curtosis=-2&entropy=-1 ---> 0
# http://127.0.0.1:5000/predict?variance=-5&skewness=9&curtosis=-0.3&entropy=-5 ---> 1
#@app.route('/predict')
def predict_note_authentication(variance,skewness,curtosis,entropy):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])  
    print(prediction)
    return prediction


def main():
    st.title("Banker Authentication")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    variance = st.text_input("Variance","Type Here")
    skewness = st.text_input("skewness","Type Here")
    curtosis = st.text_input("curtosis","Type Here")
    entropy = st.text_input("entropy","Type Here")
    result=""
    if st.button("predict"):
        result=predict_note_authentication(variance,skewness,curtosis,entropy)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__ == '__main__':
    main()