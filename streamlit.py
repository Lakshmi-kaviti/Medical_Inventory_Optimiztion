import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import joblib
sarimax_model = joblib.load('best_model_sarimax.pkl')
preprocess_fit = joblib.load('preprocess_fit.pkl')


def main():

    st.title("Forecasting")
    st.sidebar.title("Forecasting")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Medical Inventory optimization and Forecasting </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")

    uploadedFile = st.sidebar.file_uploader("Choose a file", type=[
                                            'csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None:
        try:

            data = pd.read_csv(uploadedFile)
        except:
            try:
                data = pd.read_excel(uploadedFile)
            except:
                data = pd.DataFrame(uploadedFile)

    else:
        st.sidebar.warning("you need to upload a csv or excel file.")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)

    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")

    if st.button("Predict"):
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

        medical = data
        ###############################################
        st.subheader(":red[Forecast for New data]", anchor=None)

        medical['Patient_ID'] = medical['Patient_ID'].astype(str)

        medical['Dateofbill'] = medical['Dateofbill'].astype(
            'datetime64[ns]')

        ################################### Handling Duplicates #################################
        medical.duplicated().sum()

        medical.drop_duplicates(inplace=True, keep='first')
        #################
        sarimax_data = medical.drop(['Patient_ID'], axis=1)
        

        medical['weekofbill'] = medical['Dateofbill'].dt.isocalendar().week

        sarimax_data.drop(['Typeofsales', 'DrugName', 'RtnMRP',
                          'Final_Sales', 'Dateofbill'], axis=1, inplace=True)
        # When we drop this feature i got MAPE = 5.89%
        #sarimax_data.drop(['SubCat'], axis=1, inplace=True)
        # and it not drops this i got the 2 approximately
        sarimax_data.drop(['Specialisation'], axis=1, inplace=True)
        sarimax_data.drop(['Quantity'], axis=1, inplace=True)

        sarimax_data_preprocess = pd.DataFrame(preprocess_fit.transform(sarimax_data).toarray(), columns=preprocess_fit.get_feature_names_out())

        sarimax_data = pd.concat(
            [sarimax_data_preprocess, medical[['weekofbill', 'Quantity']]], axis=1)

        sarimax_data = sarimax_data.groupby('weekofbill').sum().reset_index()

        pred_sarimax = sarimax_model.predict(
            start=sarimax_data.index[0], end=sarimax_data.index[-1])
        # , exog=train.drop(['Quantity', 'weekofbill'], axis=1)

        # So here we are applying for the old data i am commenting.

        results = pd.concat([sarimax_data[['weekofbill','Quantity']],pred_sarimax],axis =1)
        results.columns = ['weekofbill','Actual_Quantity','Predicted_Quantity']
        results.to_sql('forecast_results_dd', con=engine,
                       if_exists='replace', index=False, chunksize=1000)

        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
                # Apply the background_gradient to the Styler object
        styled_df = results.style.background_gradient(cmap=cm)

        # Set the precision for floating-point numbers
        styled_df = styled_df.format(precision=2)

        # Display the styled DataFrame in Streamlit
        st.table(styled_df)
        #st.table(result.style.background_gradient(cmap=cm).set_precision(2))

        st.text("")
        st.subheader(
            ":red[plot forecasts against actual outcomes]", anchor=None)
        # plot forecasts against actual outcomes
        fig, ax = plt.subplots()
        ax.plot(sarimax_data.Quantity, '-b', label='Actual Value')
        ax.plot(pred_sarimax, '-r', label='Predicted value')
        ax.legend()
        st.pyplot(fig)

        # data.to_sql('forecast_pred', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        # #st.dataframe(result) or
        # #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))

        # import seaborn as sns
        # cm = sns.light_palette("blue", as_cmap=True)
        # st.table(result.style.background_gradient(cmap=cm).set_precision(2))


if __name__ == '__main__':
    main()
