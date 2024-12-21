# _____________ Import Python Libraries _________________ #

import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ________________ Page Configuration Section _____________  # 

st.set_page_config(
    page_title="Data Ocean",
    page_icon= '🔥'
)

# _________________ Web Page Info Section _____________________ #

st.title(":red[Data] :blue[Analytic] :orange[Portal & Machine Learning]")
st.header(":rainbow[Explore Data With Ease]")

# __________________ File Upload Section _________________ #

file = st.file_uploader('Drop Your CSV, Excel', type=['csv', 'xlsx'])

if file is not None:
    try:
        if file.name.endswith('csv'):
            data = pd.read_csv(file)
        elif file.name.endswith('xlsx'):
            data = pd.read_excel(file)
        else:
            pass

        st.dataframe(data)
        st.success("File Successfully Uploaded" ,icon='🎉')

        # ________________ Basic Info Summary Section ______________  #

        st.subheader(':rainbow[Basic Information of The Dataset]',divider='violet')
        tab1, tab2, tab3, tab4 ,tab5 , tab6 = st.tabs(['Summary', 'Top & Bottom Rows', 'Data Types', 'Columns','Missing Values','Duplicates Value'])

        with tab1:
            st.write(f'There are {data.shape[0]} Rows and {data.shape[1]} Columns in The Dataset')
            st.subheader(':blue[Statistical Summary]')
            st.dataframe(data.describe())

        with tab2:
            st.subheader(':gray[Top Rows]')
            top_rows = st.slider('Number of Rows to Fetch', 1, data.shape[0], key='topslider')
            st.dataframe(data.head(top_rows))

            st.subheader(':green[Bottom Rows]')
            bottom_rows = st.slider('Number of Rows to Fetch', 1, data.shape[0], key='bottomslider')
            st.dataframe(data.tail(bottom_rows))

        with tab3:
            st.subheader(':orange[Data Types]')
            st.write(data.dtypes.tolist())

        with tab4:
            st.subheader(':green[Columns]')
            st.write(data.columns.tolist())

        with tab5:
            st.subheader(':red[Missing Values]')
            missing_values = data.isnull().sum()
            st.dataframe(missing_values)
            if missing_values.sum() > 0:
                remove_tab, fill_tab = st.tabs(['Remove Missing Values', 'Fill Missing Values'])

                with remove_tab:
                    if st.checkbox("Remove Rows with Missing Values"):
                        data = data.dropna(inplace=True)
                        st.success('Rows with missing values removed!', icon="🎉")

                with fill_tab:
                    replace_nulls = st.selectbox('Replace Missing Values With:', ['None', 'Mean', 'Median', 'Mode'])

                    if replace_nulls != 'None':
                        for col in data.select_dtypes(include=[np.number]):
                            if replace_nulls == 'Mean':
                                data[col].fillna(data[col].mean(), inplace=True)
                            elif replace_nulls == 'Median':
                                data[col].fillna(data[col].median(), inplace=True)
                            elif replace_nulls == 'Mode':
                                data[col].fillna(data[col].mode()[0], inplace=True)
                        st.success("Missing values replaced successfully!", icon='✅')
            else:
                st.success("No missing values detected.", icon='🔥')

        with tab6:
            st.subheader(':green[Duplicate Values]')
            duplicates = data.duplicated().sum()
            if duplicates ==0:
                st.info(f' No Duplicates Value Found',icon='🔥')

            if duplicates > 0 and st.checkbox('Remove Duplicates'):
                data = data.drop_duplicates()
                st.success('Duplicate rows removed!', icon='🔥')


        # __________________ Value Count Section _____________________ #

        st.subheader(':rainbow[Column Value Count]',divider='green')
        with st.expander('Value Count'):
            col1, col2 = st.columns(2)
            with col1:
                column = st.selectbox('Choose Column Name', options=[None] + data.columns.tolist())
            with col2:
                toprows = st.number_input('Number of Top Rows', min_value=1, step=1, value=5)

            if column:
                result = data[column].value_counts().reset_index().head(toprows)
                result.columns = [column, 'count']  
                st.dataframe(result)

                if not result.empty:
                    fig = px.bar(data_frame=result, x=column, y='count', template='plotly_white')
                    st.plotly_chart(fig)

                    fig = px.line(data_frame=result, x=column, y='count')
                    st.plotly_chart(fig)

                    fig = px.pie(data_frame=result, names=column, values='count')
                    st.plotly_chart(fig)

        # ______________ GroupBy Section _________________________ #

        st.subheader(':blue[Groupby : Simplify Your Data Analysis]',divider='violet')
        st.write("Groupby allows you to summarize data by categories.")

        with st.expander('Group By Your Columns'):
            col1, col2, col3 = st.columns(3)

            with col1:
                groupby_cols = st.multiselect('Choose Columns to Group By', options=data.columns.tolist())

            with col2:
                operation_col = st.selectbox("Choose Column for Operation", options=data.columns.tolist())

            with col3:
                operation = st.selectbox("Choose Operation", options=['sum', 'max', 'min', 'count', 'mean', 'median'])

            if groupby_cols and operation_col and operation:
                result = data.groupby(groupby_cols).agg(newcol=(operation_col, operation)).reset_index()
                st.dataframe(result)

                st.subheader(':rainbow[Data Visualization]')
                graph_type = st.selectbox('Choose Graph Type', options=['line', 'bar', 'scatter', 'pie', 'sunburst'])

                if graph_type == 'line':
                    x_axis = st.selectbox('X Axis', options=result.columns.tolist())
                    y_axis = st.selectbox('Y Axis', options=result.columns.tolist())
                    fig = px.line(data_frame=result, x=x_axis, y=y_axis)
                    st.plotly_chart(fig)

                elif graph_type == 'bar':
                    x_axis = st.selectbox('X Axis', options=result.columns.tolist())
                    y_axis = st.selectbox('Y Axis', options=result.columns.tolist())
                    color = st.selectbox('Color Information', options=[None] + result.columns.tolist())
                    fig = px.bar(data_frame=result, x=x_axis, y=y_axis, color=color)
                    st.plotly_chart(fig)

                elif graph_type == 'pie':
                    values = st.selectbox("Numerical Values", options=result.columns.tolist())
                    names = st.selectbox('Labels', options=result.columns.tolist())
                    fig = px.pie(data_frame=result, names=names, values=values)
                    st.plotly_chart(fig)

                elif graph_type == 'scatter':
                    x_axis = st.selectbox('X Axis', options=result.columns.tolist())
                    y_axis = st.selectbox('Y Axis', options=result.columns.tolist())
                    size = st.selectbox('Size Column', options=[None] + result.columns.tolist())
                    color = st.selectbox('Color Information', options=[None] + result.columns.tolist())
                    fig = px.scatter(data_frame=result, x=x_axis, y=y_axis, color=color, size=size)
                    st.plotly_chart(fig)

                elif graph_type == 'sunburst':
                    path = st.multiselect('Path', options=result.columns.tolist())
                    fig = px.sunburst(data_frame=result, path=path, values='newcol')
                    st.plotly_chart(fig)

        #_________________ Machine Learning_______________ #

        st.subheader(":orange[Basic Machine Learning]",divider='green')
        ml_task = st.selectbox("Select ML Task", ["None", "Linear Regression"])
        if ml_task == "Linear Regression":
            target_col = st.selectbox("Select Target Column", data.columns)
            feature_cols = st.multiselect("Select Feature Columns", data.columns)

            if target_col and feature_cols:
                X = data[feature_cols]
                y = data[target_col]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                st.write("Mean Squared Error:", mean_squared_error(y_test, predictions)," | " , " R2 Score :" ,r2_score(y_test,predictions)*100)


    except Exception as e:
        st.error(f"An error occurred: {e}")
