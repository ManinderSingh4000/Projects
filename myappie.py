import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
import seaborn as sns
import matplotlib.pyplot as plt

# ================== Page Configuration ================== #
st.set_page_config(
    page_title="Data Ocean",
    page_icon='🔥'
)

st.title(":red[Data] :blue[Analytic] :orange[Portal & Machine Learning]")
st.header(":rainbow[Explore Data With Ease]")

# ================== File Upload Section ================== #
file = st.file_uploader('Drop Your CSV or Excel file', type=['csv', 'xlsx'])
if file is not None:
    try:
        # Load Data
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            data = pd.read_excel(file)
        else:
            st.error("Unsupported file type.")
            st.stop()
            
        st.dataframe(data)
        st.success("File Successfully Uploaded 🎉")
        
        # ================== Basic Information Section ================== #
        st.subheader(':rainbow[Basic Information of The Dataset]', divider='violet')
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Summary', 'Top & Bottom Rows', 'Data Types', 'Columns', 'Missing Values', 'Duplicates'])
        
        with tab1:
            st.write(f'There are {data.shape[0]} rows and {data.shape[1]} columns in the dataset.')
            st.subheader(':blue[Statistical Summary]')
            st.dataframe(data.describe())
            
        with tab2:
            st.subheader(':gray[Top Rows]')
            top_rows = st.slider('Number of top rows to display', 1, data.shape[0], value=5, key='topslider')
            st.dataframe(data.head(top_rows))
            
            st.subheader(':green[Bottom Rows]')
            bottom_rows = st.slider('Number of bottom rows to display', 1, data.shape[0], value=5, key='bottomslider')
            st.dataframe(data.tail(bottom_rows))
            
        with tab3:
            st.subheader(':orange[Data Types]')
            st.write(data.dtypes)
            
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
                    if st.checkbox("Remove rows with missing values"):
                        # Drop rows with missing values (inplace does not need assignment)
                        data.dropna(inplace=True)
                        st.success('Rows with missing values removed! 🎉')
                with fill_tab:
                    replace_strategy = st.selectbox('Replace Missing Values With:', ['None', 'Mean', 'Median', 'Mode'])
                    if replace_strategy != 'None':
                        for col in data.select_dtypes(include=[np.number]).columns:
                            if replace_strategy == 'Mean':
                                data[col].fillna(data[col].mean(), inplace=True)
                            elif replace_strategy == 'Median':
                                data[col].fillna(data[col].median(), inplace=True)
                            elif replace_strategy == 'Mode':
                                data[col].fillna(data[col].mode()[0], inplace=True)
                        st.success("Missing values replaced successfully! ✅")
            else:
                st.success("No missing values detected. 🔥")
            
        with tab6:
            st.subheader(':green[Duplicate Values]')
            duplicates = data.duplicated().sum()
            if duplicates == 0:
                st.info('No duplicate values found. 🔥')
            else:
                st.write(f'Found {duplicates} duplicate rows.')
                if st.checkbox('Remove duplicate rows'):
                    data.drop_duplicates(inplace=True)
                    st.success('Duplicate rows removed! 🔥')
                    
        # ================== Value Count Section ================== #
        st.subheader(':rainbow[Column Value Count]', divider='green')
        with st.expander('Value Count'):
            col1, col2 = st.columns(2)
            with col1:
                column = st.selectbox('Choose a column', options=[None] + data.columns.tolist())
            with col2:
                num_top = st.number_input('Number of top rows', min_value=1, step=1, value=5)
            if column:
                vc = data[column].value_counts().reset_index().head(num_top)
                vc.columns = [column, 'count']
                st.dataframe(vc)
                fig_bar = px.bar(vc, x=column, y='count', template='plotly_white')
                st.plotly_chart(fig_bar)
                fig_line = px.line(vc, x=column, y='count', template='plotly_white')
                st.plotly_chart(fig_line)
                fig_pie = px.pie(vc, names=column, values='count')
                st.plotly_chart(fig_pie)
                
        # ================== GroupBy Section ================== #
        st.subheader(':blue[Groupby : Simplify Your Data Analysis]', divider='violet')
        st.write("Groupby allows you to summarize data by categories.")
        with st.expander('Group By'):
            col1, col2, col3 = st.columns(3)
            with col1:
                groupby_cols = st.multiselect('Select columns to group by', options=data.columns.tolist())
            with col2:
                agg_col = st.selectbox("Select column for aggregation", options=data.columns.tolist())
            with col3:
                operation = st.selectbox("Select aggregation operation", options=['sum', 'max', 'min', 'count', 'mean', 'median'])
            if groupby_cols and agg_col and operation:
                grouped = data.groupby(groupby_cols).agg(result=(agg_col, operation)).reset_index()
                st.dataframe(grouped)
                st.subheader(':rainbow[GroupBy Data Visualization]')
                graph_type = st.selectbox('Select Graph Type', options=['line', 'bar', 'pie', 'scatter', 'sunburst'])
                if graph_type in ['line', 'bar', 'scatter']:
                    x_axis = st.selectbox('X Axis', options=grouped.columns.tolist())
                    y_axis = st.selectbox('Y Axis', options=grouped.columns.tolist())
                    if graph_type == 'line':
                        fig = px.line(grouped, x=x_axis, y=y_axis)
                    elif graph_type == 'bar':
                        fig = px.bar(grouped, x=x_axis, y=y_axis)
                    elif graph_type == 'scatter':
                        fig = px.scatter(grouped, x=x_axis, y=y_axis)
                    st.plotly_chart(fig)
                elif graph_type == 'pie':
                    names = st.selectbox("Labels", options=grouped.columns.tolist())
                    values = st.selectbox("Values", options=grouped.columns.tolist())
                    fig = px.pie(grouped, names=names, values=values)
                    st.plotly_chart(fig)
                elif graph_type == 'sunburst':
                    path = st.multiselect('Sunburst Path', options=grouped.columns.tolist())
                    if path:
                        fig = px.sunburst(grouped, path=path, values='result')
                        st.plotly_chart(fig)
                        
        # ================== Basic Machine Learning Section ================== #
        st.subheader(":orange[Basic Machine Learning]", divider='green')
        ml_task = st.selectbox("Select ML Task", ["None", "SVM", "Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])
        if ml_task != "None":
            target_col = st.selectbox("Select Target Column", data.columns.tolist(), key='target_ml')
            feature_cols = st.multiselect("Select Feature Columns", data.columns.tolist(), key='features_ml')
            if target_col and feature_cols:
                X = data[feature_cols]
                y = data[target_col]
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X.select_dtypes(include=['object']).columns.tolist()

                # Preprocessing pipelines for numeric and categorical features
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ]
                )
                
                if ml_task == "SVM":
                    classifier = SVC()
                elif ml_task == "Logistic Regression":
                    classifier = LogisticRegression(max_iter=1000)
                elif ml_task == "Decision Tree":
                    classifier = DecisionTreeClassifier()
                elif ml_task == "K-Nearest Neighbors":
                    classifier = KNeighborsClassifier()
                    
                model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.write(f"Model Accuracy: {acc * 100:.2f}%")
                
                # Decision regions (only if exactly 2 features are used)
                if ml_task in ["SVM", "Decision Tree"] and len(feature_cols) == 2:
                    if st.checkbox("Show Decision Regions"):
                        X_combined = pd.concat([X_train, X_test])
                        y_combined = pd.concat([y_train, y_test])
                        fig, ax = plt.subplots()
                        plot_decision_regions(X_combined.values, y_combined.values, clf=model.named_steps['classifier'], legend=2)
                        ax.set_title(f"{ml_task} Decision Regions")
                        st.pyplot(fig)
                
                # Confusion Matrix for classifiers
                if ml_task in ["SVM", "Decision Tree", "Logistic Regression", "K-Nearest Neighbors"]:
                    cm = confusion_matrix(y_test, y_pred)
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                    
       # ================== Model Evaluation Graphs Section ================== #
        st.subheader(":orange[bar_chart: Model Evaluation Graphs]")
        model_selection = st.selectbox('Select The Model', 
                                       ["Linear Regression", "Polynomial Regression", "Decision Tree", "Random Forest", "SVM", "KMeans Clustering" ])
        
        if model_selection:
            target_col = st.selectbox("Select Target Column", data.columns.tolist(), key='target_eval')
            feature_cols = st.multiselect("Select Feature Columns", data.columns.tolist(), key='features_eval')
            
            if target_col and feature_cols:
                X = data[feature_cols]
                y = data[target_col]
                
                # ---------------------- Preprocessing: Encoding Features ---------------------- #
                # Identify numeric and categorical columns in the selected feature set
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
                # Define transformers for both types of features
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])
                # Ensure output as a dense array by setting sparse=False
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot' , LabelEncoding())
                ])
        
                # Combine transformers into a preprocessor
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ],
                    remainder='passthrough'  # In case there are columns not captured above
                )
        
                # Transform the selected features into numeric features
                X_transformed = pd.DataFrame( preprocessor.fit_transform(X) , columns=data.columns())
                
                # ---------------------- Model Evaluation Based On User's Selection ---------------------- #
                if model_selection == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X_transformed, y)
                    y_pred = model.predict(X_transformed)
                    fig = px.scatter(x=y, y=y_pred, 
                                     labels={'x': 'Actual', 'y': 'Predicted'}, 
                                     title="Linear Regression: Actual vs Predicted")
                    st.plotly_chart(fig)
        
                elif model_selection == "Polynomial Regression":
                    degree = st.slider("Select Polynomial Degree", 2, 5)
                    poly = PolynomialFeatures(degree=degree)
                    X_poly = poly.fit_transform(X_transformed)
                    model = LinearRegression()
                    model.fit(X_poly, y)
                    y_pred = model.predict(X_poly)
                    fig = px.scatter(x=y, y=y_pred, 
                                     labels={'x': 'Actual', 'y': 'Predicted'}, 
                                     title="Polynomial Regression Fit")
                    st.plotly_chart(fig)
        
                elif model_selection == "Decision Tree":
                    model = DecisionTreeClassifier()
                    model.fit(X_transformed, y)
                    fig, ax = plt.subplots(figsize=(12, 6))
                    # Note: feature names might not directly match the one–hot encoded features
                    plot_tree(model, filled=True, fontsize=8)
                    st.pyplot(fig)
        
                elif model_selection == "Random Forest":
                    model = RandomForestClassifier()
                    model.fit(X_transformed, y)
                    y_pred = model.predict(X_transformed)
                    cm = confusion_matrix(y, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    st.pyplot(fig)
        
                elif model_selection == "SVM":
                    # SVM decision region visualization supports exactly 2 features after encoding.
                    if X_transformed.shape[1] == 2:
                        model = SVC(kernel='linear')
                        model.fit(X_transformed, y)
                        fig, ax = plt.subplots()
                        plot_decision_regions(X_transformed, y.values, clf=model, legend=2)
                        st.pyplot(fig)
                    else:
                        st.warning("SVM visualization supports only 2 features after encoding. Your selection resulted in "
                                   + str(X_transformed.shape[1]) + " features.")
        
                elif model_selection == "KMeans Clustering":
                    k = st.slider("Select number of clusters (k)", 2, 10)
                    model = KMeans(n_clusters=k)
                    pred = model.fit_predict(X_transformed)
                    # For visualization, check if we have at least two features
                    if X_transformed.shape[1] >= 2:
                        fig = px.scatter(x=X_transformed[:, 0], y=X_transformed[:, 1], color=pred.astype(str),
                                         title="KMeans Clustering", labels={'color': 'Cluster'})
                        st.plotly_chart(fig)
                    else:
                        st.warning("KMeans clustering visualization requires at least 2 features.")
                
                    
    except Exception as e:
        st.error(f"An error occurred: {e}")
