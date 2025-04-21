
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

st.title(":red[Data] :blue[Analytic] :orange[ & Machine Learning Portal]")
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
                if ml_task in ["SVM"] and len(feature_cols) == 2:
                    if st.checkbox("Show Decision Regions"):
                        X_processed = preprocessor.fit_transform(X)
                        X_train_processed, X_test_processed, y_train_processed, y_test_processed = train_test_split(
                            X_processed, y, test_size=0.2, random_state=42
                        )
                        model_decision_boundary = SVC(kernel='linear') # Use a simpler kernel for visualization
                        model_decision_boundary.fit(X_train_processed, y_train_processed)

                        X_combined_processed = np.vstack((X_train_processed, X_test_processed))
                        y_combined_processed = np.hstack((y_train_processed, y_test_processed))

                        fig, ax = plt.subplots()
                        plot_decision_regions(X_combined_processed, y_combined_processed, clf=model_decision_boundary, legend=2)
                        ax.set_xlabel(f'Feature 1 (Processed)') # Add meaningful labels
                        ax.set_ylabel(f'Feature 2 (Processed)')
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
        st.subheader(":orange[ Model Evaluation Graphs]")
        model_selection = st.selectbox('Select The Model',
                                         ["Linear Regression", "Polynomial Regression", "Decision Tree", "SVM", "KMeans Clustering"])

        if model_selection:
            target_col_eval = st.selectbox("Select Target Column", data.columns.tolist(), key='target_eval')
            feature_cols_eval = st.multiselect("Select Feature Columns", data.columns.tolist(), key='features_eval')

            if feature_cols_eval:
                X_eval = data[feature_cols_eval]
                if model_selection != "KMeans Clustering":
                    y_eval = data[target_col_eval]

                # ---------------------- Preprocessing: Encoding Features ---------------------- #
                numeric_features_eval = X_eval.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features_eval = X_eval.select_dtypes(include=['object', 'category']).columns.tolist()

                # Define transformers for both types of features
                numeric_transformer_eval = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])

                categorical_transformer_eval = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])

                preprocessor_eval = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer_eval, numeric_features_eval),
                        ('cat', categorical_transformer_eval, categorical_features_eval)
                    ],
                    remainder='passthrough'
                )

                X_transformed_eval_array = preprocessor_eval.fit_transform(X_eval)
                X_transformed_eval = pd.DataFrame(X_transformed_eval_array)


                # ---------------------- Model Evaluation Based On User's Selection ---------------------- #
                if model_selection == "Linear Regression":
                    model_eval = LinearRegression()
                    model_eval.fit(X_transformed_eval, y_eval)
                    y_pred_eval = model_eval.predict(X_transformed_eval)
                    fig_eval = px.scatter(x=y_eval, y=y_pred_eval,
                                            labels={'x': 'Actual', 'y': 'Predicted'},
                                            title="Linear Regression: Actual vs Predicted")
                    st.plotly_chart(fig_eval)

                elif model_selection == "Polynomial Regression":
                    degree = st.slider("Select Polynomial Degree", 2, 5, key='poly_degree_eval')
                    poly_eval = PolynomialFeatures(degree=degree)
                    X_poly_eval = poly_eval.fit_transform(X_transformed_eval)
                    model_eval = LinearRegression()
                    model_eval.fit(X_poly_eval, y_eval)
                    y_pred_eval = model_eval.predict(X_poly_eval)
                    fig_eval = px.scatter(x=y_eval, y=y_pred_eval,
                                            labels={'x': 'Actual', 'y': 'Predicted'},
                                            title="Polynomial Regression Fit")
                    st.plotly_chart(fig_eval)

                elif model_selection == "Decision Tree":
                     model_eval = DecisionTreeClassifier(max_depth=3)
                     model_eval.fit(X_transformed_eval, y_eval)
                     fig_eval, ax_eval = plt.subplots(figsize=(12, 6))
                     plot_tree(model_eval, filled=True, fontsize=8)
                     st.pyplot(fig_eval)


                elif model_selection == "SVM":
                    # SVM decision region visualization supports exactly 2 features
                    if X_transformed_eval.shape[1] == 2:
                        try:
                            # Ensure inputs are NumPy arrays
                            X_array_eval = X_transformed_eval if isinstance(X_transformed_eval, np.ndarray) else X_transformed_eval.to_numpy()
                            y_array_eval = y_eval if isinstance(y_eval, np.ndarray) else y_eval.to_numpy().ravel()

                            # Convert target to integers if needed (plot_decision_regions prefers int labels)
                            if y_array_eval.dtype != int:
                                y_array_eval = y_array_eval.astype(int)

                            # Fit the SVM model
                            model_eval_svm = SVC(kernel='linear')
                            model_eval_svm.fit(X_array_eval, y_array_eval)

                            # Plot decision regions
                            fig_svm, ax_svm = plt.subplots()
                            plot_decision_regions(X_array_eval, y_array_eval, clf=model_eval_svm, legend=2)
                            ax_svm.set_xlabel(f'Feature 1 (Processed)') # Add meaningful labels
                            ax_svm.set_ylabel(f'Feature 2 (Processed)')
                            ax_svm.set_title(f"SVM Decision Regions")
                            st.pyplot(fig_svm)
                        except Exception as e:
                            st.error(f"Error during SVM visualization: {e}")
                    else:
                        st.warning(f"SVM visualization supports only 2 features. Your selection resulted in {X_transformed_eval.shape[1]} features.")

                elif model_selection == "KMeans Clustering":
                    feature_cols = st.multiselect(
                        "Select Feature Columns",
                        data.columns.tolist(),
                        key='features_kmeans'
                    )
                    
                    if feature_cols:
                        X = data[feature_cols]
            
                        k = st.slider("Select number of clusters (k)", 2, 10 )
                        model = KMeans(n_clusters=k)
                        pred = model.fit_predict(X)
            
                        # For a simple 2D scatter plot, we need at least two columns
                        if len(feature_cols) >= 2:
                            fig = px.scatter(
                                x=X[feature_cols[0]],
                                y=X[feature_cols[1]],
                                color=pred.astype(str),
                                title="KMeans Clustering"
                            )
                            st.plotly_chart(fig)
                        else:
                            st.warning("Select at least 2 features for a 2D scatter plot.")
                            st.plotly_chart(fig_kmeans)
            #         else:
            #             st.warning("Select at least 2 features for KMeans visualization.")
            #     else:
            #         st.info("Select feature columns for KMeans.")
            # else:
            #     st.info("Select target and feature columns for model evaluation.")

    except Exception as e:
        st.error(f"An error occurred: {e}")


