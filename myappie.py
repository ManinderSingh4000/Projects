# _____________ Import Python Libraries _________________ #

import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
        ml_task = st.selectbox("Select ML Task", ["None", "SVM", "Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])

        if ml_task != "None":
            target_col = st.selectbox("Select Target Column", data.columns)
            feature_cols = st.multiselect("Select Feature Columns", data.columns)

            if target_col and feature_cols:
                X = data[feature_cols]
                y = data[target_col]

                # Handle Preprocessing (Categorical and Numeric Data)
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                categorical_features = X.select_dtypes(include=['object']).columns

                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing data
                    ('scaler', StandardScaler())  # Normalize numerical data
                ])

                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing data
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-Hot Encode categorical features
                ])

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ]
                )

                # Create model pipeline based on selected task
                if ml_task == "SVM":
                    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC())])
                elif ml_task == "Logistic Regression":
                    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression())])
                elif ml_task == "Decision Tree":
                    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier())])
                elif ml_task == "K-Nearest Neighbors":
                    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', KNeighborsClassifier())])

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Evaluate the model
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        st.subheader(":bar_chart: Model Evaluation Graphs")
        model_selection = st.selectbox(' Select The Model :', ['Linear Regression' ,'Polynomial Regression','SVM','Decision Tree','KMeans Clustering'])

        # from sklearn.linear_model import LinearRegression
        # import seaborn as sns
        
        
        # if model_selection == "Linear Regression":
        #     model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
        #     model.fit(X_train, y_train)
        #     y_pred = model.predict(X_test)
        #     accuracy = model.score(X_test, y_test)
        
        #     st.write(f"Model R² Score: {accuracy * 100:.2f}%")
        
        #     fig, ax = plt.subplots()
        #     ax.scatter(y_test, y_pred)
        #     ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        #     ax.set_xlabel("Actual")
        #     ax.set_ylabel("Predicted")
        #     st.pyplot(fig)

        # from sklearn.preprocessing import PolynomialFeatures
        # from sklearn.pipeline import make_pipeline

        # if model_selection == "Polynomial Regression":
        #     degree = st.slider("Choose Degree", 2, 10, 2)
        #     model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        #     model.fit(X_train, y_train)
        #     y_pred = model.predict(X_test)
        
        #     fig, ax = plt.subplots()
        #     ax.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')
        #     ax.scatter(X_test.iloc[:, 0], y_pred, color='red', label='Predicted')
        #     ax.set_title('Polynomial Fit')
        #     ax.legend()
        #     st.pyplot(fig)

        # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        # import seaborn as sns
        
        # cm = confusion_matrix(y_test, y_pred)
        # st.subheader("Confusion Matrix")
        
        # fig, ax = plt.subplots()
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
        # ax.set_xlabel('Predicted')
        # ax.set_ylabel('Actual')
        # st.pyplot(fig)

        # from mlxtend.plotting import plot_decision_regions

        # if model_selection == "SVM" and len(feature_cols) == 2:
        #     X_combined = pd.concat([X_train, X_test])
        #     y_combined = pd.concat([y_train, y_test])
        
        #     model.fit(X_combined, y_combined)
        #     fig, ax = plt.subplots()
        #     plot_decision_regions(X_combined.values, y_combined.values, clf=model.named_steps['classifier'], legend=2)
        #     ax.set_title("SVM Decision Regions")
        #     st.pyplot(fig)

        # from sklearn.tree import plot_tree

        # if model_selection == "Decision Tree":
        #     fig, ax = plt.subplots(figsize=(15, 10))
        #     plot_tree(model.named_steps['classifier'], filled=True, feature_names=feature_cols, class_names=True)
        #     st.pyplot(fig)
        
        # elif model_selection == "Random Forest":
        #     fig, ax = plt.subplots(figsize=(15, 10))
        #     plot_tree(model.named_steps['classifier'].estimators_[0], filled=True, feature_names=feature_cols, class_names=True)
        #     st.pyplot(fig)

        # from sklearn.cluster import KMeans
        # from sklearn.decomposition import PCA
        
        # if model_selection == "KMeans Clustering":
        #     n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        #     kmeans = KMeans(n_clusters=n_clusters)
        #     X_scaled = StandardScaler().fit_transform(X)
        #     y_kmeans = kmeans.fit_predict(X_scaled)
        
        #     pca = PCA(n_components=2)
        #     reduced = pca.fit_transform(X_scaled)
        
        #     fig, ax = plt.subplots()
        #     scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=y_kmeans, cmap='viridis')
        #     ax.set_title("Cluster Visualization (PCA)")
        #     st.pyplot(fig)
        

        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.pipeline import make_pipeline, Pipeline
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
        from sklearn.svm import SVC
        from sklearn.tree import plot_tree
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from mlxtend.plotting import plot_decision_regions
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import streamlit as st
        
        # Common preprocessing
       from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Common preprocessing
if model_selection in ["Linear Regression", "Polynomial Regression", "SVM", "Decision Tree", "Random Forest", "KMeans Clustering"]:
    if 'target' in df.columns:
        y = df['target']
        X = df.drop(columns='target')
    else:
        st.warning("Please ensure your dataset has a column named 'target'")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
if model_selection == "Linear Regression":
    model = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', LinearRegression())])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    st.success(f"Model R² Score: {r2 * 100:.2f}%")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

# Polynomial Regression
elif model_selection == "Polynomial Regression":
    degree = st.slider("Choose Degree", 2, 10, 2)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.success("Polynomial Regression trained.")

    fig, ax = plt.subplots()
    ax.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')
    ax.scatter(X_test.iloc[:, 0], y_pred, color='red', label='Predicted')
    ax.set_title('Polynomial Fit')
    ax.legend()
    st.pyplot(fig)

# SVM (Only 2 features)
elif model_selection == "SVM":
    if len(X.columns) != 2:
        st.warning("SVM visualization only supports 2 feature columns.")
    else:
        model = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', SVC(kernel='linear'))])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.success(f"SVM Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

        if st.checkbox("Show Decision Regions"):
            X_combined = pd.concat([X_train, X_test])
            y_combined = pd.concat([y_train, y_test])
            fig, ax = plt.subplots()
            plot_decision_regions(X_combined.values, y_combined.values, clf=model.named_steps['classifier'], legend=2)
            ax.set_title("SVM Decision Regions")
            st.pyplot(fig)

# Decision Tree
elif model_selection == "Decision Tree":
    model = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', DecisionTreeClassifier())])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.success(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(15, 10))
    plot_tree(model.named_steps['classifier'], filled=True, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)])
    st.pyplot(fig)

# Random Forest
elif model_selection == "Random Forest":
    model = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', RandomForestClassifier(n_estimators=100))])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.success(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(15, 10))
    plot_tree(model.named_steps['classifier'].estimators_[0], filled=True, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)])
    st.pyplot(fig)

# KMeans Clustering
elif model_selection == "KMeans Clustering":
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters)
    y_kmeans = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=y_kmeans, cmap='viridis')
    ax.set_title("Cluster Visualization (PCA)")
    st.pyplot(fig)

# Optional Confusion Matrix (Only for classifiers)
if model_selection in ["SVM", "Decision Tree", "Random Forest"]:
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Common preprocessing
if model_selection in ["Linear Regression", "Polynomial Regression", "SVM", "Decision Tree", "Random Forest", "KMeans Clustering"]:
    if 'target' in df.columns:
        y = df['target']
        X = df.drop(columns='target')
    else:
        st.warning("Please ensure your dataset has a column named 'target'")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
if model_selection == "Linear Regression":
    model = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', LinearRegression())])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    st.success(f"Model R² Score: {r2 * 100:.2f}%")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

# Polynomial Regression
elif model_selection == "Polynomial Regression":
    degree = st.slider("Choose Degree", 2, 10, 2)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.success("Polynomial Regression trained.")

    fig, ax = plt.subplots()
    ax.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')
    ax.scatter(X_test.iloc[:, 0], y_pred, color='red', label='Predicted')
    ax.set_title('Polynomial Fit')
    ax.legend()
    st.pyplot(fig)

# SVM (Only 2 features)
elif model_selection == "SVM":
    if len(X.columns) != 2:
        st.warning("SVM visualization only supports 2 feature columns.")
    else:
        model = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', SVC(kernel='linear'))])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.success(f"SVM Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

        if st.checkbox("Show Decision Regions"):
            X_combined = pd.concat([X_train, X_test])
            y_combined = pd.concat([y_train, y_test])
            fig, ax = plt.subplots()
            plot_decision_regions(X_combined.values, y_combined.values, clf=model.named_steps['classifier'], legend=2)
            ax.set_title("SVM Decision Regions")
            st.pyplot(fig)

# Decision Tree
elif model_selection == "Decision Tree":
    model = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', DecisionTreeClassifier())])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.success(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(15, 10))
    plot_tree(model.named_steps['classifier'], filled=True, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)])
    st.pyplot(fig)

# Random Forest
elif model_selection == "Random Forest":
    model = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', RandomForestClassifier(n_estimators=100))])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.success(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(15, 10))
    plot_tree(model.named_steps['classifier'].estimators_[0], filled=True, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)])
    st.pyplot(fig)

# KMeans Clustering
elif model_selection == "KMeans Clustering":
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters)
    y_kmeans = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=y_kmeans, cmap='viridis')
    ax.set_title("Cluster Visualization (PCA)")
    st.pyplot(fig)

# Optional Confusion Matrix (Only for classifiers)
if model_selection in ["SVM", "Decision Tree", "Random Forest"]:
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)


                
    except Exception as e:
        st.error(f"An error occurred: {e}")
