import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from mlxtend.plotting import plot_decision_regions

# Page setup
st.set_page_config(page_title="ML Data Analyzer", layout="wide")
st.title("ML-Based Data Analysis and Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data successfully uploaded!")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Overview", "Preprocessing", "GroupBy", "Modeling", "Evaluation"])

    with tab1:
        st.subheader("Data Overview")
        st.write(df.head())
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write("Info:")
        buffer = df.info(buf=None)
        st.text(buffer)
        st.write("Describe:")
        st.write(df.describe())

        st.subheader("Data Visualization")
        plot_col = st.selectbox("Select column for plotting", df.columns)
        plot_type = st.selectbox("Plot type", ["Histogram", "Boxplot", "Barplot"])
        if plot_type == "Histogram":
            fig = px.histogram(df, x=plot_col)
            st.plotly_chart(fig)
        elif plot_type == "Boxplot":
            fig = px.box(df, y=plot_col)
            st.plotly_chart(fig)
        elif plot_type == "Barplot":
            fig = px.bar(df, x=plot_col)
            st.plotly_chart(fig)

    with tab2:
        st.subheader("Handle Missing & Duplicate Values")
        if df.isnull().sum().sum() > 0:
            st.warning("Missing values found!")
            if st.button("Drop missing values"):
                df.dropna(inplace=True)
                st.success("Missing values dropped.")
        if df.duplicated().sum() > 0:
            st.warning("Duplicate rows found!")
            if st.button("Drop duplicate rows"):
                df.drop_duplicates(inplace=True)
                st.success("Duplicate rows dropped.")

        st.subheader("Correlation Heatmap")
        num_df = df.select_dtypes(include=np.number)
        if not num_df.empty:
            fig, ax = plt.subplots()
            sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    with tab3:
        st.subheader("GroupBy Analysis")
        group_col = st.selectbox("Select column to group by", df.columns)
        agg_func = st.selectbox("Aggregation", ["mean", "sum", "count"])
        if st.button("Group Data"):
            grouped = df.groupby(group_col).agg(agg_func)
            st.write(grouped)
            st.bar_chart(grouped)

    with tab4:
        st.subheader("Modeling")
        target = st.selectbox("Select target column", df.columns)
        features = st.multiselect("Select feature columns", df.columns.drop(target))

        model_choice = st.selectbox("Select Model", ["Logistic Regression", "SVM", "Decision Tree", "KNN", "Linear Regression", "Polynomial Regression"])
        test_size = st.slider("Test size (%)", 10, 50, step=5, value=20)

        if st.button("Train Model"):
            X = df[features]
            y = df[target]

            # Column type handling
            num_cols = X.select_dtypes(include=np.number).columns.tolist()
            cat_cols = X.select_dtypes(include=object).columns.tolist()

            # Preprocessing
            transformers = [
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
            ]
            preprocessor = ColumnTransformer(transformers=transformers)

            # Model initialization
            model_map = {
                "Logistic Regression": LogisticRegression(),
                "SVM": SVC(probability=True),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "Linear Regression": LinearRegression()
            }

            if model_choice == "Polynomial Regression":
                pipeline = Pipeline([
                    ("preprocess", preprocessor),
                    ("poly", PolynomialFeatures(degree=2)),
                    ("model", LinearRegression())
                ])
            else:
                model = model_map[model_choice]
                pipeline = Pipeline([
                    ("preprocess", preprocessor),
                    ("model", model)
                ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                st.success("Model trained successfully!")
                st.write("Accuracy:" if model_choice != "Linear Regression" else "R² Score:",
                         accuracy_score(y_test, y_pred) if model_choice not in ["Linear Regression", "Polynomial Regression"]
                         else r2_score(y_test, y_pred))

                if model_choice in ["Linear Regression", "Polynomial Regression"]:
                    st.write("MSE:", mean_squared_error(y_test, y_pred))
                    fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"})
                    st.plotly_chart(fig)

                # Confusion Matrix for classification
                elif model_choice in ["Logistic Regression", "SVM", "Decision Tree", "KNN"]:
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    st.pyplot(fig)

                    # Decision boundary (only for 2D features)
                    if model_choice == "SVM" and len(features) == 2:
                        st.subheader("SVM Decision Boundary")
                        pre_X = preprocessor.fit_transform(X)
                        fig, ax = plt.subplots()
                        try:
                            plot_decision_regions(pre_X, y.to_numpy(), clf=model)
                            st.pyplot(fig)
                        except:
                            st.warning("Failed to plot decision regions. Try with numeric and binary classes.")

            except Exception as e:
                st.error(f"Model training failed: {e}")

    with tab5:
        st.subheader("Evaluation Summary")
        st.write("For detailed evaluation, use the 'Modeling' tab and observe the metrics and plots shown after training.")
