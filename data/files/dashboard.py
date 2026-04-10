import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os
import sklearn
import connection

if "db_initialized" not in st.session_state:
    connection.create_user_table()   # user table
    connection.create_table()        # data table
    st.session_state["db_initialized"] = True

# WORKS ON LOCAL + CLOUD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # data/files
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) # data

# Model path
model_path = os.path.join(DATA_DIR, "bestt_model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="Churn Prediction", layout="wide")
# Custom Theme via CSS
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #262730;
}

/* Buttons */
.stButton>button {
    background-color: #6C63FF;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
}

/* Input boxes */
.stTextInput>div>div>input, 
.stNumberInput input {
    background-color: #262730;
    color: white;
}

/* Selectbox */
.stSelectbox div {
    background-color: #262730;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
image_path = os.path.join(DATA_DIR,"Images", "churn_logo.jpg")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "logout_msg" in st.session_state:
    st.success(st.session_state["logout_msg"])
    del st.session_state["logout_msg"]

background_img = os.path.join(DATA_DIR,"Images", "Login_Signup.jpg")
st.markdown(f"""
<style>
.stApp {{
    background: url("file:///{background_img}") no-repeat center center fixed;
    background-size: cover;
}}
</style>
""", unsafe_allow_html=True)

if not st.session_state["logged_in"]:
    st.title("🔒 Login/Sign Up")

    menu=["Login","Sign Up"]
    choice=st.selectbox("Select Option",menu)

    username=st.text_input("Username")
    password=st.text_input("Password",type="password")

    if choice=="Sign Up":
        if st.button("Create Account"):
            if not username or not password:
                st.warning("⚠️ Please enter username and password")
            else:
                if connection.signup_user(username, password):
                    st.success("✅ Account created successfully! Please Login.")
                else:
                    st.error("❌ Username already exists")
        
    elif choice=="Login":
        if st.button("Login"):
            if not username or not password:
                st.warning("⚠️ Please enter username and password")
            else:
                user = connection.login_user(username,password)
                if user:
                    st.success("✅Logged in successfully!")
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username 
                    st.rerun()
                else:
                    st.error("❌Invalid Credentials")
    st.stop()  #stop further execution until login

st.sidebar.image(image_path, width="stretch")
st.sidebar.markdown("## 💡 Customer Intelligence & Churn Prediction System")
st.sidebar.markdown("---")
st.sidebar.markdown("## 🚀 Control Center")
page = st.sidebar.radio("🚀 Explore App", [
    "🏠 Home",
    "📊 Data Overview",
    "📈Exploratory Data Analysis(EDA)",
    "📌 Feature Insights",
    "📌 Customer Segments",
    "✔️Prediction",
    "📂 Saved Data",
    "📊 Model Performance"
])
if st.sidebar.button("🚪 Logout"):
    st.session_state["logged_in"] = False
    st.session_state["logout_msg"] = "👋 Logged out successfully!"
    st.rerun()
st.sidebar.info("""
👩‍💻 Built by: Ramandeep Chounkaria
""")
st.sidebar.markdown("---")
st.sidebar.markdown("📂 [GitHub Repo](https://github.com/rd966443-web/ProjectChurn.git)")
st.sidebar.markdown("🔗 [Streamlit Community Cloud](https://rd966443-web-projectchurn-datafilesdashboard-pjhqtu.streamlit.app/)")
st.sidebar.caption("© 2026 Churn Dashboard")

#Home Page
if page == "🏠 Home":
    st.markdown("""
    # 🚀Customer Intelligence & Churn Prediction System
    ### 🔍 Predict • 📊 Analyze • 💡 Retain Customers

    ---
    """)
    st.info("This dashboard helps analyze customer behaviour and predict customer churn.")
    st.caption("💡 Use the sidebar to navigate through different sections.")
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("""
        This system helps businesses:
        - 📉 Reduce Churn Rate
        - 🎯 Identify High-Risk Customers
        - 📊 Understand Customer Behaviour
        """)
    with col2:
        st.markdown("---")
        st.write("**Know your Customers, Stop the Churn**")
        st.markdown("---")
    home_img = os.path.join(DATA_DIR,"Images", "Customer_Churn.png")
    st.image(home_img)

#Data Overview

elif page == "📊 Data Overview":
    st.title("📊 Data Overview")

    st.info("Enter customer details carefully. This data will be used for churn prediction.")
    st.caption("💡 Accurate input gives better prediction results.")

    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age=st.number_input("Age",0,100)
    SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    Tenure = st.slider("Tenure", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 10000.0, 100.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 100000.0, 1000.0)

    st.session_state["input_data"] =({
        "Gender": Gender,
        "Age":Age,
        "SeniorCitizen":1 if SeniorCitizen == "Yes" else 0,
        "Partner":Partner,
        "Dependents": Dependents,
        "Tenure": Tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection":DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    })
    if st.button("💾 Save Data for Prediction"):
        st.success("✅ Data captured successfully!")
        st.info("👉 For check the prediction, now go to Prediction page")
    
#EDA
elif page == "📈Exploratory Data Analysis(EDA)":
    st.title("📈Exploratory Data Analysis(EDA)")

    st.info("Explore patterns and trends in customer data to understand churn behaviour.")
    st.caption("💡 Look for relationships between features and churn.")

    # Load dataset
    final_data_path = os.path.join(DATA_DIR,"processed", "final_dataset.csv")
    merged_data_path = os.path.join(DATA_DIR,"processed", "merged_data.csv")
    final_data= pd.read_csv(final_data_path) 
    merged_data= pd.read_csv(merged_data_path)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(final_data))
    col2.metric("Churn Rate", f"{final_data['Churn'].mean()*100:.2f}%")
    col3.metric("Avg Monthly Charges", f"{final_data['MonthlyCharges'].mean():.2f}")
    sns.set_style("whitegrid")
    # Tabs Start Here
    tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Charts", "💻Workflow"])

    # TAB 1 → Data
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(merged_data.head(), width="stretch")

        st.download_button(
        label="📥 Download Full Dataset",
        data=merged_data.to_csv(index=False),
        file_name='churn_data.csv',
        mime='text/csv'
    )

    #TAB 2 → Charts
    with tab2:
        st.subheader("📊 Churn Distribution")
        churn_counts = final_data["Churn"].map({0: "No Churn", 1: "Churn"}).value_counts()
        st.bar_chart(churn_counts)

        st.subheader("👥 Gender vs Churn")
        if final_data["Gender"].dtype != "object":
            final_data["Gender"] = final_data["Gender"].map({0: "Female", 1: "Male"})
        gender_churn = pd.crosstab(final_data["Gender"], final_data["Churn"])
        gender_churn.columns = ["No Churn", "Churn"]
        st.bar_chart(gender_churn)

        #tenure more churn less
        st.subheader("⏳ Tenure vs Churn")
        tenure_churn = pd.crosstab(final_data["Tenure"], final_data["Churn"])
        tenure_churn.columns = ["No Churn", "Churn"]
        st.line_chart(tenure_churn)

         # Monthly Charges Distribution
        import matplotlib.pyplot as plt
        st.subheader("📊 Monthly Charges Distribution")
        fig, ax = plt.subplots()
        ax.hist(final_data["MonthlyCharges"], bins=30)
        st.pyplot(fig)

        st.subheader("📄 Contract vs Churn") 
        #har row ko dekhkar actual contract nikalega
        # Convert one-hot → single Contract column-->or churn no churn btayega
        def get_contract(row):
            if row["Contract_One year"] == 1:
                return "One Year"
            elif row["Contract_Two year"] == 1:
                return "Two Year"
            else:
                return "Month-to-Month"
        final_data["Contract_Type"] = final_data.apply(get_contract, axis=1)
        contract_churn = pd.crosstab(final_data["Contract_Type"], final_data["Churn"])
        contract_churn.columns = ["No Churn", "Churn"]
        st.bar_chart(contract_churn)

        st.subheader("🔥 Correlation Heatmap")
        temp_data = final_data.copy()
        for col in temp_data.select_dtypes(include='object').columns:
            temp_data[col] = temp_data[col].astype('category').cat.codes
        fig, ax = plt.subplots()
        sns.heatmap(temp_data.corr(), annot=False, linewidths=0.5)
        st.pyplot(fig)
               

    #TAB 3 →Workflow
    with tab3:
        st.write("""
            Data → Preprocessing → Model → Prediction → Output
                """)
        st.success("✅ Now, Go to the Feature Insights Page")
   
# Feature Insights Page->model konse feature k basis par pred kr rha hai 

elif page == "📌 Feature Insights":
    st.title("📌 Feature Importance & Insights")

    st.info("This section shows which features influence the model's prediction the most.")
    st.caption("💡 Higher importance means stronger impact on churn.")

    # Load dataset
    final_data_path = os.path.join(DATA_DIR,"processed", "final_dataset.csv")
    final_data= pd.read_csv(final_data_path) 
    final_data.columns = final_data.columns.str.strip()

    # Drop unnecessary columns safely
    X = final_data.drop(columns=['Churn', 'Customer ID'],errors='ignore')
    y = final_data['Churn']

    # Extract pipeline parts
    try:
        preprocessor = model.named_steps['preprocessing']
        final_model = model.named_steps['model']
    except:
        st.error("❌Model pipeline structure not recognized.")
        st.stop()
    
     # Align columns BEFORE transform
    try:
        required_cols = preprocessor.feature_names_in_
        for col in required_cols:
            if col not in X.columns:
                X[col] = 0
        # reorder columns
        X = X[required_cols]

        #transform 
        X_transformed = preprocessor.transform(X)

    except Exception as e:
        st.error(f"❌ Preprocessing failed: {e}")
        st.stop()

    #feature importance
    if hasattr(final_model, "feature_importances_"):
        # Get feature names after encoding
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]

        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': final_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.subheader("🌟 Top Feature Importance")
        st.bar_chart(importance.head(20).set_index('Feature'))
    else:
        st.info("Feature importance not available for this model.")

# Customer Segments Page

elif page == "📌 Customer Segments":
    st.title("📌 Customer Segmentation (KMeans Clustering)")

    st.info("Customers are grouped into clusters based on similar behavior.")
    st.caption("💡 Use this to identify different customer types.")

    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    # Load dataset
    final_data_path = os.path.join(DATA_DIR,"processed", "final_dataset.csv")
    merged_data_path = os.path.join(DATA_DIR,"processed", "merged_data.csv")
    final_data= pd.read_csv(final_data_path) 
    raw_data= pd.read_csv(merged_data_path)
    
    # Clean column names
    final_data.columns = final_data.columns.str.strip()
    
    n_clusters = st.slider("Number of Clusters", 2, 22, 4)

    available_features = list(raw_data.columns)
    default_features = [col for col in ["TotalCharges", "MonthlyCharges"] if col in available_features]
     # Feature selection
    features = st.multiselect(
        "Select Features for Clustering",
        options=available_features,
        default=default_features
    )
    if st.button("🚀 Run Clustering"):

        if len(features) < 2:
            st.warning("⚠️ Please select at least 2 features")

        else:
            # Raw data
            display_data = raw_data[features].copy()

            # Data for clustering
            cluster_data = final_data[features].copy()

            # Drop ID if exists
            if 'Customer ID' in cluster_data.columns:
                cluster_data = cluster_data.drop(columns=['Customer ID'])

            # Identify columns
            num_cols = cluster_data.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = cluster_data.select_dtypes(include=['object']).columns

            # Preprocessing pipeline
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ])

            processed_data = preprocessor.fit_transform(cluster_data)

            # KMeans Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(processed_data)


            #Align rows before assigning clusters
            display_data = display_data.loc[cluster_data.index].copy()
            # Add cluster labels to display data
            display_data["Cluster"] = clusters

            # Visualization
            st.subheader("📊 Cluster Distribution")
            st.bar_chart(display_data["Cluster"].value_counts())

            # Clean Preview 
            st.subheader("📌 Clustered Data Preview")
            st.dataframe(display_data.head(10), width="stretch")

            # Scatter Plot (if 2 features)
            if len(features) == 2:

                fig, ax = plt.subplots()
                ax.scatter(display_data[features[0]], display_data[features[1]], c=clusters)
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.set_title("Customer Segments")

                st.pyplot(fig)

            # 💡 Cluster-wise Sample View
            st.subheader("🔍 Cluster-wise Samples")

            for i in range(n_clusters):
                st.write(f"### Cluster {i}")
                st.dataframe(display_data[display_data["Cluster"] == i].head(5))
            st.success("✅ Now, Go to the Prediction Page for check the Prediction")


# prediction

elif page=="✔️Prediction":
    st.title("✔️Customer Churn Prediction")

    st.info("Predict whether a customer is likely to churn based on input data.")
    st.caption("💡 Higher probability means higher churn risk.")

    # Show which model is being used
    st.markdown("🧠 Prediction Model: **AdaBoostClassifier (Best Model)**")

    # Check if data exists
    if "input_data" not in st.session_state:
        st.warning("⚠️ Please enter data first in 📊 Data Overview page") 
    else:
        st.subheader("📌 Saved Customer Data")
        st.write(st.session_state["input_data"])
        if st.button("🚀 Predict Churn"):
            input_df = pd.DataFrame([st.session_state["input_data"]])
            # Align with model features
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        
            with st.spinner("🔄 Processing..."):
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
            
            #Database Save
            try:
                connection.insert_data(
                   st.session_state["username"],
                   st.session_state["input_data"],
                   int(prediction),
                   float(probability)
                )
                st.success("✅ Data saved to database successfully!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

            st.subheader("📊 Prediction Result")
            # Result UI
            if prediction == 1:
                st.error(f"⚠️ Customer is likely to CHURN")
                st.markdown("### 🔴 High Churn Risk")
            else:
                st.success(f"✅ Customer is likely to STAY")
                st.markdown("### 🟢 Low Churn Risk")
            st.write(f"📈 Churn Probability: **{probability:.2f}**")
            st.progress(int(probability * 100))

            # Risk Level Messages
            if probability > 0.7:
                st.error("🔥 High Risk Customer! Take immediate action")
            elif probability > 0.4:
                st.warning("⚡ Medium Risk - Monitor customer")
            else:
                st.success("💚 Low Risk Customer")
            
            if prediction == 0:
              st.success("🎉 Great! Customer is safe (No Churn)")
              st.balloons()
 
elif page == "📂 Saved Data":
    st.title("📂 Saved Customer Data in Database")

    st.info("View and manage all saved customer predictions.")
    st.caption("💡 You can update, search, or delete records.")

    # Show data
    df = connection.fetch_data(st.session_state["username"])

    if not df.empty:
        st.dataframe(df,width="stretch")
         #Download Button
        st.download_button(
        label="📥 Download Data as CSV",
        data=df.to_csv(index=False),
        file_name="customer_data.csv",
        mime="text/csv"
    )
    else:
        st.warning("No data found")
    st.markdown("---")
    
    st.markdown("## 📊 Live Dashboard Stats")
    total, avg_prob, high_risk = connection.get_stats(st.session_state["username"])
    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Customers", total)
    col2.metric("📈 Avg Probability", f"{avg_prob:.2f}" if avg_prob else "0.00")
    col3.metric("⚠️ High Risk Customers", high_risk)
    st.markdown("---")

    st.subheader("✏️ Update Prediction")
    update_id = st.number_input("Enter Customer ID", min_value=1, step=1,help="Enter ID from table above")
    new_prediction = st.selectbox("Prediction", [0, 1])
    new_probability = st.slider("Probability", 0.0, 1.0, 0.5)
    if st.button("Update", key="update_btn"):
        result=connection.update_prediction(st.session_state["username"],update_id, new_prediction, new_probability)
        if result:
           st.success("✅ Record updated successfully")
           st.toast("Data refreshed 🔄")
           st.rerun()  
        else:
           st.error("❌ CustomerID not found")
    
    st.subheader("🔍 Search Data")
    keyword = st.text_input("Search by Gender or Payment Method or Contract or InternetService")
    if st.button("Search",key="search_btn"):
        results = connection.search_data(st.session_state["username"], keyword)
        if not results.empty:
            st.success(f"🔍 Found {len(results)} records")
            st.dataframe(results,width="stretch")
        else:
            st.warning("No results found")

    st.subheader("🗑️ Delete Record")
    delete_id = st.number_input("Enter Customer ID to Delete", min_value=1, step=1,help="Enter ID from table above", key="delete_id")
    if st.button("Delete", key="delete_btn"):
        result = connection.delete_data(st.session_state["username"], delete_id)
        if result:
            st.warning("🗑️ Record deleted successfully")
            st.rerun()
        else:
            st.error("❌ ID not found")

# Model Performance Page

elif page == "📊 Model Performance":
    st.title("📊 Model Performance Metrics")

    st.info("Evaluate how well the model performs on the dataset.")
    st.caption("💡 Focus on accuracy, precision, recall, and ROC-AUC.")
   
    from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,f1_score,recall_score
    from sklearn.metrics import confusion_matrix,classification_report

    # Load data
    final_data_path = os.path.join(DATA_DIR,"processed", "final_dataset.csv")
    data= pd.read_csv(final_data_path) 
    data.columns = data.columns.str.strip()#remove extra spaces

    # Separate features & target
    X = data.drop(columns=['Customer ID', 'Churn'], errors='ignore')
    y_true = data['Churn'] 
    
    #Match training columns
    try:
        expected_cols = model.feature_names_in_
        X = X.reindex(columns=expected_cols)
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
             X[col] = X[col].fillna(0)
            else:
              X[col] = X[col].fillna("Unknown")
    except:
        st.warning("⚠️ Could not fetch model feature names. Ensure training columns match.")

    # Predictions
    y_pred = model.predict(X)

    # Handle models without predict_proba
    try:
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    except:
        y_prob = None
        auc = None
   
    st.subheader("📊 Model Performance")

    accuracy = accuracy_score(y_true, y_pred)#total ryt pred
    precision=precision_score(y_true, y_pred, zero_division=0)#churn predicted mein shi kitne
    recall = recall_score(y_true, y_pred, pos_label=1)#actual churn mein se kitne
    f1 = f1_score(y_true, y_pred, pos_label=1)#dono ka balance

    if auc is not None:
        st.write(f"📈 ROC-AUC Score: {auc:.4f}")

    # Confusion Matrix
    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)
    st.dataframe(pd.DataFrame(cm,
                              index=["Actual No", "Actual Yes"],
                              columns=["Predicted No", "Predicted Yes"]))

    # Classification Report
    st.subheader("📄 Classification Report")
    report = classification_report(y_true, y_pred, zero_division=0)
    st.text(report)

    st.download_button(
    label="📥 Download Predictions",
    data=pd.DataFrame({"Actual": y_true, "Predicted": y_pred}).to_csv(index=False),
    file_name='predictions.csv',
    mime='text/csv'
)








   

    

  







   

    

  


