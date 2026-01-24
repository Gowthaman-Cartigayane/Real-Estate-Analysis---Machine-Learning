import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('india_housing_prices.csv')
pages=st.sidebar.radio("Select Your Pagae for Navigation", ['Introduction and view Table' ,'Exploratory Data Analysis',
                                                            'Investment Analysis', 'Explanation of Machine Learning'])

if pages == 'Introduction and view Table':
    st.title("Real Estate Investment Advisor")
    st.header("Introduction")
    st.text("This is a Real Estate Advisory platform built using Pandas, Streamlit, and Exploratory Data Analysis (EDA). It empowers users to navigate market data and determine whether a property represents a sound investment opportunity.")
    st.header("View Database")
    st.write(data)

elif pages == 'Exploratory Data Analysis':
    st.header('Exploratory Data Analysis')
    st.text("The process of interrogating datasets to summarize their main characteristics, often using visual methods.")

    plot = st.selectbox("Select any one option", ['What is the distribution of property prices?' ,'What is the distribution of property sizes?',
                                                  'How does the price per sq ft vary by property type?', 'Is there a relationship between property size and price?',
                                                  'Are there any outliers in price per sq ft or property size?',
                                                  'What is the average price per sq ft by state?',
                                                  'What is the average property price by city?',
                                                  'What is the median age of properties by locality?',
                                                  'How is BHK distributed across cities?',
                                                  'What are the price trends for the top 5 most expensive localities?',
                                                  'How do nearby schools relate to price per sq ft',
                                                  'How do nearby hospitals relate to price per sq ft?',
                                                  'How does price vary by furnished status?',
                                                  'How does price per sq ft vary by property facing direction?',
                                                  'How many properties belong to each owner type',
                                                  'How many properties are available under each availability status?',
                                                  'Does parking space affect property price?',
                                                  'How do amenities affect price per sq ft?',
                                                  'How does public transport accessibility relate to price per sq ft or investment potential?'])
    
    if plot == 'What is the distribution of property prices?':
        fig, ax=plt.subplots()
        plt.hist(data['Price_per_SqFt'], bins=20, color='skyblue', edgecolor='black')
        plt.title("Distribution of Property Prices")
        plt.xlabel("Price per SqFt")
        plt.ylabel("Frequency")
        st.pyplot(fig)
    
    elif plot == 'What is the distribution of property sizes?':
        fig, ax = plt.subplots()
        sns.kdeplot(data['Size_in_SqFt'], fill=True)
        plt.title('Distribution of Property Sizes')
        st.pyplot(fig)
    
    elif plot == 'How does the price per sq ft vary by property type?':
        fig, ax = plt.subplots()
        sns.barplot(data=data, x='Property_Type', y='Price_per_SqFt', palette='Set2')
        plt.title('Price per SqFt by Property Type')
        plt.xlabel('Property Type')
        plt.ylabel('Price per SqFt')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    elif plot == 'Is there a relationship between property size and price?':
        fig , ax = plt.subplots()
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.regplot(
            x=data['Size_in_SqFt'],
            y=data['Price_in_Lakhs'],
            data=data,
            scatter_kws={'alpha': 0.6},
            line_kws={'color': 'red'} )
        plt.title('Relationship between Property Size and Price')
        st.pyplot(fig)

    elif plot == 'Are there any outliers in price per sq ft or property size?':
        fig , ax = plt.subplots()
        sns.boxplot(x=data['Price_per_SqFt'])
        plt.title('Box Plot of Price per Square Feet')
        plt.xlabel('Price per Square Feet')
        st.pyplot(fig)

    elif plot == 'What is the average price per sq ft by state?':
        fig, ax = plt.subplots()
        state_avg = data.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False).reset_index()
        sns.barplot(data=state_avg, x='Price_per_SqFt', y='State', palette='viridis')
        st.pyplot(fig)

    elif plot == 'What is the average property price by city?':
        fig, ax = plt.subplots()
        price_avg=data.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).reset_index()
        plt.figure(figsize=(10, 12))
        sns.barplot(data=price_avg, x='Price_in_Lakhs', y='City', palette='viridis')
        st.pyplot(fig)

    elif plot == 'What is the median age of properties by locality?':
        fig, ax = plt.subplots()
        locality_median = data.groupby('Locality')['Age_of_Property'].median().sort_values().reset_index()
        plt.figure(figsize=(10, 12))
        sns.barplot(data=locality_median, x='Locality', y='Age_of_Property', palette='viridis')
        st.pyplot(fig)

    elif plot == 'How is BHK distributed across cities?':
        fig, ax = plt.subplots()
        ct = pd.crosstab(data['City'], data['BHK'])
        ct_pct = ct.div(ct.sum(1), axis=0) * 100
        ax = ct_pct.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
        st.pyplot(fig)


    elif plot == ' What are the price trends for the top 5 most expensive localities?':
        fig, ax = plt.subplots()
        top_5_localities = data.groupby('Locality')['Price_in_Lakhs'].mean().nlargest(5).index

        filtered_data = data[data['Locality'].isin(top_5_localities)]

        price_trends = filtered_data.groupby(['Locality', 'Year_Built'])['Price_in_Lakhs'].mean().reset_index()

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=price_trends, x='Year_Built', y='Price_in_Lakhs', hue='Locality', marker='o', linewidth=2.5)
        plt.title('Price Trends for Top 5 Most Expensive Localities by Year Built')
        plt.xlabel('Year Built')
        plt.ylabel('Average Price in Lakhs')
        plt.grid(True)
        plt.legend(title='Locality')
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)

    elif plot ==  'How do nearby schools relate to price per sq ft?':
        
        fig, ax = plt.subplots()
        sns.regplot(
        data=data,
        x=data['Nearby_Schools'],
        y=data['Price_per_SqFt'],
        scatter_kws={'alpha':0.6, 'color':'teal'},
        line_kws={'color':'red'})
        st.pyplot(fig)

    elif plot == 'How do nearby hospitals relate to price per sq ft?':
        fig, ax = plt.subplots()
        sns.regplot(
        data=data,
        x=data['Nearby_Hospitals'],
        y=data['Price_per_SqFt'],
        scatter_kws={'alpha':0.6, 'color':'teal'},
        line_kws={'color':'red'})
        st.pyplot(fig)

    elif plot == 'How does price vary by furnished status?':
        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 5))
        sns.barplot(data=data, x='Furnished_Status', y='Price_per_SqFt', palette='viridis')
        plt.title('Price per SqFt by Furnished Status')
        plt.xlabel('Furnished Status')
        plt.ylabel('Price per SqFt')
        st.pyplot(fig)

    elif plot == 'How does price per sq ft vary by property facing direction?':
        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=data, x='Furnished_Status', y='Price_per_SqFt', palette='viridis')
        plt.xlabel('Facing')
        plt.ylabel('Price per SqFt')
        st.pyplot(fig)

    elif plot == 'How many properties belong to each owner type':
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='Owner_Type', order=data['Owner_Type'].value_counts().index, palette='viridis')
        plt.title('Number of Properties by Owner Type', fontsize=14)
        plt.xlabel('Owner Type')
        plt.ylabel('Count of Properties')
        st.pyplot(fig)
        count = data['Owner_Type'].value_counts()
        st.write(count)

    elif plot == 'How many properties are available under each availability status?':

        fig, ax = plt.subplots()
        sns.countplot(data=data, x='Availability_Status', 
        order=data['Availability_Status'].value_counts().index, 
        palette='plasma'
        )

        plt.title('Property Count by Availability Status', fontsize=15)
        plt.xlabel('Availability Status', fontsize=12)
        plt.ylabel('Number of Properties', fontsize=12)
        st.pyplot(fig)
        property = data['Availability_Status'].value_counts()
        st.write(property)

    elif plot == 'Does parking space affect property price?':
        fig, ax = plt.subplots()
        plt.figure(figsize=(10, 6))
        sns.pointplot(data=data, x='Parking_Space', y='Price_in_Lakhs', color='red')
        plt.title('Average Price Trend vs. Parking Space', fontsize=15)
        plt.grid(axis='y', alpha=0.3)
        plt.show()
        st.pyplot(fig)

    elif plot == 'How do amenities affect price per sq ft?':
        fig, ax = plt.subplots()
        data['Amenity_Count'] = data['Amenities'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
        plt.figure(figsize=(12, 6))
        sns.pointplot(data=data, x='Amenity_Count', y='Price_per_SqFt')

        plt.title('Impact of Amenity Count on Price per SqFt', fontsize=15)
        plt.xlabel('Number of Amenities Available', fontsize=12)
        plt.ylabel('Price per SqFt (Lakhs)', fontsize=12)
        plt.show()
        st.pyplot(fig)

    elif plot == 'How does public transport accessibility relate to price per sq ft or investment potential?':
        fig, ax = plt.subplots()
        plt.figure(figsize=(10, 6))
        sns.pointplot(data=data, x='Public_Transport_Accessibility', y='Price_per_SqFt', 
                    order=['Low', 'Medium', 'High'], color='red', markers='D', capsize=.1)

        plt.title('The Transit Value Escalator', fontsize=15)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()
        st.pyplot(fig)

        


elif pages == "Investment Analysis":
    st.header("Investment Analysis")
    st.write("**Review the preprocessed data on this page to determine if this is a profitable investment.**")

    target_cols = ['State', 'City', 'Locality']
    ohe_cols = ['Property_Type', 'Facing' , 'Owner_Type']
    label_cols = ['Security', 'Parking_Space', 'Public_Transport_Accessibility']
    multilabel_cols = ['Amenities']
    ordinal_cols = ['Furnished_Status', 'Availability_Status']
    num_cols = ['BHK', 'Size_in_SqFt','Price_per_SqFt', 'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property','Nearby_Schools', 'Nearby_Hospitals' ]


    with open("model_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    model = artifacts['model']
    target_encoder = artifacts['target_encoder']
    ohe = artifacts['ohe']
    label_encoder =  artifacts['label_encoder']
    mlb_encoder = artifacts['mlb_encoder']
    scaler = artifacts['scaler']
    final_columns = artifacts['final_columns']

    
    state_options = sorted(data['State'].unique())
    selected_state = st.selectbox("Select the state:", options = state_options, index = None)

    city_options = sorted(data['City'].unique())
    selected_city = st.selectbox("Select the city:", options = city_options, index = None)

    locality_options = sorted(data['Locality'].unique())
    selected_locality = st.selectbox("Select the locality", options = locality_options, index = None)

    property_Type = st.radio("Select the property type", ['Apartment', 'Villa', 'Independent House'])

    facing = st.selectbox("Select the view facing of your building", ['North', 'South', 'East', 'West'])

    owner_type = st.radio("Select the owner type", ['Owners', 'Builder'])

    security = st.radio("Select your preference of the security", ['Yes', 'No'])

    Parking_Space = st.radio("Select your preference of the parking space" , ['Yes', 'No'])

    Public_Transport_Accessibility = st.radio("Select the transport accesibility", ['High', 'Low'])

    furniture = st.radio("Select the furniture status", ['Furnished', 'Semi-Furnished', 'Unfurnished'])

    
    property_type = st.radio("Select the apartmentn type:", ['Apartment', 'Independent House'])

    Avaliability_status= st.radio("Select the./  avaliablity status", ['Under construction', 'Ready to move'])

    options_amenities = sorted(data['Amenities'].unique())
    selected_amenities = st.selectbox("Select your Amenities type", options = options_amenities, index = None)

    BHK = st.slider("Select BHK", min_value=1, max_value=5, step=1)

    size_per_sqFt = sorted(data['Size_in_SqFt'].unique())
    selected_per_SqFt = st.selectbox("Select the size", options = size_per_sqFt, index = None)

    Price_per_SqFt = sorted(data['Price_per_SqFt'].unique())
    selected_price = st.selectbox("Select the price", options = Price_per_SqFt, index = None)

    year_options = sorted(data['Year_Built'].unique())
    selected_year= st.selectbox("Select the year built", options = year_options, index = None)

    floor_no_options = sorted(data['Floor_No'].unique())
    selected_floor = st.selectbox("Select the floor no", options = floor_no_options, index = None)

    total_floors_options = sorted(data['Total_Floors'].unique())
    selected_total_floor = st.selectbox("Select the total floor", options = total_floors_options, index = None)

    age_options = sorted(data['Age_of_Property'].unique())
    selected_age_of_property = st.selectbox("Select the age of property", options = age_options, index = None)

    options_school = sorted(data['Nearby_Schools'].unique())
    nearby_school = st.selectbox("Select the no of school nearby", options = options_school, index = None)

    options_hospital = sorted(data['Nearby_Hospitals'].unique())
    nearby_hospital = st.selectbox("Select the no of hospital nearby", options = options_hospital, index = None)

    input_df = pd.DataFrame({
    "State": [selected_state],
    "City": [selected_city],
    "Locality" : [selected_locality],
    "Property_Type": [property_type],
    "Facing": [facing],
    "Owner_Type": [owner_type],
    "Security": ([security]),
    "Parking_Space":[Parking_Space],
    "Public_Transport_Accessibility":[Public_Transport_Accessibility],
    "Furnished_Status" : [furniture],
    "Availability_Status": [Avaliability_status],
    "Amenities" : [selected_amenities],
    "BHK": [BHK],
    'Size_in_SqFt' : [selected_per_SqFt],
    'Price_per_SqFt' : [selected_price],
    "Year_Built" : [selected_year],
    "Floor_No" : [selected_floor],
    "Total_Floors" : [selected_total_floor],
    "Age_of_Property" : [selected_age_of_property],
    "Nearby_Schools" : [nearby_school],
    'Nearby_Hospitals' : [nearby_hospital],
  
    })

    if st.button("Predict"):


    # Target Encoding
        X_target = target_encoder.transform(input_df[target_cols])

        # OneHot
        X_ohe = ohe.transform(input_df[ohe_cols])
        X_ohe = pd.DataFrame(X_ohe, columns=ohe.get_feature_names_out(ohe_cols))

        # Label Encoding
        X_label = pd.DataFrame()
        for col in label_cols:
            X_label[col] = label_encoder[col].transform(input_df[col])

        # MultiLabel Encoding
        X_multi = pd.DataFrame()
        for col in multilabel_cols:
            mlb = mlb_encoder[col]
            temp = mlb.transform(input_df[col])
            temp_df = pd.DataFrame(
                temp,
                columns=[f"{col}_{c}" for c in mlb.classes_]
            )
            X_multi = pd.concat([X_multi, temp_df], axis=1)

        # # Scale Numeric
        X_num = scaler.transform(input_df[num_cols])
        X_num = pd.DataFrame(X_num, columns=num_cols)

        X_final = pd.concat([
        X_target.reset_index(drop=True),
        X_ohe.reset_index(drop=True),
        X_label.reset_index(drop=True),
        X_multi.reset_index(drop=True),
        X_num.reset_index(drop=True)
        ], axis=1)

        X_final = X_final.reindex(columns=final_columns, fill_value=0)

        prediction = model.predict(X_final.values)
        st.success(f"Prediction: {prediction[0]}")

elif pages == 'Explanation of Machine Learning':
    st.header("Approach of the process")
    process = st.radio("Select the Process", ["Handling Missing Value" ,'Adding Target Features',
                                              "Handling Categorial Features","Machine Learning"])
    if process == "Handling Missing Value":
        st.write("### Handling  Missing Value")
        null = data.isnull().sum()
        st.write(null)
        st.success("There is no missing value in this dataframe / dataset")
    
    elif process == 'Adding Target Features':
        st.write("Incorporating two target features: **Future Investment Value and Investment Class**")

        st.markdown("""
        ### Investment Classification Logic
        Evaluate properties based on their projected growth rate (r) and physical age.
        **1. Growth Adjustments:**
        * **Transit Premium:** +1.5% to growth rate if Public Transport is 'High'.
        * **New Property Bonus:** +1.0% to growth rate if the property is less than 5 years old.

        **2. Calculation:**
        The future value is calculated using the compound interest formula:
        $$Future Value = Price \times (1 + r)^5$$

        **3. Categorization:**
        * **Best Investment:** Growth rate $\ge$ 8% and Age < 10 years.
        * **Better Investment:** Growth rate $\ge$ 7%.
        * **Worst Investment:** All other properties.
        """)

    elif process == "Handling Categorial Features":
        st.write("### Specialized encoders for individual columns")
        st.markdown("""
        * **Target Encoder** -> State, City , Locality
        * **One Hot Encoder** -> Property_Type , Facing , Owner_Type
        * **Label Encoder** -> Security , Parking_Space ,  Public_Transport_Accessibility ,  Furnished_Status ,  Availability_Status
        * **Multi Label Encoder** ->Amenities 
                    """)
    elif process == "Machine Learning":
        st.write("**The process of optimizing the model so that it can predict the correct response based on the training data samples**")
        
        st.markdown("""
        * **KN Neighbors**: Score - 0.97
        * **Decision Forest**: Score - 1.0
        * **Random Forest**: Score - 1.0
                    """)