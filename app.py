import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import numpy as np

# Optional: real-time prediction model
import joblib

# ------------------------------------------------
# Load data (cached for performance)
# ------------------------------------------------
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?export=download&id=1oLPYkBuwTavj7wqpokPIC-r7gJnECP0R"
    return pd.read_csv(url)


df = load_data()

# ------------------------------------------------
# Try to load Random Forest pipeline (optional)
# ------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("rf_pipeline.pkl")   # pipeline from your notebook
        return model
    except Exception:
        return None

rf_pipeline = load_model()

grav_mapping = {
    1: "Uninjured",
    2: "Slightly injured",
    3: "Hospitalised",
    4: "Killed"
}

# ------------------------------------------------
# Sidebar – navigation and global filters
# ------------------------------------------------
st.sidebar.title("Presentation menu")

page = st.sidebar.radio("Go to", [
    "1. Introduction",
    "2. Methodology",
    "3. Interactive map",
    "4. Descriptive statistics",
    "5. Correlation analysis",
    "6. Model results",
    "7. Live prediction",
    "8. Conclusion"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Global filters**")

# Year filter
years = sorted(df["annee_accident"].dropna().unique())
year_choice = st.sidebar.selectbox(
    "Year",
    options=["All years"] + list(map(int, years))
)

# Department filter
deps = sorted(df["departement_code"].dropna().unique())
dept_choice = st.sidebar.selectbox(
    "Department",
    options=["All departments"] + list(deps)
)

# Apply filters
df_filtered = df.copy()
if year_choice != "All years":
    df_filtered = df_filtered[df_filtered["annee_accident"] == int(year_choice)]

if dept_choice != "All departments":
    df_filtered = df_filtered[df_filtered["departement_code"] == dept_choice]

# ------------------------------------------------
# PAGE 1 : INTRODUCTION
# ------------------------------------------------
if page == "1. Introduction":
    st.title("Predictive Analysis of Road Accident Severity in France")

    st.write(
        """
        This web app is a dynamic summary of my project on road accident
        severity in France. It combines data engineering, exploratory analysis
        and predictive modelling into a single interactive interface.

        The underlying dataset is the BAAC database (ONISR), merged at
        **user level** so that each row corresponds to one road user involved
        in an accident.
        """
    )

    st.markdown(
        f"""
        **Number of users in the dataset:** {len(df):,}  
        **Number of variables after preprocessing:** {df.shape[1]}
        """
    )

    st.write(
        """
        You can move between sections using the menu on the left.
        Global filters (year, department) are applied to most pages so that
        you can focus on a specific period or region if needed.
        """
    )

    if st.checkbox("Show a preview of the filtered dataset"):
        st.dataframe(df_filtered.head())

# ------------------------------------------------
# PAGE 2 : METHODOLOGY
# ------------------------------------------------
elif page == "2. Methodology":
    st.title("Methodology")

    st.subheader("1. Data sources and integration")
    st.write(
        """
        The analysis is based on the French BAAC files published by ONISR.
        Four tables were used: **Characteristics**, **Locations**, **Vehicles**
        and **Users**.  

        These tables were merged step by step using the accident identifier
        `Num_Acc`, the vehicle identifier and the user identifier, in order to
        obtain a single analytical dataset with one row per user and a
        consolidated severity variable.
        """
    )

    st.subheader("2. Cleaning and feature engineering")
    st.write(
        """
        The raw files contained miscoded values, missing data and inconsistent
        coordinates. The main cleaning operations were:
        - type harmonisation for dates, codes and coordinates;  
        - handling of missing values using ONISR conventions;  
        - removal of impossible ages and invalid GPS coordinates;  
        - creation of derived variables (date, weekday, time-of-day band,
          simplified road and weather categories, age classes, generational
          cohorts, etc.).
        """
    )

    st.subheader("3. Modelling strategy")
    st.write(
        """
        The target variable is the **four-level severity score**  
        (Uninjured, Slightly injured, Hospitalised, Killed).  

        Two models were estimated:

        - a **multinomial logistic regression**, used as a linear baseline;  
        - a **Random Forest classifier**, built on the same set of features
          and able to capture non-linear effects and interactions.

        Categorical variables were encoded with one-hot encoding, age was kept
        as a continuous variable, and a stratified train/test split was used to
        preserve the severity distribution.
        """
    )

# ------------------------------------------------
# PAGE 3 : INTERACTIVE MAP
# ------------------------------------------------
elif page == "3. Interactive map":
    st.title("Geographical distribution of accidents")

    st.write(
        """
        The map below shows a sample of accidents with valid GPS coordinates
        after applying the selected filters.  
        Each point corresponds to a user involved in an accident, coloured by
        injury severity.
        """
    )

    df_map = df_filtered.dropna(subset=["latitude", "longitude"])
    if df_map.empty:
        st.warning("No data available with the current filters.")
    else:
        severity_colors = {1: "green", 2: "orange", 3: "red", 4: "darkred"}
        m = folium.Map(location=[46.5, 2.5], zoom_start=6)

        sample_size = min(3000, len(df_map))
        for _, row in df_map.sample(sample_size, random_state=0).iterrows():
            folium.CircleMarker(
                [row["latitude"], row["longitude"]],
                radius=3,
                color=severity_colors.get(row["gravite"], "gray"),
                fill=True,
                fill_opacity=0.7,
            ).add_to(m)

        st_folium(m, height=600)

# ------------------------------------------------
# PAGE 4 : DESCRIPTIVE STATISTICS
# ------------------------------------------------
elif page == "4. Descriptive statistics":
    st.title("Descriptive statistics")

    st.write(
        """
        This section summarises some key variables related to accident severity
        for the current selection (year / department).
        """
    )

    if df_filtered.empty:
        st.warning("No data available with the current filters.")
    else:
        # ----- Age distribution -----
        if "age" in df_filtered.columns and df_filtered["age"].notna().sum() > 0:
            fig_age = px.histogram(
                df_filtered,
                x="age",
                nbins=40,
                title="Age distribution",
            )
            st.plotly_chart(fig_age, use_container_width=True)

        # ----- Severity distribution (built from numeric 'gravite') -----
        if "gravite" not in df_filtered.columns:
            st.error("The numeric severity column 'gravite' is missing.")
        else:
            df_tmp = df_filtered.copy()
            df_tmp["gravite_str"] = df_tmp["gravite"].map(grav_mapping)

            if df_tmp["gravite_str"].dropna().empty:
                st.warning("No valid severity data for the current filters.")
            else:
                fig_sev = px.histogram(
                    df_tmp,
                    x="gravite_str",
                    title="Distribution of injury severity",
                    labels={"gravite_str": "Severity"},
                )
                st.plotly_chart(fig_sev, use_container_width=True)

        # ----- Time-of-day histogram (use 'heure_accident') -----
        hour_col = None
        if "heure" in df_filtered.columns:
            hour_col = "heure"
        elif "heure_accident" in df_filtered.columns:
            hour_col = "heure_accident"

        if hour_col is None:
            st.info("No hour column available in the dataset.")
        else:
            col_no_na = df_filtered[hour_col].dropna()
            if col_no_na.empty:
                st.info("No valid hour data available for the current filters.")
            else:
                fig_hour = px.histogram(
                    df_filtered,
                    x=hour_col,
                    nbins=24,
                    title="Accidents by hour of day",
                    labels={hour_col: "Hour"},
                )
                st.plotly_chart(fig_hour, use_container_width=True)

        st.write(
            """
            Main observations, which are stable across most filters:

            - Severe accidents are more frequent among older users and
              vulnerable road users (pedestrians and two-wheelers).  
            - Most accidents occur during everyday mobility (commuting,
              local trips), with clear peaks at the beginning and end of
              the working day.  
            - Adverse weather and poor visibility systematically increase
              the proportion of severe outcomes.
            """
        )


# ------------------------------------------------
# PAGE 5 : CORRELATION ANALYSIS
# ------------------------------------------------
elif page == "5. Correlation analysis":
    st.title("Correlation between severity and key variables")

    st.write(
        """
        Here we look at the linear correlations between the severity score and
        a small set of numerical or coded variables. This is not a causal
        analysis, but it gives a rough idea of which dimensions move together.
        """
    )

    if df_filtered.empty:
        st.warning("No data available with the current filters.")
    else:
        # choose a subset of numeric / coded variables
        num_cols = [
            "gravite",
            "age",
            "vitesse_max",
            "luminosite",
            "meteo_code",
            "etat_surface",
            "en_agglomeration",
        ]
        num_cols = [c for c in num_cols if c in df_filtered.columns]

        corr_df = df_filtered[num_cols].dropna()
        if corr_df.empty:
            st.warning("Not enough numeric data to compute correlations.")
        else:
            corr = corr_df.corr()

            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation matrix (filtered sample)",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            st.write(
                """
                As expected, the severity score is positively correlated with
                higher speed limits, non-urban locations and degraded surface
                conditions, and negatively correlated with good visibility.
                """
            )

# ------------------------------------------------
# PAGE 6 : MODEL RESULTS
# ------------------------------------------------
elif page == "6. Model results":
    st.title("Predictive model results")

    st.write(
        """
        Two models were estimated:

        - a multinomial logistic regression, used as a transparent baseline;  
        - a Random Forest classifier, used as the main predictive model.
        """
    )

    st.write(
        """
        In out-of-sample evaluation on the test set:

        - Logistic regression reaches an accuracy of about **49%**,  
        - Random Forest improves this to about **51%**, with a higher
          macro-averaged F1-score.

        Both models classify **uninjured** and **slightly injured** cases
        reasonably well, but they struggle with the rarest and most serious
        outcomes (hospitalised and killed). This is mainly due to the strong
        class imbalance in the data.
        """
    )

    st.write(
        """
        According to the Random Forest, the most influential variables are:

        - user category (driver, passenger, pedestrian),  
        - weather and light conditions,  
        - road category and surface condition,  
        - speed limit,  
        - age.
        """
    )

# ------------------------------------------------
# PAGE 7 : LIVE PREDICTION
# ------------------------------------------------
elif page == "7. Live prediction":
    st.title("Live prediction of accident severity")

    if rf_pipeline is None:
        st.warning(
            "The Random Forest pipeline (`rf_pipeline.pkl`) is not available "
            "in the app directory. Save your trained pipeline from the "
            "notebook and reload the app to use this page."
        )
    else:
        st.write(
            """
            This form simulates the information available at the time of
            an accident and uses the Random Forest pipeline to predict the
            most likely severity level.
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            age_input = st.number_input("Age", min_value=0, max_value=100, value=35)
            sexe_input = st.selectbox("Sex", {"Male": 1, "Female": 2}.keys())
            catu_input = st.selectbox(
                "User category",
                {
                    "Driver": 1,
                    "Passenger": 2,
                    "Pedestrian": 3,
                }.keys(),
            )
            trajet_input = st.selectbox(
                "Trip purpose",
                {
                    "Home–work": 1,
                    "Home–school": 2,
                    "Business": 3,
                    "Leisure": 4,
                    "Other/unknown": 9,
                }.keys(),
            )

        with col2:
            catr_input = st.selectbox(
                "Road category",
                {
                    "Motorway": 1,
                    "National": 2,
                    "Departmental": 3,
                    "Communal": 4,
                    "Off-road / other": 9,
                }.keys(),
            )
            meteo_input = st.selectbox(
                "Weather",
                {
                    "Normal": 1,
                    "Light rain": 2,
                    "Heavy rain": 3,
                    "Snow / hail": 4,
                    "Fog / smoke": 5,
                    "Strong wind": 6,
                    "Other / unknown": 9,
                }.keys(),
            )
            lumi_input = st.selectbox(
                "Light conditions",
                {
                    "Daylight": 1,
                    "Twilight": 2,
                    "Night, lit": 3,
                    "Night, unlit": 4,
                }.keys(),
            )
            surf_input = st.selectbox(
                "Road surface condition",
                {
                    "Dry": 1,
                    "Wet": 2,
                    "Snow / ice": 3,
                    "Flooded / other": 4,
                }.keys(),
            )

        if st.button("Predict severity"):
            # Map labels to codes
            sex_map = {"Male": 1, "Female": 2}
            catu_map = {"Driver": 1, "Passenger": 2, "Pedestrian": 3}
            trajet_map = {
                "Home–work": 1,
                "Home–school": 2,
                "Business": 3,
                "Leisure": 4,
                "Other/unknown": 9,
            }
            catr_map = {
                "Motorway": 1,
                "National": 2,
                "Departmental": 3,
                "Communal": 4,
                "Off-road / other": 9,
            }
            meteo_map = {
                "Normal": 1,
                "Light rain": 2,
                "Heavy rain": 3,
                "Snow / hail": 4,
                "Fog / smoke": 5,
                "Strong wind": 6,
                "Other / unknown": 9,
            }
            lumi_map = {
                "Daylight": 1,
                "Twilight": 2,
                "Night, lit": 3,
                "Night, unlit": 4,
            }
            surf_map = {
                "Dry": 1,
                "Wet": 2,
                "Snow / ice": 3,
                "Flooded / other": 4,
            }

            X_new = pd.DataFrame(
                [{
                    "age": age_input,
                    "sexe": sex_map[sexe_input],
                    "categorie_usager": catu_map[catu_input],
                    "motif_trajet": trajet_map[trajet_input],
                    "categorie_route": catr_map[catr_input],
                    "meteo_code": meteo_map[meteo_input],
                    "luminosite": lumi_map[lumi_input],
                    "etat_surface": surf_map[surf_input],
                    "regime_circulation": 1,  # neutral default
                }]
            )

            y_proba = rf_pipeline.predict_proba(X_new)[0]
            y_pred = rf_pipeline.predict(X_new)[0]

            st.subheader("Predicted severity")
            st.write(f"Most likely class: **{grav_mapping.get(y_pred, 'Unknown')}**")

            proba_df = pd.DataFrame(
                {
                    "Severity": [grav_mapping[k] for k in grav_mapping.keys()],
                    "Probability": y_proba,
                }
            )
            fig_proba = px.bar(
                proba_df,
                x="Severity",
                y="Probability",
                title="Predicted probability for each severity level",
                range_y=[0, 1],
            )
            st.plotly_chart(fig_proba, use_container_width=True)

# ------------------------------------------------
# PAGE 8 : CONCLUSION
# ------------------------------------------------
elif page == "8. Conclusion":
    st.title("Conclusion and perspectives")

    st.write(
        """
        The project shows that it is possible to build a coherent analytical
        and predictive framework for road accident severity based on public
        BAAC data.

        Human factors (age, user category), environmental conditions (weather,
        light) and structural aspects (road type, surface condition and speed
        limit) are all strongly linked to the severity of outcomes.
        """
    )

    st.write(
        """
        The current models perform reasonably well for mild accidents but
        still have difficulties with the rarest and most serious events.  
        This is a limitation shared by many studies in this field.

        Future work could include richer behavioural variables, spatial
        features and more specialised algorithms focused on rare severe
        injuries. Another natural extension would be to link severity to
        the financial cost of claims.
        """
    )

    st.success("End of the interactive presentation. Thank you for your attention.")
