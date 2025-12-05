import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import numpy as np
import os
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
    path = "rf_pipeline.pkl"

    # 1. Vérifier que le fichier existe bien côté Streamlit
    if not os.path.exists(path):
        st.warning(f"Model file '{path}' was not found in the app directory.")
        st.write(f"Current working directory: {os.getcwd()}")
        st.write(f"Files here: {os.listdir('.')}")
        return None

    # 2. Essayer de charger et afficher l'erreur exacte si ça plante
    try:
        model = joblib.load(path)
        st.success("Random Forest model successfully loaded.")
        return model
    except Exception as e:
        st.error("Error while loading 'rf_pipeline.pkl'.")
        st.write(f"Details: {e}")
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

    st.write("""
    This section provides a structured exploratory analysis based on  
    **(1) user vulnerabilities, (2) environmental risk factors, and (3) temporal risk patterns**.
    """)

    if df_filtered.empty:
        st.warning("No data available for current filters.")
    else:
        df_desc = df_filtered.copy()

        # ============= PREPROCESSING ============
        # Severity (numeric + label)
        df_desc["severity"] = df_desc["gravite"]
        df_desc["severity_label"] = df_desc["gravite"].map(grav_mapping)

        # User types
        catu_map = {1: "Driver", 2: "Passenger", 3: "Pedestrian"}
        df_desc["user_type"] = df_desc["categorie_usager"].map(catu_map).fillna("Other")

        sex_map = {1: "Male", 2: "Female"}
        df_desc["sex_label"] = df_desc["sexe"].map(sex_map).fillna("Unknown")

        # Time handling (robust)
        # Hour (can be float or string)
        if "heure_accident" in df_desc.columns:
            df_desc["hour"] = pd.to_numeric(df_desc["heure_accident"], errors="coerce")
        else:
            df_desc["hour"] = np.nan

        # Month from date (string -> datetime)
        if "date" in df_desc.columns:
            df_desc["date_parsed"] = pd.to_datetime(
                df_desc["date"], errors="coerce", dayfirst=True
            )
            df_desc["month"] = df_desc["date_parsed"].dt.month
        else:
            df_desc["month"] = np.nan

        # Speed category (if not already there)
        if "vma_cat" in df_desc.columns:
            df_desc["speed_cat"] = df_desc["vma_cat"]
        elif "vitesse_max" in df_desc.columns:
            df_desc["speed_cat"] = pd.cut(
                df_desc["vitesse_max"],
                bins=[0, 30, 50, 70, 90, 110, 150],
                labels=["≤30", "31–50", "51–70", "71–90", "91–110", ">110"],
                include_lowest=True,
            )
        else:
            df_desc["speed_cat"] = np.nan

        # Light labels
        lumi_map = {1: "Daylight", 2: "Twilight", 3: "Night (lit)", 4: "Night (unlit)"}
        df_desc["light_label"] = df_desc["luminosite"].map(lumi_map)

        # =================================================
        # 1️⃣ USER VULNERABILITY ANALYSIS
        # =================================================
        st.header("1. User vulnerability")

        colA, colB = st.columns(2)

        # G1: Age distribution
        with colA:
            if df_desc["age"].notna().sum() > 0:
                fig_age = px.histogram(
                    df_desc.dropna(subset=["age"]),
                    x="age",
                    nbins=40,
                    title="Age distribution",
                )
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.info("No valid age information available.")

        # G2: Average severity by age
        with colB:
            tmp = (
                df_desc.dropna(subset=["age"])
                .groupby("age")["severity"]
                .mean()
                .reset_index()
            )
            # On évite la courbe bruitée sur les âges rares
            if not tmp.empty:
                fig_age_sev = px.line(
                    tmp,
                    x="age",
                    y="severity",
                    title="Average severity by age",
                )
                st.plotly_chart(fig_age_sev, use_container_width=True)

        colC, colD = st.columns(2)

        # G3: Severity rate by user type
        with colC:
            tmp = (
                df_desc.groupby("user_type")["severity"]
                .mean()
                .reset_index()
                .sort_values("severity", ascending=False)
            )
            fig_user_sev = px.bar(
                tmp,
                x="user_type",
                y="severity",
                title="Average severity by user type",
            )
            st.plotly_chart(fig_user_sev, use_container_width=True)

        # G4: Severity by sex (boxplot)
        with colD:
            fig_sex_box = px.box(
                df_desc,
                x="sex_label",
                y="severity",
                title="Distribution of severity by sex",
            )
            st.plotly_chart(fig_sex_box, use_container_width=True)

        # =================================================
        # 2️⃣ ENVIRONMENTAL RISK ANALYSIS
        # =================================================
        st.header("2. Environmental and infrastructure risks")

        colE, colF = st.columns(2)

        road_col = "type_route_simple"
        weather_col = "meteo_simple"

        # G5: Boxplot severity by road type
        with colE:
            if road_col in df_desc.columns:
                fig_road = px.box(
                    df_desc,
                    x=road_col,
                    y="severity",
                    title="Severity by road type",
                )
                st.plotly_chart(fig_road, use_container_width=True)
            else:
                st.info("Road type information is not available in this subset.")

        # G6: Severity rate by weather
        with colF:
            if weather_col in df_desc.columns:
                tmp = (
                    df_desc.groupby(weather_col)["severity"]
                    .mean()
                    .reset_index()
                    .sort_values("severity", ascending=False)
                )
                fig_weather = px.bar(
                    tmp,
                    x=weather_col,
                    y="severity",
                    title="Average severity by weather condition",
                )
                st.plotly_chart(fig_weather, use_container_width=True)

        colG, colH = st.columns(2)

        # G7: Severity rate by road surface condition
        with colG:
            if "etat_surface" in df_desc.columns:
                tmp = (
                    df_desc.groupby("etat_surface")["severity"]
                    .mean()
                    .reset_index()
                    .sort_values("severity", ascending=False)
                )
                fig_surface = px.bar(
                    tmp,
                    x="etat_surface",
                    y="severity",
                    title="Average severity by surface condition (code)",
                )
                st.plotly_chart(fig_surface, use_container_width=True)

        # G8: Heatmap road × weather
        with colH:
            if (weather_col in df_desc.columns) and (road_col in df_desc.columns):
                tmp = (
                    df_desc.groupby([road_col, weather_col])["severity"]
                    .mean()
                    .reset_index()
                )
                fig_heat_rw = px.density_heatmap(
                    tmp,
                    x=road_col,
                    y=weather_col,
                    z="severity",
                    color_continuous_scale="RdBu_r",
                    title="Heatmap: severity by road type × weather",
                )
                st.plotly_chart(fig_heat_rw, use_container_width=True)

        # G9: Severity by speed limit category
        st.subheader("Speed environment")
        if df_desc["speed_cat"].notna().any():
            tmp = (
                df_desc.dropna(subset=["speed_cat"])
                .groupby("speed_cat")["severity"]
                .mean()
                .reset_index()
            )
            fig_speed = px.bar(
                tmp,
                x="speed_cat",
                y="severity",
                title="Average severity by speed limit category",
            )
            st.plotly_chart(fig_speed, use_container_width=True)

        # =================================================
        # 3️⃣ TEMPORAL RISK PATTERNS
        # =================================================
        st.header("3. Temporal risk patterns")

        colI, colJ = st.columns(2)

        # G10: Hourly severity rate
        with colI:
            tmp = (
                df_desc.dropna(subset=["hour"])
                .groupby("hour")["severity"]
                .mean()
                .reset_index()
            )
            if not tmp.empty:
                fig_hour = px.line(
                    tmp,
                    x="hour",
                    y="severity",
                    title="Average severity by hour of day",
                )
                st.plotly_chart(fig_hour, use_container_width=True)
            else:
                st.info("No valid hour information for the current filters.")

        # G11: Heatmap hour × user type
        with colJ:
            tmp = (
                df_desc.dropna(subset=["hour"])
                .groupby(["hour", "user_type"])["severity"]
                .mean()
                .reset_index()
            )
            if not tmp.empty:
                fig_heat_hu = px.density_heatmap(
                    tmp,
                    x="hour",
                    y="user_type",
                    z="severity",
                    title="Heatmap: severity by hour × user type",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig_heat_hu, use_container_width=True)

        # G12: Monthly severity pattern
        tmp = (
            df_desc.dropna(subset=["month"])
            .groupby("month")["severity"]
            .mean()
            .reset_index()
        )
        if not tmp.empty:
            fig_month = px.line(
                tmp,
                x="month",
                y="severity",
                title="Monthly severity pattern",
            )
            st.plotly_chart(fig_month, use_container_width=True)
        else:
            st.info("No valid monthly information available for the current filters.")


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
