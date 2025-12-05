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
        This page provides a structured descriptive analysis of accident severity,
        based on three dimensions:
        **(1) user profile, (2) road & environment, (3) time patterns.**
        All graphs use the current year / department filters.
        """
    )

    if df_filtered.empty:
        st.warning("No data available with the current filters.")
    else:
        # -----------------------------
        # Common helpers
        # -----------------------------
        df_desc = df_filtered.copy()

        # Severity as label
        if "gravite" in df_desc.columns:
            df_desc["severity_label"] = df_desc["gravite"].map(grav_mapping)
        else:
            st.error("Column 'gravite' is missing from the dataset.")
            st.stop()

        # User category mapping (BAAC codes)
        catu_map = {
            1: "Driver",
            2: "Passenger",
            3: "Pedestrian",
            4: "Pedestrian (sidewalk)",
            5: "Other",
        }
        if "categorie_usager" in df_desc.columns:
            df_desc["user_type"] = df_desc["categorie_usager"].map(catu_map).fillna("Other")

        # Sex mapping
        sex_map = {1: "Male", 2: "Female"}
        if "sexe" in df_desc.columns:
            df_desc["sex_label"] = df_desc["sexe"].map(sex_map).fillna("Unknown")

        # Day-of-week labels
        dow_map = {
            0: "Mon", 1: "Tue", 2: "Wed",
            3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"
        }
        if "jour_semaine" in df_desc.columns:
            df_desc["day_of_week"] = df_desc["jour_semaine"].map(dow_map)

        # Weekend label
        if "weekend" in df_desc.columns:
            df_desc["weekend_label"] = df_desc["weekend"].map({0: "Weekday", 1: "Weekend"})

        # -----------------------------
        # 1. USER PROFILE & SEVERITY
        # -----------------------------
        st.subheader("1. User profile and severity")

        cols = st.columns(2)

        # 1.1 Age distribution
        with cols[0]:
            if "age" in df_desc.columns and df_desc["age"].notna().sum() > 0:
                fig_age = px.histogram(
                    df_desc,
                    x="age",
                    nbins=40,
                    title="Age distribution",
                )
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.info("No age information available.")

        # 1.2 Severity by age band (using age_classe if present)
        with cols[1]:
            age_band_col = None
            if "age_classe" in df_desc.columns:
                age_band_col = "age_classe"
            elif "age" in df_desc.columns:
                # crude age bands if age_classe is missing
                df_desc["age_band"] = pd.cut(
                    df_desc["age"],
                    bins=[0, 17, 24, 44, 64, 120],
                    labels=["0–17", "18–24", "25–44", "45–64", "65+"],
                    right=True,
                )
                age_band_col = "age_band"

            if age_band_col is not None:
                tmp = (
                    df_desc.dropna(subset=[age_band_col])
                    .groupby([age_band_col, "severity_label"])
                    .size()
                    .reset_index(name="count")
                )
                tmp["prop"] = tmp["count"] / tmp.groupby(age_band_col)["count"].transform("sum")
                fig_age_sev = px.bar(
                    tmp,
                    x=age_band_col,
                    y="prop",
                    color="severity_label",
                    barmode="stack",
                    title="Severity distribution by age band",
                    labels={"prop": "Share within age band"},
                )
                st.plotly_chart(fig_age_sev, use_container_width=True)

        cols2 = st.columns(2)

        # 1.3 Severity by user type
        with cols2[0]:
            if "user_type" in df_desc.columns:
                tmp = (
                    df_desc.groupby(["user_type", "severity_label"])
                    .size()
                    .reset_index(name="count")
                )
                tmp["prop"] = tmp["count"] / tmp.groupby("user_type")["count"].transform("sum")
                fig_user = px.bar(
                    tmp,
                    x="user_type",
                    y="prop",
                    color="severity_label",
                    barmode="stack",
                    title="Severity distribution by user type",
                    labels={"prop": "Share within user type"},
                )
                st.plotly_chart(fig_user, use_container_width=True)
            else:
                st.info("User category is not available in the dataset.")

        # 1.4 Severity by sex
        with cols2[1]:
            if "sex_label" in df_desc.columns:
                tmp = (
                    df_desc.groupby(["sex_label", "severity_label"])
                    .size()
                    .reset_index(name="count")
                )
                tmp["prop"] = tmp["count"] / tmp.groupby("sex_label")["count"].transform("sum")
                fig_sex = px.bar(
                    tmp,
                    x="sex_label",
                    y="prop",
                    color="severity_label",
                    barmode="stack",
                    title="Severity distribution by sex",
                    labels={"prop": "Share within sex"},
                )
                st.plotly_chart(fig_sex, use_container_width=True)
            else:
                st.info("Sex information is not available.")

        st.markdown(
            """
            **Takeaways:** the descriptive patterns confirm that severe injuries are
            more frequent among older users and among vulnerable road users
            such as pedestrians and two-wheelers. Men are slightly over-represented
            in the most serious outcomes.
            """
        )

        # -----------------------------
        # 2. ENVIRONMENT & INFRASTRUCTURE
        # -----------------------------
        st.subheader("2. Road environment and infrastructure")

        cols3 = st.columns(2)

        # 2.1 Severity vs speed limit
        with cols3[0]:
            if "vitesse_max" in df_desc.columns:
                tmp = (
                    df_desc.groupby("vitesse_max")
                    .agg(
                        mean_severity=("gravite", "mean"),
                        n=("gravite", "size"),
                    )
                    .reset_index()
                )
                tmp = tmp[tmp["n"] > 50]  # remove very small groups
                fig_speed = px.bar(
                    tmp,
                    x="vitesse_max",
                    y="mean_severity",
                    title="Average severity by speed limit",
                    labels={"vitesse_max": "Speed limit (km/h)", "mean_severity": "Average severity"},
                )
                st.plotly_chart(fig_speed, use_container_width=True)
            else:
                st.info("Speed limit is not available in the dataset.")

        # 2.2 Severity distribution by road type
        with cols3[1]:
            road_col = None
            if "type_route_simple" in df_desc.columns:
                road_col = "type_route_simple"
            elif "categorie_route" in df_desc.columns:
                road_col = "categorie_route"

            if road_col is not None:
                tmp = (
                    df_desc.groupby([road_col, "severity_label"])
                    .size()
                    .reset_index(name="count")
                )
                tmp["prop"] = tmp["count"] / tmp.groupby(road_col)["count"].transform("sum")
                fig_road = px.bar(
                    tmp,
                    x=road_col,
                    y="prop",
                    color="severity_label",
                    barmode="stack",
                    title="Severity distribution by road type",
                    labels={"prop": "Share within road type"},
                )
                st.plotly_chart(fig_road, use_container_width=True)
            else:
                st.info("No road type information available.")

        cols4 = st.columns(2)

        # 2.3 Severity distribution by weather
        with cols4[0]:
            weather_col = None
            if "meteo_simple" in df_desc.columns:
                weather_col = "meteo_simple"
            elif "meteo_code" in df_desc.columns:
                weather_col = "meteo_code"

            if weather_col is not None:
                tmp = (
                    df_desc.groupby([weather_col, "severity_label"])
                    .size()
                    .reset_index(name="count")
                )
                tmp["prop"] = tmp["count"] / tmp.groupby(weather_col)["count"].transform("sum")
                fig_weather = px.bar(
                    tmp,
                    x=weather_col,
                    y="prop",
                    color="severity_label",
                    barmode="stack",
                    title="Severity distribution by weather conditions",
                    labels={"prop": "Share within weather group"},
                )
                st.plotly_chart(fig_weather, use_container_width=True)
            else:
                st.info("No weather information available.")

        # 2.4 Severity distribution by light conditions
        with cols4[1]:
            lumi_col = None
            if "luminosite" in df_desc.columns:
                lumi_col = "luminosite"

            if lumi_col is not None:
                lumi_map = {
                    1: "Daylight",
                    2: "Twilight",
                    3: "Night (lit)",
                    4: "Night (unlit)",
                }
                df_desc["light_label"] = df_desc[lumi_col].map(lumi_map).fillna("Other")
                tmp = (
                    df_desc.groupby(["light_label", "severity_label"])
                    .size()
                    .reset_index(name="count")
                )
                tmp["prop"] = tmp["count"] / tmp.groupby("light_label")["count"].transform("sum")
                fig_light = px.bar(
                    tmp,
                    x="light_label",
                    y="prop",
                    color="severity_label",
                    barmode="stack",
                    title="Severity distribution by light conditions",
                    labels={"prop": "Share within light group"},
                )
                st.plotly_chart(fig_light, use_container_width=True)
            else:
                st.info("No light condition information available.")

        st.markdown(
            """
            **Takeaways:** higher speed limits, non-urban roads and degraded
            weather or light conditions are clearly associated with more severe
            outcomes. This is fully consistent with the exploratory analysis
            performed in the notebook.
            """
        )

        # -----------------------------
        # 3. TIME PATTERNS & HEATMAPS
        # -----------------------------
        st.subheader("3. Time patterns and heatmaps")

        cols5 = st.columns(2)

        # 3.1 Heatmap: accident counts by hour x day of week
        with cols5[0]:
            if "heure_accident" in df_desc.columns and "day_of_week" in df_desc.columns:
                tmp = (
                    df_desc.dropna(subset=["heure_accident", "day_of_week"])
                    .groupby(["day_of_week", "heure_accident"])
                    .size()
                    .reset_index(name="count")
                )
                if not tmp.empty:
                    fig_heat_count = px.density_heatmap(
                        tmp,
                        x="heure_accident",
                        y="day_of_week",
                        z="count",
                        color_continuous_scale="Viridis",
                        title="Number of accidents by hour and day",
                        labels={"heure_accident": "Hour"},
                    )
                    st.plotly_chart(fig_heat_count, use_container_width=True)
                else:
                    st.info("Not enough data to build the hourly heatmap.")
            else:
                st.info("Hour or day-of-week is missing from the dataset.")

        # 3.2 Heatmap: average severity by hour x day of week
        with cols5[1]:
            if "heure_accident" in df_desc.columns and "day_of_week" in df_desc.columns:
                tmp = (
                    df_desc.dropna(subset=["heure_accident", "day_of_week"])
                    .groupby(["day_of_week", "heure_accident"])
                    .agg(mean_severity=("gravite", "mean"))
                    .reset_index()
                )
                if not tmp.empty:
                    fig_heat_sev = px.density_heatmap(
                        tmp,
                        x="heure_accident",
                        y="day_of_week",
                        z="mean_severity",
                        color_continuous_scale="RdBu_r",
                        title="Average severity by hour and day",
                        labels={"heure_accident": "Hour", "mean_severity": "Average severity"},
                    )
                    st.plotly_chart(fig_heat_sev, use_container_width=True)
                else:
                    st.info("Not enough data to build the severity heatmap.")
            else:
                st.info("Hour or day-of-week is missing from the dataset.")

        # 3.3 Weekend vs weekday severity
        if "weekend_label" in df_desc.columns:
            tmp = (
                df_desc.groupby("weekend_label")
                .agg(mean_severity=("gravite", "mean"))
                .reset_index()
            )
            fig_we = px.bar(
                tmp,
                x="weekend_label",
                y="mean_severity",
                title="Average severity – weekday vs weekend",
                labels={"mean_severity": "Average severity"},
            )
            st.plotly_chart(fig_we, use_container_width=True)

        st.markdown(
            """
            **Takeaways:** the heatmaps show clear temporal patterns: accident
            frequency peaks during commuting hours, and average severity tends
            to increase at night and during the weekend, when speeds are higher
            and visibility is lower.
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
