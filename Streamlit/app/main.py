import streamlit as st
import numpy as np
import pandas as pd
import cufflinks
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from colors import *

@st.cache
def get_data(url):
    df = pd.read_csv("owid-covid-data.csv")
    df["date"] = pd.to_datetime(df.date).dt.date
    df['date'] = pd.DatetimeIndex(df.date)
    
    return df

colors = get_colors()

url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
data = get_data(url)
columns = ['total_cases', 'new_cases',
                'new_cases_smoothed', 'total_deaths', 'new_deaths',
                'new_deaths_smoothed', 'total_cases_per_million',
                'new_cases_per_million', 'new_cases_smoothed_per_million',
                'total_deaths_per_million', 'new_deaths_per_million',
                'new_deaths_smoothed_per_million', 'new_tests', 'total_tests',
                'total_tests_per_thousand', 'new_tests_per_thousand',
                'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
                'tests_per_case', 'positive_rate', 'stringency_index',
                'population', 'population_density', 'median_age', 'aged_65_older',
                'aged_70_older', 'gdp_per_capita', 'extreme_poverty',
                'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers',
                'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
                'life_expectancy', 'human_development_index']

locations = data.location.unique().tolist()

sidebar = st.sidebar

mode = sidebar.radio("Mode", ["EDA", "Clustering"])
st.markdown("<h1 style='text-align: center; color: #ff0000;'>COVID-19</h1>", unsafe_allow_html=True)
st.markdown("# Mode: {}".format(mode), unsafe_allow_html=True)

if mode=="EDA":
    analysis_type = sidebar.radio("Analysis Type", ["Single", "Multiple"])
    st.markdown(f"# Analysis Mode: {analysis_type}")

    if analysis_type=="Single":
        location_selector = sidebar.selectbox(
            "Select a Location",
            locations
        )
        st.markdown(f"# Currently Selected {location_selector}")
        trend_level = sidebar.selectbox("Trend Level", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
        st.markdown(f"### Currently Selected {trend_level}")

        show_data = sidebar.checkbox("Show Data")

        trend_kwds = {"Daily": "1D", "Weekly": "1W", "Monthly": "1M", "Quarterly": "1Q", "Yearly": "1Y"}
        trend_data = data.query(f"location=='{location_selector}'").\
            groupby(pd.Grouper(key="date", 
            freq=trend_kwds[trend_level])).aggregate(new_cases=("new_cases", "sum"),
            new_deaths = ("new_deaths", "sum"),
            new_vaccinations = ("new_vaccinations", "sum"),
            new_tests = ("new_tests", "sum")).reset_index()

        trend_data["date"] = trend_data.date.dt.date

        new_cases = sidebar.checkbox("New Cases")
        new_deaths = sidebar.checkbox("New Deaths")
        new_vaccinations = sidebar.checkbox("New Vaccinations")
        new_tests = sidebar.checkbox("New Tests")

        lines = [new_cases, new_deaths, new_vaccinations, new_tests]
        line_cols = ["new_cases", "new_deaths", "new_vaccinations", "new_tests"]
        trends = [c[1] for c in zip(lines,line_cols) if c[0]==True]


        if show_data:
            tcols = ["date"] + trends
            st.dataframe(trend_data[tcols])

        subplots=sidebar.checkbox("Show Subplots", True)
        if len(trends)>0:
            fig=trend_data.iplot(kind="line", asFigure=True, xTitle="Date", yTitle="Values",
                                x="date", y=trends, title=f"{trend_level} Trend of {', '.join(trends)}.", subplots=subplots)
            st.plotly_chart(fig, use_container_width=False)

    if analysis_type=="Multiple":
        selected = sidebar.multiselect("Select Locations ", locations)
        st.markdown(f"## Selected Locations: {', '.join(selected)}")
        show_data = sidebar.checkbox("Show Data")
        trend_level = sidebar.selectbox("Trend Level", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
        st.markdown(f"### Currently Selected {trend_level}")

        trend_kwds = {"Daily": "1D", "Weekly": "1W", "Monthly": "1M", "Quarterly": "1Q", "Yearly": "1Y"}
        
        trend_data = data.query(f"location in {selected}").\
            groupby(["location", pd.Grouper(key="date", 
            freq=trend_kwds[trend_level])]).aggregate(new_cases=("new_cases", "sum"),
            new_deaths = ("new_deaths", "sum"),
            new_vaccinations = ("new_vaccinations", "sum"),
            new_tests = ("new_tests", "sum")).reset_index()
        
        trend_data["date"] = trend_data.date.dt.date

        new_cases = sidebar.checkbox("New Cases")
        new_deaths = sidebar.checkbox("New Deaths")
        new_vaccinations = sidebar.checkbox("New Vaccinations")
        new_tests = sidebar.checkbox("New Tests")

        lines = [new_cases, new_deaths, new_vaccinations, new_tests]
        line_cols = ["new_cases", "new_deaths", "new_vaccinations", "new_tests"]
        trends = [c[1] for c in zip(lines,line_cols) if c[0]==True]

        ndf = pd.DataFrame(data=trend_data.date.unique(),columns=["date"])
        
        for s in selected:
            new_cols = ["date"]+[f"{s}_{c}" for c in line_cols]
            tdf = trend_data.query(f"location=='{s}'")
            tdf.drop("location", axis=1, inplace=True)
            tdf.columns=new_cols
            ndf=ndf.merge(tdf,on="date",how="inner")

        if show_data:
            if len(ndf)>0:
                st.dataframe(ndf)
            else:
                st.markdown("Empty Dataframe")
                
        new_trends = []
        for c in trends:
            new_trends.extend([f"{s}_{c}" for s in selected])
        
        subplots=sidebar.checkbox("Show Subplots", True)
        if len(trends)>0:
            st.markdown("### Trend of Selected Locations")
            
            fig=ndf.iplot(kind="line", asFigure=True, xTitle="Date", yTitle="Values",
                                x="date", y=new_trends, title=f"{trend_level} Trend of {', '.join(trends)}.", subplots=subplots)
            st.plotly_chart(fig, use_container_width=False)

elif mode=="Clustering":
    colors = get_colors()    
    features = sidebar.multiselect("Select Features", columns, default=columns[:3])
    
    # select a  clustering algorithm
    calg = sidebar.selectbox("Select a clustering algorithm", ["K-Means","K-Medoids"])
    algs = {"K-Means": KMeans, "K-Medoids": KMedoids}
        
    # select number of clusters
    ks = sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=2)
    
    # select a dataframe to apply cluster on
    
    udf = data.sort_values("date").drop_duplicates(subset=["location"],keep="last").dropna(subset=features)
    udf = udf[~udf.location.isin(["Lower middle income", "North America", "World", "Asia", "Europe", 
                           "European Union", "Upper middle income", 
                           "High income", "South America"])]

    # udf[features].dropna()
    
    if len(features)>=2:
        st.markdown(f"### {calg} Clustering")      
        
        # if using PCA or not
        use_pca = sidebar.radio("Use PCA?",["Yes","No"])
        # if not using pca, do default clustering
        if use_pca=="No":
            st.markdown("### Not Using PCA")
            inertias = []
            for c in range(1,ks+1):
                tdf = udf.copy()
                X = tdf[features]                
                # colors=['red','green','blue','magenta','black','yellow']
                model = algs[calg](n_clusters=c)
                model.fit(X)
                y_kmeans = model.predict(X)
                tdf["cluster"] = y_kmeans
                inertias.append((c,model.inertia_))
                
                
                trace0 = go.Scatter(x=tdf[features[0]],y=tdf[features[1]],mode='markers',  
                                    marker=dict(
                                            color=tdf.cluster.apply(lambda x: colors[x]),
                                            colorscale='Viridis',
                                            showscale=True,
                                            size = udf["total_cases"]%20,
                                            opacity = 0.9,
                                            reversescale = True,
                                            symbol = 'pentagon'
                                            ),
                                    name="Locations", text=udf["location"])
                
                trace1 = go.Scatter(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1],
                                    mode='markers', 
                                    marker=dict(
                                        color=colors,
                                        size=20,
                                        symbol="circle",
                                        showscale=True,
                                        line = dict(
                                            width=1,
                                            color='rgba(102, 102, 102)'
                                            )
                                        
                                        ),
                                    name="Cluster Center")
                    
                data7 = go.Data([trace0, trace1])
                fig = go.Figure(data=data7)
                layout = go.Layout(
                            height=600, width=800, title=f"{calg} Cluster Size {c}",
                            xaxis=dict(
                                title=features[0],
                            ),
                            yaxis=dict(
                                title=features[1]
                            ) ) 
                
                fig.update_layout(layout)
                st.plotly_chart(fig)

            inertias=np.array(inertias).reshape(-1,2)
            performance = go.Scatter(x=inertias[:,0], y=inertias[:,1])
            layout = go.Layout(
                title="Cluster Number vs Inertia",
                xaxis=dict(
                    title="Ks"
                ),
                yaxis=dict(
                    title="Inertia"
                ) ) 
            fig=go.Figure(data=go.Data([performance]))
            fig.update_layout(layout)
            st.plotly_chart(fig)
        
        # if using pca, use pca to reduce dimensionality and then do clustering    
        if use_pca=="Yes":
            st.markdown("### Using PCA")
            comp = sidebar.number_input("Choose Components",1,10,3)
            
            tdf=udf.copy()
            
            X = udf[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=int(comp))
            principalComponents = pca.fit_transform(X_scaled)
            feat = list(range(pca.n_components_))
            PCA_components = pd.DataFrame(principalComponents, columns=list(range(len(feat))))
            choosed_component = sidebar.multiselect("Choose Components",feat,default=[1,2])
            choosed_component=[int(i) for i in choosed_component]
            inertias = []
            if len(choosed_component)>1:
                for c in range(1,ks+1):
                    X = PCA_components[choosed_component]
                    
                    model = algs[calg](n_clusters=c)
                    model.fit(X)
                    y_kmeans = model.predict(X)
                    tdf["cluster"] = y_kmeans
                    inertias.append((c,model.inertia_))
                    
                    trace0 = go.Scatter(x=X[choosed_component[0]],y=X[choosed_component[1]],mode='markers',  
                                        marker=dict(
                                                color=tdf.cluster.apply(lambda x: colors[x]),
                                                colorscale='Viridis',
                                                showscale=True,
                                                size = udf["total_cases"]%20,
                                                opacity = 0.9,
                                                reversescale = True,
                                                symbol = 'pentagon'
                                                ),
                                        name="Locations", text=udf["location"])
                    
                    trace1 = go.Scatter(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1],
                                        mode='markers', 
                                        marker=dict(
                                            color=colors,
                                            size=20,
                                            symbol="circle",
                                            showscale=True,
                                            line = dict(
                                                width=1,
                                                color='rgba(102, 102, 102)'
                                                )
                                            
                                            ),
                                        name="Cluster Center")
                    
                        
                    data7 = go.Data([trace0, trace1])
                    fig = go.Figure(data=data7)
                    
                    layout = go.Layout(
                                height=600, width=800, title=f"{calg} Cluster Size {c}",
                                xaxis=dict(
                                    title=f"Component {choosed_component[0]}",
                                ),
                                yaxis=dict(
                                    title=f"Component {choosed_component[1]}"
                                ) ) 
                    fig.update_layout(layout)
                    st.plotly_chart(fig)

                inertias=np.array(inertias).reshape(-1,2)
                performance = go.Scatter(x=inertias[:,0], y=inertias[:,1])
                layout = go.Layout(
                    title="Cluster Number vs Inertia",
                    xaxis=dict(
                        title="Ks"
                    ),
                    yaxis=dict(
                        title="Inertia"
                    ) ) 
                fig=go.Figure(data=go.Data([performance]))
                fig.update_layout(layout)
                st.plotly_chart(fig)
        
        
    else:
        st.markdown("### Please Select at Least 2 Features for Visualization.")
