import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import pandas as pd
import cufflinks

@st.cache
def fetch_and_clean_data(url):
     # Fetch data from URL here, and then clean it up.
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df.date).dt.date
    df['date'] = pd.DatetimeIndex(df.date)
    # df = df[df.date>pd.to_datetime("2022-01-01")]


    return df

url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
data = fetch_and_clean_data(url)

countries = data.location.unique().tolist()

sidebar = st.sidebar
country_selector = sidebar.selectbox(
    "Select a Location",
    countries
)
trend = sidebar.selectbox("Select a Trend Type", ["Daily", "Weekly", "Monthly", "Yearly"])
levels = {"Daily":"1D", "Weekly":"1W", "Monthly":"1M", "Yearly":"1Y"}

subplots = sidebar.checkbox("Subplots")
show_data = sidebar.checkbox("Show Data")
sidebar.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
st.markdown(f"# Currently Selected {country_selector}")


if show_data:
    st.markdown(f"## Data of {country_selector}")
    tdf = data.query(f"location=='{country_selector}'").groupby(pd.Grouper(key="date", freq=levels[trend])).agg("sum").reset_index()
    tdf["date"] = tdf.date.dt.date
    st.dataframe(tdf)


new_cases = sidebar.checkbox("New Cases")
new_deaths = sidebar.checkbox("New Deaths")
new_vaccinations = sidebar.checkbox("New Vaccinations")
new_tests = sidebar.checkbox("New Tests")

lines = [new_cases, new_deaths, new_vaccinations, new_tests]
line_cols = ["new_cases", "new_deaths", "new_vaccinations", "new_tests"]

trends = [c[1] for c in zip(lines,line_cols) if c[0]==True]


if len(trends)>0:
    st.markdown(f"## {trend} Trend of {country_selector}.")
    mdf = data.query(f"location=='{country_selector}'")
    tdf = mdf.groupby(pd.Grouper(key="date", freq=levels[trend])).aggregate(new_cases=("new_cases", "sum"),
                                   new_deaths = ("new_deaths", "sum"),
                                   new_vaccinations = ("new_vaccinations", "sum"),
                                   new_tests = ("new_tests", "sum")).reset_index()

    tdf["date"] = tdf["date"].dt.date
    # fig = px.line(tdf, x="date", y=line_plots, width=800, height=600,
    #  title=f'Trend of {country_selector}', subplot=True)
    fig = tdf.iplot(kind="line", asFigure=True, x="date", y=trends, xTitle='Date', yTitle="Values",
                    title=f"{trend} Trend of {', '.join(trends)}.", subplots=subplots)
    st.plotly_chart(fig, use_container_width=False)
