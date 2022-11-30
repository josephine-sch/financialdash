# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import streamlit as st
import datetime
from datetime import date, datetime, timedelta
import yahoo_fin.stock_info as si
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas_datareader.data as web
import datetime as dt

duration = {'DurationText': ['1M', '3M', '6M', 'YTD', '1Y', '2Y', '5Y', 'MAX'],
              'Duration': [30, 90, 120, 335, 365, 730, 1825, 18250]}
dfduration = pd.DataFrame(duration)
end_date = datetime.today().date()


def tabdes():
    st.caption('Company Description')
    string_logo = '<img src=%s width = 25>' % tickerData.info['logo_url']
    string_name = tickerData.info['longName']
    st.markdown(f'## {string_logo} {string_name}', unsafe_allow_html=True)
    string_summary = tickerData.info['longBusinessSummary']
    st.info(string_summary)
    st.caption('Top 5 Shareholders')
    tickerData.institutional_holders[:5]


def tabsum():
    st.title(tickerSymbol)
    SelectPeriod = st.radio("select period", dfduration['DurationText'], horizontal=True)

    def tabdf(SelectPeriod):

        datetabdf = end_date - timedelta(dfduration.loc[dfduration['DurationText'] == SelectPeriod, 'Duration'].iloc[0].item())
        lastdate = si.get_data(tickerSymbol, start_date=datetabdf, end_date=end_date)

        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax2 = ax1.twinx()
        ax1.plot(lastdate['close'], color='green', lw=4)
        plt.fill_between(lastdate.index, lastdate['close'], color='green')
        ax1.set_ylabel("Closing price", color='black', fontsize=14)
        ax1.tick_params(axis="y", labelcolor='black')

        ax2.bar(lastdate.index, lastdate['volume'], color='tab:green', edgecolor="black", width=1.0)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volume (Billion $)", color='black', fontsize=14)
        ax2.tick_params(axis="y", labelcolor='black')
        ax2.set_ylim(0, ((lastdate['volume'].max()) * 5))  # diminishing the scale of the bar char
        ax2.bar(lastdate.index, lastdate['volume'], color='tab:green', edgecolor="black", width=1.0)

        fig.autofmt_xdate()
        fig.suptitle("NasdaqGS - NasdaqGS Real time exchange rate. USD currency", fontsize=20);

        st.pyplot(fig)

    tabdf(SelectPeriod=SelectPeriod)

    col1, col2 = st.columns(2)
    Tabsummary = si.get_quote_table(tickerSymbol, dict_result=False) # http://theautomatic.net/yahoo_fin-documentation/
    Tabsummary['value'] = Tabsummary['value'].astype(str)
    with col1: st.dataframe(Tabsummary.iloc[:8, :])
    with col2: st.dataframe(Tabsummary.iloc[8:, ])


def tabchart():
    st.title(tickerSymbol)
    col1, col2 = st.columns(2)
    sd1= col1.date_input("Start date", datetime.today().date() - timedelta(days=30))
    ed1= col2.date_input("End date", datetime.today().date())

    SelectPeriodchart = st.radio("select period", dfduration['DurationText'],horizontal=True)  # https://docs.streamlit.io/library/api-reference/widgets/st.radio
    datetabdf2 = end_date - timedelta(dfduration.loc[dfduration['DurationText'] == SelectPeriodchart, 'Duration'].iloc[0].item())
    interval = {'IntervalButton': ['Daily', 'Weekly', 'Monthly'], 'IntervalCode': ['1d', '1wk', '1mo']}
    interval2 = pd.DataFrame(interval)
    IB = st.radio('Interval', interval2['IntervalButton'])
    plottab = st.radio('Plot type', ['Line', 'Candle'])

    if plottab== 'Line':

        def tabchart2(ed1,datetabdf2, IB):

            lastdate2 = si.get_data(tickerSymbol, start_date=datetabdf2, end_date=ed1, interval=(
                interval2.loc[interval2['IntervalButton'] == IB, 'IntervalCode'].iloc[0]))
            fig, ax1 = plt.subplots(figsize=(8, 8))
            ax2 = ax1.twinx()
            ax1.plot(lastdate2['close'], color='green', lw=4)
            ax1.plot(lastdate2['close'].rolling(50).mean(), color='purple', label="50MA")
            ax1.set_ylabel("Closing price", color='black', fontsize=15)
            ax1.tick_params(axis="y", labelcolor='black')

            ax2.bar(lastdate2.index, lastdate2['volume'], color='tab:green', edgecolor="black", width=1.0)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Volume (Billion $)", color='black', fontsize=15)
            ax2.tick_params(axis="y", labelcolor='black')
            ax2.set_ylim(0, ((lastdate2['volume'].max()) * 5))  # diminishing the scale of the bar char
            ax2.bar(lastdate2.index, lastdate2['volume'], color='tab:green', edgecolor="black", width=1.0)

            fig.autofmt_xdate()
            fig.suptitle("NasdaqGS - Real time exchange rate. USD Currency", fontsize=15);

            st.pyplot(fig)
        tabchart2(ed1=ed1, IB=IB, datetabdf2=datetabdf2)
    else:
        lastdate3 = si.get_data(tickerSymbol, start_date=datetabdf2, end_date=ed1, interval=(
        interval2.loc[interval2['IntervalButton'] == IB, 'IntervalCode'].iloc[0]))
        fig = go.Figure(data=[go.Candlestick(x=lastdate3.index,
                                     open=lastdate3['open'],
                                     high=lastdate3['high'],
                                     low=lastdate3['low'],
                                     close=lastdate3['close'],)])
        fig.update_layout(xaxis_rangeslider_visible=False)

        st.plotly_chart(fig)


def tabfi():

    st.caption('In this page, find financials data on your selected company :')
    st.title(tickerSymbol)
    type1 = st.radio('Please select:', ['Income Statement', 'Balance Sheet', 'Cash Flow'], horizontal=True)
    time = {'Report': ['Annual', 'Quarterly'], 'Boolean': [True, False]}
    timedf = pd.DataFrame(time)
    typetime = st.radio('Report:', timedf['Report'], horizontal=True)

    def tabfidf(type1,typetime):
        if type1 == 'Income Statement':
            st.subheader(typetime + ' Income Statement for ' + tickerSymbol)
            Report = si.get_income_statement(tickerSymbol, yearly=(timedf.loc[timedf['Report'] == typetime, 'Boolean'].iloc[0]))
            Report['TTM'] = Report.sum(axis=1)
        elif type1 == 'Balance Sheet':
            st.subheader(typetime + ' Balance Sheet for ' + tickerSymbol)
            Report = si.get_cash_flow(tickerSymbol, yearly=(timedf.loc[timedf['Report'] == typetime, 'Boolean'].iloc[0]))
            Report['TTM'] = Report.sum(axis=1)
        elif type1 == 'Cash Flow':
            st.subheader(typetime + ' Cash Flow for ' + tickerSymbol)
            Report = si.get_balance_sheet(tickerSymbol, yearly=(timedf.loc[timedf['Report'] == typetime, 'Boolean'].iloc[0]))
            Report['TTM'] = Report.sum(axis=1)
        return st.dataframe(Report)
    tabfidf(type1=type1, typetime=typetime)

# cours 3
def tabMCS():

    nbsimu = st.radio('Number of simulations', [200, 500, 1000], horizontal= True)
    time = st.radio('Time horizon', [30, 60, 90], horizontal= True)
    st.subheader('MonteCarlo simulation for ' + tickerSymbol)

    class MonteCarlo(object):

        def __init__(self, tickerSymbol, data_source, start_date, end_date, time, nbsimu, seed):

            # Initiate class variables
            self.ticker = tickerSymbol
            self.data_source = data_source
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
            self.time = time
            self.nbsimu = nbsimu
            self.seed = seed
            self.simulation_df = pd.DataFrame()
            self.stock_price = web.DataReader(tickerSymbol, data_source, self.start_date, self.end_date)

            self.daily_return = self.stock_price['Close'].pct_change()
            # Volatility (of close price)
            self.daily_volatility = np.std(self.daily_return)

        def run_simulation(self):

            # Run the simulation
            np.random.seed(self.seed)
            self.simulation_df = pd.DataFrame()  # Reset

            for i in range(self.nbsimu):

                # The list to store the next stock price
                next_price = []

                # Create the next stock price
                last_price = self.stock_price['Close'][-1]

                for j in range(self.time):
                    # Generate the random percentage change around the mean (0) and std (daily_volatility)
                    future_return = np.random.normal(0, self.daily_volatility)

                    # Generate the random future price
                    future_price = last_price * (1 + future_return)

                    # Save the price and go next
                    next_price.append(future_price)
                    last_price = future_price

                # Store the result of the simulation
                self.simulation_df[i] = next_price

        def plot_simulation_price(self):

            # Plot the simulation stock price in the future
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 10, forward=True)

            plt.plot(self.simulation_df)
            plt.title('Monte Carlo simulation for ' + self.ticker + \
                      ' stock price in next ' + str(self.time) + ' days')
            plt.xlabel('Day')
            plt.ylabel('Price')

            plt.axhline(y=self.stock_price['Close'][-1], color='red')
            plt.legend(['Current stock price is: ' + str(np.round(self.stock_price['Close'][-1], 2))])
            ax.get_legend().legendHandles[0].set_color('red')

            st.pyplot(fig)

        def value_at_risk(self):
            # Price at 95% confidence interval
            future_price_95ci = np.percentile(self.simulation_df.iloc[-1:, :].values[0,], 5)
            # Value at Risk
            VaR = self.stock_price['Close'][-1] - future_price_95ci
            st.subheader('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')

    today = datetime.today().date().strftime('%Y-%m-%d')
    MCS = MonteCarlo(tickerSymbol=tickerSymbol, data_source='yahoo',
                     start_date='2022-01-01', end_date=today,
                     time=time, nbsimu=nbsimu, seed=123)
    MCS.run_simulation()
    MCS.plot_simulation_price()
    MCS.value_at_risk()

def tabMOA():
    st.title('Earnings for:' + tickerSymbol)
    earnings = tickerData.earnings
    st.table(earnings)
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.plot(earnings['Earnings'], color='green', lw=4)
    ax1.set_ylabel("Earning in $", color='black', fontsize=15)
    ax1.tick_params(axis="y", labelcolor='black')
    ax2 = ax1.twinx()
    ax2.plot(earnings['Revenue'], color='Orange', lw=4)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Revenues", color='black', fontsize=15)
    ax2.tick_params(axis="y", labelcolor='black')

    fig.autofmt_xdate()
    fig.suptitle("Evolution of earnings trough years", fontsize=15);

    plt.legend(['Earnings'])
    plt.legend(['Revenue'])
    st.pyplot(fig)


def sidebar():
    ticker_list = ['-'] + si.tickers_sp500()

    global tickerSymbol
    tickerSymbol = st.sidebar.selectbox("Select a ticker", ticker_list)
    global tickerData
    tickerData = yf.Ticker(tickerSymbol)
    ticker_data = yf.download(tickerSymbol)
    ticker_data.reset_index(inplace=True)

    run_button = st.sidebar.button('Update Data')
    if run_button:
        st.experimental_rerun()

    select_tab = st.sidebar.radio("Select tab",
                                  ['Description', 'Summary', 'Chart', 'Financials', 'Monte Carlo simulation','My analysis'], default=default)

    # Show the selected tab
    if select_tab == 'Description':
        tabdes()
    elif select_tab == 'Summary':
        tabsum()
    elif select_tab == 'Chart':
        tabchart()
    elif select_tab == 'Financials':
        tabfi()
    elif select_tab == 'Monte Carlo simulation':
        tabMCS()
    elif select_tab == 'My analysis':
        tabMOA()


if __name__ == "__main__":
    sidebar()


