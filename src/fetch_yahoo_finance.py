import yfinance as yf
import json

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = {
        'ticker': ticker,
        'financials': stock.financials.to_dict(),
        'balance_sheet': stock.balance_sheet.to_dict(),
        'cashflow': stock.cashflow.to_dict()
    }
    with open(f'../data/yahoo_reports/{ticker}.json', 'w') as f:
        json.dump(data, f, indent=4)

fetch_stock_data('AAPL')
