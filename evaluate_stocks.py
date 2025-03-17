#!/usr/bin/env python

import argparse
import sys
import yfinance as yf
import pandas as pd
import numpy as np

# Function to calculate Compound Annual Growth Rate (CAGR)
def calculate_cagr(data, years):
    start = data[0]
    end = data[-1]
    if start == 0 or end == 0:
        return 0
    return (end / start) ** (1 / years) - 1

# Function to calculate the Discounted Cash Flow (DCF) fair value
def calculate_dcf(free_cash_flow, growth_rate, discount_rate=0.1, terminal_growth_rate=0.03, years=5):
    dcf_value = 0
    for year in range(1, years + 1):
        future_fcf = free_cash_flow * (1 + growth_rate) ** year
        discounted_fcf = future_fcf / (1 + discount_rate) ** year
        dcf_value += discounted_fcf

    # Terminal Value
    terminal_value = (free_cash_flow * (1 + growth_rate) ** years * (1 + terminal_growth_rate)) / (
        discount_rate - terminal_growth_rate
    )
    discounted_terminal_value = terminal_value / (1 + discount_rate) ** years
    dcf_value += discounted_terminal_value

    return dcf_value

# Read stock symbols from input file
def read_stock_symbols(file_path):
    with open(file_path, "r") as file:
        symbols = [line.strip() for line in file if line.strip()]
    return symbols

# Parse command-line arguments using argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze stock financial data and calculate metrics.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", type=str, help="Comma-separated list of stock symbols (e.g., AAPL,MSFT,GOOGL).")
    group.add_argument("--file", type=str, help="Path to a file containing stock symbols (one per line).")
    parser.add_argument("--output", type=str, default="stock_analysis_results.csv", help="Path to save the output CSV file.")
    return parser.parse_args()

# Main function
def main():
    args = parse_args()

    # Determine stock symbols
    if args.list:
        symbols = [symbol.strip().upper() for symbol in args.list.split(",")]
    elif args.file:
        try:
            symbols = read_stock_symbols(args.file)
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
    else:
        print("Error: Either --list or --file must be provided.")
        sys.exit(1)

    # Output file
    output_file = args.output

    results = []

    for symbol in symbols:
        try:
            print(f"Processing {symbol}...")
            stock = yf.Ticker(symbol)

            # Get financial data
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow

            # Extract metrics for the last 5 years
            revenue = financials.loc["Total Revenue"][:5].values[::-1]  # Reverse for chronological order
            eps = financials.loc["Net Income"][:5].values[::-1] / stock.info.get("sharesOutstanding", 1)
            free_cash_flow = cashflow.loc["Free Cash Flow"][:5].values[::-1]
            debt_to_equity = balance_sheet.loc["Total Debt"][:5].values[::-1] / balance_sheet.loc["Stockholders Equity"][:5].values[::-1]
            debt_to_assets = balance_sheet.loc["Total Debt"][:5].values[::-1] / balance_sheet.loc["Total Assets"][:5].values[::-1]
            roe = financials.loc["Net Income"][:5].values[::-1] / balance_sheet.loc["Stockholders Equity"][:5].values[::-1]
            net_margin = financials.loc["Net Income"][:5].values[::-1] / financials.loc["Total Revenue"][:5].values[::-1]

            # Book Value Per Share
            book_value_per_share = balance_sheet.loc["Stockholders Equity"][:5].values[::-1] / stock.info.get("sharesOutstanding", 1)

            # Get years
            years = len(financials.columns[:5])

            # Calculate average annual growth rates
            revenue_cagr = calculate_cagr(revenue, years)
            eps_cagr = calculate_cagr(eps, years)
            fcf_cagr = calculate_cagr(free_cash_flow, years)

            # Calculate DCF fair value using 5-year growth rate and 10% discount rate
            latest_fcf = free_cash_flow[-1]
            dcf_value_5yr_growth = calculate_dcf(latest_fcf, growth_rate=revenue_cagr)
            dcf_value_10pct_growth = calculate_dcf(latest_fcf, growth_rate=0.10)

            # Get current stock price
            current_price = stock.history(period="1d")["Close"].iloc[-1]

            # Store results
            result = {
                "Symbol": symbol,
                "Revenue CAGR": revenue_cagr,
                "EPS CAGR": eps_cagr,
                "FCF CAGR": fcf_cagr,
                "DCF Fair Value (5yr Growth)": dcf_value_5yr_growth,
                "DCF Fair Value (10% Growth)": dcf_value_10pct_growth,
                "Current Price": current_price,
                "Debt to Equity Ratio (Latest)": debt_to_equity[-1],
                "Debt to Asset Ratio (Latest)": debt_to_assets[-1],
                "ROE (Latest)": roe[-1],
                "Net Margin (Latest)": net_margin[-1],
                "Book Value Per Share (Latest)": book_value_per_share[-1],
            }

            # Add historical data for each year
            for i, year in enumerate(financials.columns[:5][::-1]):
                result[f"Revenue ({year.year})"] = revenue[i]
                result[f"EPS ({year.year})"] = eps[i]
                result[f"FCF ({year.year})"] = free_cash_flow[i]
                result[f"Debt to Equity ({year.year})"] = debt_to_equity[i]
                result[f"Debt to Assets ({year.year})"] = debt_to_assets[i]
                result[f"ROE ({year.year})"] = roe[i]
                result[f"Net Margin ({year.year})"] = net_margin[i]
                result[f"Book Value Per Share ({year.year})"] = book_value_per_share[i]

            results.append(result)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to {output_file}.")

if __name__ == "__main__":
    main()
