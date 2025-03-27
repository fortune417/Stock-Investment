#!/usr/bin/env python

import argparse
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import os

__version__ = "0.0.2"

# Function to calculate Compound Annual Growth Rate (CAGR)
def calculate_cagr(data, dates):
    """
    Calculate Compound Annual Growth Rate (CAGR) considering actual time periods between valid data points
    
    Args:
        data: numpy array or pandas Series of values
        dates: array-like of dates (can be strings in 'YYYY-MM-DD' format or datetime objects)
    Returns:
        float: CAGR value, or 0 if calculation is not possible
    """
    # Convert to numpy arrays with float type
    data = np.array(data, dtype=float)
    
    # Convert dates to pandas datetime
    dates = pd.to_datetime(dates)
    
    # Create mask for valid data (not NaN and not infinite)
    mask = ~np.isnan(data) & ~np.isinf(data)
    valid_data = data[mask]
    valid_dates = dates[mask]
    
    if len(valid_data) < 2:
        return 0
    
    try:
        # Sort dates in chronological order and get sorting indices
        sort_idx = np.argsort(valid_dates)
        valid_dates = valid_dates[sort_idx]
        valid_data = valid_data[sort_idx]
        
        # Calculate year-over-year growth rates with actual time periods
        growth_rates = []
        for i in range(len(valid_data)-1):
            if valid_data[i] <= 0 or valid_data[i+1] <=0:  # Skip negative or zero values
                continue
                
            # Calculate actual years between data points
            years_between = (valid_dates[i+1] - valid_dates[i]).days / 365.25
            # assume the data are in chronological order
            if years_between <= 0:
                continue
                
            # Calculate annualized growth rate between these points
            growth_rate = (valid_data[i+1] / valid_data[i]) ** (1/years_between) - 1
            growth_rates.append(growth_rate)
        
        if not growth_rates:
            return 0
        
        # Calculate geometric mean of (1 + growth_rate)
        growth_factors = np.array([1 + r for r in growth_rates])
        geometric_mean = np.exp(np.mean(np.log(growth_factors)))
        
        return geometric_mean - 1
        
    except Exception as e:
        print(f"CAGR calculation error: {str(e)}")
        return 0

# Function to calculate the Discounted Cash Flow (DCF) fair value
def calculate_dcf(free_cash_flow, growth_rate, shares_outstanding, discount_rate=0.1, terminal_growth_rate=0.05, years=5):
    """Calculate DCF fair value per share"""
    if shares_outstanding <= 0:
        return 0
        
    dcf_value = 0
    for year in range(1, years + 1):
        future_fcf = free_cash_flow * (1 + growth_rate) ** year
        discounted_fcf = future_fcf / (1 + discount_rate) ** year
        dcf_value += discounted_fcf

    # Terminal Value calculation (using perpetuity growth formula)
    final_year_fcf = free_cash_flow * (1 + growth_rate) ** years
    terminal_value = final_year_fcf * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    discounted_terminal_value = terminal_value / (1 + discount_rate) ** years
    
    # Add terminal value to DCF and convert to per-share value
    total_value = dcf_value + discounted_terminal_value
    return total_value / shares_outstanding

# Read stock symbols from input file
def read_stock_symbols(file_path):
    with open(file_path, "r") as file:
        symbols = [line.strip() for line in file if line.strip()]
        # Remove duplicates and convert to uppercase
        symbols = list(set([symbol.upper() for symbol in symbols]))
    return symbols

# Parse command-line arguments using argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze stock financial data and calculate metrics.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", type=str, help="Comma-separated list of stock symbols (e.g., AAPL,MSFT,GOOGL).")
    group.add_argument("--file", type=str, help="Path to a file containing stock symbols (one per line).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--output", type=str, default="stock_analysis_results.csv", help="Path to save the output CSV file. (default: %(default)s)")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to save output files (default: current directory)")
    parser.add_argument("--years", type=int, default=5, help="Number of years of historical data to analyze (default: %(default)s)")
    parser.add_argument("--terminal-rate", type=float, default=0.05, help="Terminal growth rate for DCF calculation (default: %(default)s)")
    parser.add_argument("--discount-rate", type=float, default=0.1, help="Discount rate for DCF calculation (default: %(default)s)")
    return parser.parse_args()

def save_financial_data(data, symbol, data_type, output_dir):
    """Save financial data to CSV file with date-based filename"""
    if data is None or data.empty:
        return
        
    # Get current date for filename
    current_date = pd.Timestamp.now().strftime('%Y%m')
    filename = f"{current_date}_{symbol}_{data_type}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save to CSV
    data.to_csv(filepath)
    print(f"Saved {data_type} data to {filepath}")

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
    
    # Update output file path to use output directory
    output_file = os.path.join(args.outdir, args.output)

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

    results = []

    for symbol in symbols:
        try:
            print(f"Processing {symbol}...")
            stock = yf.Ticker(symbol)

            # Validate stock data
            if not stock or not stock.info:
                print(f"Warning: Could not fetch data for {symbol}")
                continue

            # Get financial data with validation
            financials = stock.financials.sort_index(ascending=False) # sorting rows
            balance_sheet = stock.balance_sheet.sort_index(ascending=False)
            cashflow = stock.cashflow.sort_index(ascending=False)
            
            # Align indices and fill missing years with NaN
            ## sort columns in reverse chronological order
            if financials is not None and balance_sheet is not None and cashflow is not None:
                all_years = financials.columns.union(balance_sheet.columns).union(cashflow.columns)
                all_years = sorted(all_years, reverse=True)  # Sort years in descending order
                financials = financials.reindex(columns=all_years)
                balance_sheet = balance_sheet.reindex(columns=all_years)
                cashflow = cashflow.reindex(columns=all_years)
            else:
                print(f"Warning: Missing financial data for {symbol}")
                continue

            # Save detailed financial data
            save_financial_data(financials, symbol, "financials", args.outdir)
            save_financial_data(balance_sheet, symbol, "balancesheet", args.outdir)
            save_financial_data(cashflow, symbol, "cashflow", args.outdir)

            # Extract metrics with dates for the specified number of years (no need for ::-1 anymore)
            revenue_data = financials.loc["Total Revenue"][:args.years]
            revenue = revenue_data.values
            revenue_dates = revenue_data.index
            
            eps_data = financials.loc["Net Income"][:args.years].values / stock.info.get("sharesOutstanding", 1)
            eps_dates = financials.loc["Net Income"][:args.years].index
            
            fcf_data = cashflow.loc["Free Cash Flow"][:args.years]
            fcf = fcf_data.values
            fcf_dates = fcf_data.index
            
            # Extract cash and debt data
            cash_data = balance_sheet.loc["Cash And Cash Equivalents"][:args.years]
            cash = cash_data.values
            total_debt_data = balance_sheet.loc["Total Debt"][:args.years]
            total_debt = total_debt_data.values

            # Calculate ratios (data is already sorted)
            debt_to_equity = balance_sheet.loc["Total Debt"][:args.years].values / balance_sheet.loc["Stockholders Equity"][:args.years].values
            debt_to_assets = balance_sheet.loc["Total Debt"][:args.years].values / balance_sheet.loc["Total Assets"][:args.years].values
            roe = financials.loc["Net Income"][:args.years].values / balance_sheet.loc["Stockholders Equity"][:args.years].values
            net_margin = financials.loc["Net Income"][:args.years].values / financials.loc["Total Revenue"][:args.years].values

            # Book Value Per Share
            book_value_per_share = balance_sheet.loc["Stockholders Equity"][:args.years].values / stock.info.get("sharesOutstanding", 1)

            # Calculate average annual growth rates with dates
            revenue_cagr = calculate_cagr(revenue, revenue_dates)
            eps_cagr = calculate_cagr(eps_data, eps_dates)
            fcf_cagr = calculate_cagr(fcf, fcf_dates)

            # Get shares outstanding with validation
            shares_outstanding = stock.info.get("sharesOutstanding", 0)
            if shares_outstanding <= 0:
                print(f"Warning: Invalid shares outstanding for {symbol}")
                continue

            # Calculate DCF fair value using revenue growth rate and user-specified terminal rate
            latest_fcf = fcf[0]
            dcf_value_5yr_growth = calculate_dcf(
                free_cash_flow=latest_fcf,
                growth_rate=revenue_cagr,
                shares_outstanding=shares_outstanding,
                terminal_growth_rate=args.terminal_rate,
                discount_rate=args.discount_rate,
                years=args.years
            )
            dcf_value_10pct_growth = calculate_dcf(
                free_cash_flow=latest_fcf,
                growth_rate=0.10,
                shares_outstanding=shares_outstanding,
                terminal_growth_rate=args.terminal_rate,
                discount_rate=args.discount_rate,
                years=args.years
            )

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
                "Debt to Equity Ratio (Latest)": debt_to_equity[0],
                "Debt to Asset Ratio (Latest)": debt_to_assets[0],
                "ROE (Latest)": roe[0],
                "Net Margin (Latest)": net_margin[0],
                "Book Value Per Share (Latest)": book_value_per_share[0],
                "Cash & Equivalents (Latest)": cash[0],
                "Total Debt (Latest)": total_debt[0],
            }

            # Add historical data for each year
            for i, year in enumerate(all_years[:args.years]):  # Use all_years instead of financials.columns
                result[f"Revenue ({year.year})"] = revenue[i]
                result[f"EPS ({year.year})"] = eps_data[i]
                result[f"FCF ({year.year})"] = fcf[i]
                result[f"Debt to Equity ({year.year})"] = debt_to_equity[i]
                result[f"Debt to Assets ({year.year})"] = debt_to_assets[i]  # Fixed truncated line
                result[f"ROE ({year.year})"] = roe[i]
                result[f"Net Margin ({year.year})"] = net_margin[i]
                result[f"Book Value Per Share ({year.year})"] = book_value_per_share[i]
                result[f"Cash & Equivalents ({year.year})"] = cash[i]
                result[f"Total Debt ({year.year})"] = total_debt[i]

            results.append(result)

        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to {output_file}.")

if __name__ == "__main__":
    main()