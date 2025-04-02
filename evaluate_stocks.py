#!/usr/bin/env python

import argparse
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import os

__version__ = "0.0.4"

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
    
    # Check for invalid growth rates for terminal case
    if terminal_growth_rate >= discount_rate:
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

def calculate_wacc(stock, risk_free_rate=0.035, market_return=0.1, tax_rate=0.21):
    """
    This function computes the WACC for a given stock using financial data 
    retrieved from a yfinance Ticker object. The WACC is calculated as a 
    weighted average of the cost of equity and the after-tax cost of debt.

    Args:
        stock (yfinance.Ticker): The stock object containing financial data.
        risk_free_rate (float, optional): The risk-free rate, typically based 
            on the 10-year Treasury yield. Default is 0.035 (3.5%).
        market_return (float, optional): The expected market return. Default is 0.1 (10%).
        tax_rate (float, optional): The corporate tax rate. Default is 0.21 (21%).

    Returns:
        float: The calculated WACC value. Returns None if the calculation 
            is not possible due to missing or invalid data.

    Raises:
        Exception: If an error occurs during the calculation, it is caught 
            and logged, and the function returns None.
    """
    try:
        # Get required data
        market_price = stock.history(period="1d")["Close"].iloc[-1]
        shares_outstanding = stock.info.get("sharesOutstanding", 0)
        # to tet the most recent, the debt array should be sorted first based on
        # date, so we can use iloc[0] to get the most recent value
        # Get total debt from balance sheet
        # Note: "Total Debt" may not be available for all stocks,
        # so we need to handle this case
        if "Total Debt" not in stock.balance_sheet.index:
            print(f"Warning: 'Total Debt' not found in balance sheet for {stock.ticker}")
            return None

        total_debts = stock.balance_sheet.loc["Total Debt"].sort_index(ascending=False)
        total_debt = total_debts.iloc[0]  # Most recent
        if total_debt is None:
            print(f"Warning: Total debt doesn't exist for {stock.ticker}")
            return None

        # Calculate market cap
        market_cap = market_price * shares_outstanding

        # Calculate weights
        total_capital = market_cap + total_debt
        if total_capital <= 0:
            return None

        equity_weight = market_cap / total_capital
        debt_weight = total_debt / total_capital

        # Cost of Equity using CAPM
        beta = stock.info.get("beta", 1.0)  # Default to 1.0 if not available
        market_risk_premium = market_return - risk_free_rate  # Typical market risk premium: 6%, market return - risk-free rate
        cost_of_equity = risk_free_rate + beta * market_risk_premium

        # Cost of Debt (using latest interest expense / total debt)
        try:
            # sort the interest expense based on date
            interest_expenses = stock.financials.loc["Interest Expense"].sort_index(ascending=False).dropna()
            interest_expense = abs(interest_expenses.iloc[0])
            cost_of_debt = interest_expense / total_debt
        except:
            cost_of_debt = 0.05  # Default to 5% if calculation fails

        # Apply tax shield to cost of debt (assuming 21% corporate tax rate)
        after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)

        # Calculate WACC
        wacc = (equity_weight * cost_of_equity) + (debt_weight * after_tax_cost_of_debt)

        return max(wacc, 0.08)  # Ensure minimum discount rate of 8%

    except Exception as e:
        print(f"WACC calculation error: {str(e)}")
        return None

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
    parser.add_argument("--fixed-growth", type=float, default=0.10, 
                       help="Fixed growth rate for alternative DCF calculation (default: %(default)s)")
    parser.add_argument("--risk-free-rate", type=float, default=0.035, help="Risk-free rate for WACC calculation (default: %(default)s)")
    parser.add_argument("--market-return", type=float, default=0.1, help="Market return for WACC calculation (default: %(default)s)")
    parser.add_argument("--tax-rate", type=float, default=0.21, help="Tax rate for WACC calculation (default: %(default)s)")
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

def get_currency_rate(currency):
    """Get conversion rate from currency to USD"""
    rates = {
        'USD': 1.0,
        'TWD': 0.0313,  # Taiwan Dollar
        'CNY': 0.1382,  # Chinese Yuan
        'HKD': 0.1278,  # Hong Kong Dollar
        'EUR': 1.0843,  # Euro
        'JPY': 0.00673  # Japanese Yen
    }
    return rates.get(currency, 1.0)

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

            # Get currency information and conversion rate
            ## convert EPS and DCF values to USD, and keep others in original currency
            currency = stock.info.get('financialCurrency', 'USD')
            currency_rate = get_currency_rate(currency)
            
            # Extract metrics with dates for the specified number of years (no need for ::-1 anymore)
            revenue_data = financials.loc["Total Revenue"][:args.years]
            revenue = revenue_data.values
            revenue_dates = revenue_data.index
            
            eps_data = financials.loc["Net Income"][:args.years].values / stock.info.get("sharesOutstanding", 1)*currency_rate
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

            # Calculate WACC for discount rate
            wacc = calculate_wacc(
                stock,
                risk_free_rate=args.risk_free_rate,
                market_return=args.market_return,
                tax_rate=args.tax_rate
            )
            
            # Get current stock price
            current_price = stock.history(period="1d")["Close"].iloc[-1]
            
            # Calculate DCF with different scenarios (convert to USD)
            latest_fcf = fcf[0]
            DCF_base_metric = "Free Cash Flow"
            # Use net income as fallback if FCF is not available or negative
            if latest_fcf is None or latest_fcf <= 0:
                latest_fcf = financials.loc["Net Income"].values[0]
                DCF_base_metric = "Net Income"
            
            # Also calculate the current FCF per share and save in results
            current_fcf_per_share = latest_fcf / shares_outstanding * currency_rate
           
            # 1. DCF with revenue CAGR and WACC
            dcf_value_cagr_wacc = "N/A"
            if wacc is not None:
                dcf_value_cagr_wacc = calculate_dcf(
                    free_cash_flow=latest_fcf,
                    growth_rate=revenue_cagr,
                    shares_outstanding=shares_outstanding,
                    terminal_growth_rate=args.terminal_rate,
                    discount_rate=wacc,
                    years=args.years
                )
                if dcf_value_cagr_wacc != "N/A":
                    dcf_value_cagr_wacc *= currency_rate
            
            # 2. DCF with revenue CAGR and fixed discount rate
            dcf_value_cagr_fixed = calculate_dcf(
                free_cash_flow=latest_fcf,
                growth_rate=revenue_cagr,
                shares_outstanding=shares_outstanding,
                terminal_growth_rate=args.terminal_rate,
                discount_rate=args.discount_rate,
                years=args.years
            ) * currency_rate
            
            # 3. DCF with fixed growth rate and WACC
            dcf_value_fixed_wacc = "N/A"
            if wacc is not None:
                dcf_value_fixed_wacc = calculate_dcf(
                    free_cash_flow=latest_fcf,
                    growth_rate=args.fixed_growth,
                    shares_outstanding=shares_outstanding,
                    terminal_growth_rate=args.terminal_rate,
                    discount_rate=wacc,
                    years=args.years
                )
                if dcf_value_fixed_wacc != "N/A":
                    dcf_value_fixed_wacc *= currency_rate
            
            # 4. DCF with fixed growth rate and discount rate
            dcf_value_fixed_both = calculate_dcf(
                free_cash_flow=latest_fcf,
                growth_rate=args.fixed_growth,
                shares_outstanding=shares_outstanding,
                terminal_growth_rate=args.terminal_rate,
                discount_rate=args.discount_rate,
                years=args.years
            ) * currency_rate
            
            # Store results with currency information
            result = {
                "Symbol": symbol,
                "Revenue CAGR": revenue_cagr,
                "EPS CAGR": eps_cagr,
                "FCF CAGR": fcf_cagr,
                "Fixed growth rate": args.fixed_growth,
                "Fixed discount rate": args.discount_rate,
                "WACC": wacc if wacc is not None else "N/A",
                "DCF Base Metric": DCF_base_metric,
                "Current FCF per Share": current_fcf_per_share,  # Now in USD
                "DCF (CAGR growth, WACC)": dcf_value_cagr_wacc,  # Now in USD
                "DCF (CAGR growth, Fixed Rate)": dcf_value_cagr_fixed,  # Now in USD
                f"DCF ({args.fixed_growth:.1%} growth, WACC)": dcf_value_fixed_wacc,  # Now in USD
                f"DCF ({args.fixed_growth:.1%} growth, Fixed Rate)": dcf_value_fixed_both,  # Now in USD
                "Original Currency": currency,
                "Currency Rate": currency_rate,
                "Current Price": current_price,
                "ROE (Latest)": roe[0],
                "Net Margin (Latest)": net_margin[0],
                "Book Value Per Share (Latest)": book_value_per_share[0],
                "EPS (Latest)": eps_data[0],
                "Debt to Equity Ratio (Latest)": debt_to_equity[0],
                "Debt to Asset Ratio (Latest)": debt_to_assets[0],
                "Cash & Equivalents (Latest)": cash[0],
                "Total Debt (Latest)": total_debt[0],
                #"Discount Rate Used": discount_rate,
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