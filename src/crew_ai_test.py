#!/usr/bin/env python
"""
Multi-Agent Financial Analysis Report from SEC 10-K Filings

This script performs the following:
1. Fetches SEC 10-K filings by directly scraping the SEC EDGAR index pages.
2. Saves the fetched filings data as a JSON file.
3. Uses a multi-agent Crew AI chain (with Agents for planning, writing, and editing)
   to generate a detailed financial analysis report that compares filings across companies
   and years, highlights trends, and shows year-over-year improvements.
4. Incorporates external tools (for scraping and semantic search) in the multi-agent tasks.
5. Contains error handling and a TEST_MODE flag for flexible usage.
"""


import os
import sys
import json
import subprocess
import shutil
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

# --------------------------
# Utility Functions & Package Installation
# --------------------------
def install_if_needed(package):
    """Ensure a package is installed."""
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure required packages are installed.
for pkg in ["pandas", "openpyxl", "requests", "beautifulsoup4"]:
    install_if_needed(pkg)
for pkg in ["crewai", "crewai_tools", "langchain_community"]:
    install_if_needed(pkg)

# --------------------------
# Set up Environment Variables & Global Constants
# --------------------------
# Update these with your actual API keys if not set via your environment.
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
if "SERPER_API_KEY" not in os.environ:
    os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY"
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"  # Change if needed

# Global SEC request header – update your email accordingly.
HEADERS = {
    "User-Agent": "FinancialReport/1.0 (your_email@example.com)"
}

# --------------------------
# SEC 10-K Data Fetching Functions
# --------------------------
def get_all_tickers_info():
    """
    Fetch the SEC-provided JSON mapping of company tickers to CIK numbers.
    Returns:
        dict: Mapping data.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching tickers info: {e}")
        return {}

def find_cik(ticker, tickers_data):
    """
    Look up the CIK for the given ticker.
    Args:
        ticker (str): Company ticker.
        tickers_data (dict): Mapping data.
    Returns:
        str or None: The CIK if found.
    """
    try:
        for _, info in tickers_data.items():
            if info["ticker"].lower() == ticker.lower():
                return info["cik_str"]
    except Exception as e:
        print(f"Error finding CIK for {ticker}: {e}")
    return None

def fetch_10k_text_and_link_from_index(index_url):
    """
    Given an EDGAR index page URL for a filing, parse the document table to extract:
      - The primary 10-K document link (or the complete submission text file, if available),
      - The filing text, and
      - Any graphic file references.
    Returns:
        dict: {"tenk_link": URL, "tenk_text": filing text, "graphics": [ ... ]}
    """
    result = {"tenk_link": None, "tenk_text": "", "graphics": []}
    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("table.tableFile tr")
        tenk_link = None
        complete_sub_url = None

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            description = cells[1].get_text(strip=True).lower()
            doc_type = cells[3].get_text(strip=True).lower()
            link_tag = cells[2].find("a")
            if not link_tag:
                continue
            href = link_tag.get("href", "")
            if href.startswith("/"):
                href = "https://www.sec.gov" + href

            # Primary 10-K document link.
            if doc_type in ["10-k", "10-k/a"]:
                tenk_link = href

            # Use complete submission text file if available.
            if "complete submission text file" in description:
                complete_sub_url = href

            # Collect any graphics references.
            if doc_type == "graphic":
                filename = cells[2].get_text(strip=True)
                result["graphics"].append({
                    "filename": filename,
                    "graphic_url": href,
                    "local_path": ""
                })

        # Prefer complete submission text file.
        if complete_sub_url:
            r_txt = requests.get(complete_sub_url, headers=HEADERS, timeout=20)
            r_txt.raise_for_status()
            result["tenk_link"] = complete_sub_url
            result["tenk_text"] = r_txt.text
            return result

        # Else fallback to primary 10-K document.
        if tenk_link:
            r2 = requests.get(tenk_link, headers=HEADERS, timeout=20)
            r2.raise_for_status()
            soup2 = BeautifulSoup(r2.text, "html.parser")
            text = soup2.get_text(separator="\n")
            result["tenk_link"] = tenk_link
            result["tenk_text"] = text
            return result

        print(f"Could not find a 10-K document link at index URL: {index_url}")
        return result
    except Exception as e:
        print(f"Error fetching 10-K text from index URL: {e}")
        return result

def fetch_10k_filings_for_year(cik, year):
    """
    Fetch all SEC 10-K filings for a given CIK and year.
    Args:
        cik (str): Company CIK.
        year (int): Filing year.
    Returns:
        list: Filings details.
    """
    base_url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
    results = []
    try:
        r = requests.get(base_url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            print(f"Status code {r.status_code} for CIK {cik} in {year}")
            return results
        data = r.json()
    except Exception as e:
        print(f"Error fetching submission data for CIK {cik} in {year}: {e}")
        return results

    try:
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        for i in range(len(forms)):
            form_type = forms[i]
            filing_date = dates[i]
            accession_no = accessions[i]
            if form_type in ["10-K", "10-K/A"] and filing_date.startswith(str(year)):
                doc_index_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{accession_no.replace('-', '')}/{accession_no}-index.htm"
                )
                tenk_data = fetch_10k_text_and_link_from_index(doc_index_url)
                results.append({
                    "form_type": form_type,
                    "filing_date": filing_date,
                    "accession_no": accession_no,
                    "doc_index_url": doc_index_url,
                    "tenk_link": tenk_data.get("tenk_link"),
                    "tenk_text": tenk_data.get("tenk_text"),
                    "graphics": tenk_data.get("graphics", [])
                })
    except Exception as e:
        print(f"Error processing filings for CIK {cik} in {year}: {e}")
    return results

# --------------------------
# Crew AI Multi-Agent Setup for Financial Analysis
# --------------------------
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI

# Instantiate external tools for additional context.
scrape_tool = ScrapeWebsiteTool(website_url="https://www.sec.gov/Archives/edgar/data/")
search_tool = WebsiteSearchTool(website_url="https://www.sec.gov/Archives/edgar/data/")

# Define multi-agent roles with detailed backstories and goals.
planner = Agent(
    role="Financial Analyst Planner",
    goal="Plan a detailed financial analysis report using SEC 10-K filings data across multiple companies and years. "
         "Focus on identifying trends, year-over-year changes, and key comparisons.",
    backstory="You are a seasoned financial analyst proficient in synthesizing data from SEC filings and external market data. "
              "Your task is to generate an outline that clearly presents comparisons and trends among companies.",
    verbose=True,
    allow_delegation=False,
    tools=[scrape_tool, search_tool]
)

writer = Agent(
    role="Financial Report Writer",
    goal="Draft a comprehensive financial analysis report based on the outline. "
         "Ensure the report highlights comparisons, trends, and improvements over time.",
    backstory="You are an expert financial writer with a deep understanding of financial reports. "
              "Using the outline and external context, you produce a clear and insightful report.",
    verbose=True,
    allow_delegation=False
)

editor = Agent(
    role="Financial Report Editor",
    goal="Review and edit the drafted report to maximize clarity, accuracy, and professional quality. "
         "Ensure that all comparisons and trends are clearly communicated.",
    backstory="As an experienced editor in financial communication, you refine reports to meet high standards for precision and clarity.",
    verbose=True,
    allow_delegation=False
)

# Define tasks for each agent.
plan_task = Task(
    description=(
        "Analyze the provided SEC 10-K filings JSON data (covering multiple companies and years). "
        "Utilize external context via website scraping and semantic search to enrich your analysis. "
        "Generate a detailed outline for a financial analysis report covering overall trends, "
        "year-over-year improvements, and cross-company comparisons."
    ),
    expected_output=(
        "A structured content plan outlining sections such as an introduction, comparative analysis, "
        "trend identification, anomalies, and a conclusion with actionable insights."
    ),
    tools=[scrape_tool, search_tool],
    agent=planner
)

write_task = Task(
    description=(
        "Based on the outline provided by the Financial Analyst Planner, draft a complete financial analysis report. "
        "Ensure the report clearly compares SEC 10-K filings across companies and over the years, highlighting trends, "
        "year-over-year improvements, and potential areas of concern."
    ),
    expected_output=(
        "A thorough, well-structured financial analysis report with distinct sections and clear comparisons."
    ),
    agent=writer
)

edit_task = Task(
    description=(
        "Review the draft financial analysis report for clarity, accuracy, and consistency. "
        "Refine the report to ensure it meets professional standards and that all comparisons and trends are effectively communicated."
    ),
    expected_output=(
        "A polished, publication-ready financial analysis report."
    ),
    agent=editor
)

# Assemble the Crew with a hierarchical process and enable memory.
financial_crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan_task, write_task, edit_task],
    manager_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
    process=Process.hierarchical,
    verbose=True,
    memory=True,
)

# --------------------------
# Main Function: Integration & Execution
# --------------------------
def main():
    try:
        # Set TEST_MODE to process a single company-year pair for testing.
        TEST_MODE = True
        if TEST_MODE:
            companies = [("AAPL", 2023)]
        else:
            excel_file = "Data_companies_list.xlsx"  # Update path as needed.
            df = pd.read_excel(excel_file)
            companies = [(str(row["Symbol"]).strip(), year)
                         for _, row in df.iterrows()
                         for year in range(2012, 2026)]
        
        # Dictionary to store fetched filings data.
        filings_data = {}
        tickers_data = get_all_tickers_info()
        if not tickers_data:
            print("No tickers data fetched; exiting.")
            return

        for ticker, year in companies:
            cik = find_cik(ticker, tickers_data)
            if not cik:
                print(f"CIK not found for ticker: {ticker}")
                continue
            print(f"Fetching 10-K filings for {ticker} in {year} ...")
            filings = fetch_10k_filings_for_year(cik, year)
            if filings:
                filings_data[f"{ticker}_{year}"] = filings
            else:
                print(f"No 10-K filings found for {ticker} in {year}")
            time.sleep(0.2)  # Respect SEC rate limits.

        # Save the fetched filings data for future reference.
        output_dir = "sec_10k_data"
        os.makedirs(output_dir, exist_ok=True)
        filings_data_file = os.path.join(output_dir, "filings_data.json")
        try:
            with open(filings_data_file, "w", encoding="utf-8") as f:
                json.dump(filings_data, f, indent=2)
        except Exception as e:
            print(f"Error saving filings data: {e}")

        # Prepare input for the multi-agent process. The input is the JSON data as a string.
        input_text = json.dumps(filings_data, indent=2)[:10000]  # Truncate if necessary.
        
        # Kick off the multi-agent process to generate the financial analysis report.
        result = financial_crew.kickoff(inputs={"topic": input_text})
        
        # Save the generated report.
        report_file = os.path.join(output_dir, "financial_analysis_report.txt")
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(result)
        except Exception as e:
            print(f"Error saving the financial report: {e}")
        
        print("Financial analysis report generated successfully.")
        print(f"Report saved at: {report_file}")
    
    except Exception as e:
        print(f"Unexpected error during main execution: {e}")

if __name__ == "__main__":
    main()
