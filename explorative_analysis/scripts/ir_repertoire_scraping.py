import re
import time
import argparse

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
import traceback


def return_back_to_metadata(driver, wait):
    print("Returning to metadata page...")
    driver.back()
    time.sleep(2)

    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "td.col_ir_sequence_count a")))
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "td.col_repertoire_id")))
    #wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "td.col_study_id a")))

    try:
        wait.until(lambda d: d.find_element(By.CSS_SELECTOR, "td.col_study_id a") or
                             d.find_element(By.CSS_SELECTOR, "td.col_study_id span"))
    except TimeoutException:
        print("Warning: Could not find expected study_id element.")

    return driver


def get_repertoires_from_web(driver, start_query_id: str, min_sequences: int = 20000):
    base_url = f"https://gateway.ireceptor.org/samples?query_id={start_query_id}"
    current_page = 1
    repertoires_to_check = []
    seen_study_ids = set()
    too_small_dataset = False

    wait = WebDriverWait(driver, 40)

    while True:
        print(f"=== Loading page {current_page}... ===")
        driver.get(f"{base_url}&page={current_page}")

        WebDriverWait(driver, 40).until(EC.presence_of_element_located((By.CSS_SELECTOR, "td.col_ir_sequence_count a")))

        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        print(f"Found {len(rows)} rows on page {current_page}.")

        for i in range(len(rows)):
            try:
                rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
                row = rows[i]
                time.sleep(2)
                # wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "td.col_ir_sequence_count a")))
                # wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "td.col_repertoire_id")))
                # Wait for expected cells in this row only
                wait.until(lambda d: row.find_element(By.CSS_SELECTOR, "td.col_ir_sequence_count a"))
                wait.until(lambda d: row.find_element(By.CSS_SELECTOR, "td.col_repertoire_id"))
                wait.until(lambda d: row.find_element(By.CSS_SELECTOR, "td.col_study_id"))

                # Now safely get study_id from this row
                study_td = row.find_element(By.CLASS_NAME, "col_study_id")
                links = study_td.find_elements(By.TAG_NAME, "a")
                spans = study_td.find_elements(By.TAG_NAME, "span")

                # study_cell = row.find_element(By.CSS_SELECTOR, "td.col_study_id a")
                if links:
                    study_id = links[0].text.strip()
                elif spans:
                    study_id = spans[0].text.strip()
                else:
                    study_id = ""
                #study_id = study_cell.text.strip()
                if study_id in seen_study_ids:
                    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
                    continue

                seen_study_ids.add(study_id)

                seq_cell = row.find_element(By.CSS_SELECTOR, "td.col_ir_sequence_count a")
                rep_cell = row.find_element(By.CSS_SELECTOR, "td.col_repertoire_id")

                seq_count_text = seq_cell.text.strip().replace(",", "")
                if not seq_count_text.isdigit():
                    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
                    continue

                seq_count = int(seq_count_text)
                if seq_count <= min_sequences:
                    too_small_dataset = True
                    break  # break since table is sorted by sequence count

                repertoire_id = rep_cell.get_attribute("textContent").strip()

                print(f"--- Parsing row {i + 1} with new study id {study_id} and repertoire {repertoire_id} "
                      f"(page {current_page})... ---")

                try:
                    seq_cell.click()
                    wait.until(EC.text_to_be_present_in_element((By.TAG_NAME, "h1"), "2. Sequence Search"))

                    current_url = driver.current_url
                    print(f"Found URL: {current_url}")
                    query_id_match = re.search(r"query_id=([a-zA-Z0-9\-]+)", current_url)
                    if query_id_match:
                        query_id = query_id_match.group(1)
                        print(f"Found query_id: {query_id}")
                    else:
                        print("No query_id found in URL:", current_url)
                        driver = return_back_to_metadata(driver, wait)
                        #rows = driver.find_elements(By.CSS_SELECTOR, "tr.ng-scope")
                        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
                        continue

                    # Append collected information
                    repertoires_to_check.append((study_id, repertoire_id, query_id))
                    driver = return_back_to_metadata(driver, wait)
                    #rows = driver.find_elements(By.CSS_SELECTOR, "tr.ng-scope")

                except Exception as e:
                    print(f"Error clicking url or waiting for metadata to reload (row: {i+1}): {e}")
                    traceback.print_exc()

            except Exception as e:
                print(f"Error parsing metadata row {i+1}:", e)
                traceback.print_exc()

            #rows = driver.find_elements(By.CSS_SELECTOR, "tr.ng-scope")
            #rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

        if too_small_dataset:
            print(f"Datasets have become too small. Done with page {current_page}. Exiting.")
            break

        next_button = driver.find_elements(
            By.XPATH,
            "//ul[@class='pagination']//a[normalize-space(text())='%d']" % (current_page + 1)
        )

        if not next_button:
            print("No more pages. Exiting.")
            break

        print(f"Done with page {current_page}.")
        current_page += 1
        time.sleep(10)

    return repertoires_to_check


def check_umi_preview(chrome_driver, sample_query_id: str):
    query_url = f"https://gateway.ireceptor.org/sequences?query_id={sample_query_id}"
    chrome_driver.get(query_url)

    # Wait for the table to render by waiting for UMI count cells
    print("Waiting for table...")
    WebDriverWait(chrome_driver, 60).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "td.col_umi_count"))
    )
    print("Table loaded.")

    umi_cells = chrome_driver.find_elements(By.CSS_SELECTOR, "td.col_umi_count")

    umi_found = False
    print(len(umi_cells), "UMI count cells found.")
    for cell in umi_cells[0:1]:
        try:
            spans = cell.find_elements(By.TAG_NAME, "span")
            if not spans:
                print("No span found in UMI cell.")
                continue
            umi_count = spans[0].get_attribute("title").strip()
            if umi_count.isdigit():
                print(f"UMI count: {umi_count}")
                umi_found = True
        except Exception as e:
            print("Error reading cell:", e)

    return umi_found


def main():
    parser = argparse.ArgumentParser(description="Scrape UMI counts from iReceptor Gateway.")
    parser.add_argument("--username", type=str, required=True, help="Username for iReceptor Gateway")
    parser.add_argument("--password", type=str, required=True, help="Password for iReceptor Gateway")
    args = parser.parse_args()
    user_name = args.username
    user_password = args.password

    # Set up Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # comment this out to see the browser
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        print("Waiting for login page...")
        driver.get("https://gateway.ireceptor.org/login")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "username")))
        driver.find_element(By.ID, "username").send_keys(user_name)
        driver.find_element(By.ID, "password").send_keys(user_password)
        driver.find_element(By.CSS_SELECTOR, "input[type='submit']").click()
        print("Login submitted.")
        time.sleep(5)  # Wait for the login to process

        # Get query_ids
        print("Getting query_ids...")
        start_query_id = "120301"  # query_id for metadata with umi library method
        repertoires_to_check = get_repertoires_from_web(driver, start_query_id, min_sequences=20000)
        print(f"Found {len(repertoires_to_check)} sample_query_ids over threshold.")
        print(repertoires_to_check)

        # dump repertoires_to_check to file
        with open("explorative_analysis/results/repertoires_to_check.txt", "w") as f:
            for study_id, repertoire_id, sample_query_id in repertoires_to_check:
                f.write(f"{study_id}\t{repertoire_id}\t{sample_query_id}\n")

        # Load repertoires to check
        print("Loading repertoires to check...")
        with open("explorative_analysis/results/repertoires_to_check.txt", "r") as f:
            repertoires_to_check = [line.strip().split("\t") for line in f.readlines()]
        print(f"Loaded {len(repertoires_to_check)} repertoires to check.")
        print(repertoires_to_check)

        results = []
        checked_study_ids = []
        for study_id, repertoire_id, sample_query_id in repertoires_to_check:
            if study_id not in checked_study_ids:
                print(f"Checking repertoire {repertoire_id} of study {study_id}...")
                has_umi = check_umi_preview(driver, sample_query_id)
                results.append((repertoire_id, study_id, has_umi))
                checked_study_ids.append(study_id)
                print(f"Study {study_id} checked. Sleeping for 10 seconds...")
                time.sleep(10)
            else:
                print(f"Study {study_id} already checked, skipping.")

        # Save results
        results_df = pd.DataFrame(results, columns=['repertoire_id', 'study_id', 'has_umi'])
        results_df.to_csv("explorative_analysis/results/umi_presence_results.csv", index=False)

    except Exception as e:
        print("Error occurred:", type(e).__name__, e)

    driver.quit()
    print("Web Driver closed.")


if __name__ == "__main__":
    main()
