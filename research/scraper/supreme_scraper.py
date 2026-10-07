from playwright.sync_api import sync_playwright
import os
import time
import random

def scrape_court_cases():
    TARGET_COUNT = 1200
    
    court_levels = {
        # "First_Instance": "https://shuukh.mn/cases/1/1",
        # "Appellate": "https://shuukh.mn/cases/1/2",
        "Supreme": "https://shuukh.mn/cases/1/3"
    }

    DATE_RANGE = "2018-06-01 - 2026-07-04"
    
    # Easily add more dispute types from the dropdown here to expand your dataset
    DISPUTE_TYPES = [
        "Бусад зээл",
        # "Түрээсийн гэрээ",
        # "Худалдах-худалдан авах болон арилжааны гэрээ",
        # "Ажил гүйцэтгэх"
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False) 
        context = browser.new_context()
        
        for dispute_type in DISPUTE_TYPES:
            for level_name, url in court_levels.items():
                print(f"\n{'='*60}\nTarget: {dispute_type} | {level_name} Court\n{'='*60}")
                
                # Nesting folders by Dispute Type -> Court Level for clean dataset organization
                clean_dispute_name = dispute_type.replace('/', '_').replace(' ', '_')
                save_dir = f"case_details/{clean_dispute_name}/{level_name}"
                os.makedirs(save_dir, exist_ok=True)
                
                existing_files = [f for f in os.listdir(save_dir) if f.endswith(".txt")]
                collected_count = len(existing_files)
                print(f"Current progress: {collected_count}/{TARGET_COUNT} cases.")
                
                if collected_count >= TARGET_COUNT:
                    print("Target reached for this section. Skipping.")
                    continue

                list_tab = context.new_page()
                max_retries = 3
                success = False
                for attempt in range(max_retries):
                    try:
                        list_tab.goto(url, timeout=60000)
                        success = True
                        break
                    except Exception as e:
                        print(f"Failed to load {url} (Attempt {attempt+1}/{max_retries}): {e}")
                        time.sleep(10)
                
                if not success:
                    print(f"Website is unstable or returning 500 errors. Skipping {level_name}.")
                    list_tab.close()
                    continue
                

                # ---------------------------------------------------------
                # APPLY FILTERS (Handling the Complex JS Dropdown)
                # ---------------------------------------------------------
                print("Applying Date and Dispute Type filters...")
                try:
                    # 1. Fill the date range
                    date_input = list_tab.locator("xpath=//div[contains(text(), 'Огноо')]/parent::*//input").first
                    if date_input.is_visible():
                        date_input.fill("")
                        date_input.fill(DATE_RANGE)
                        date_input.press("Enter")
                    
                    # 2. Fill the dispute type (Supreme Court specific logic)
                    # The Supreme Court page uses a standard HTML <select> element, 
                    # unlike the lower courts which use a complex JS dropdown.
                    select_element = list_tab.locator("select#hh")
                    if select_element.is_visible():
                        # We use label=dispute_type to match the visible text (e.g., "Бусад зээл")
                        select_element.select_option(label=dispute_type)
                    else:
                        print(f"Warning: Could not find the select dropdown (#hh) for {dispute_type}")
                        
                    list_tab.wait_for_timeout(500)
                    
                    # 3. Click Search
                    search_btn = list_tab.locator("button:has-text('Хайх')").first
                    if search_btn.is_visible():
                        search_btn.click()
                        
                    print("Filters applied! Waiting for results to load...")
                    list_tab.wait_for_timeout(5000)
                    
                except Exception as e:
                    print(f"Warning: Could not auto-fill filters. Error: {e}")
                    print("Please fill them manually in the browser right now. The script will wait 15 seconds...")
                    list_tab.wait_for_timeout(15000)

                # ---------------------------------------------------------
                # SCRAPING LOOP
                # ---------------------------------------------------------
                while collected_count < TARGET_COUNT:
                    try:
                        list_tab.wait_for_selector('a[href*="/single_case/"]', timeout=30000)
                    except:
                        print("No more cases found. Moving to next target.")
                        break
                        
                    elements = list_tab.locator('a[href*="/single_case/"]').all()
                    
                    valid_cases = []
                    for el in elements:
                        text = el.inner_text().strip()
                        href = el.get_attribute('href')
                        if '/' in text and href:
                            valid_cases.append((text, href))
                            
                    print(f"Found {len(valid_cases)} cases on this page.")
                    
                    for raw_text, href in valid_cases:
                        if collected_count >= TARGET_COUNT:
                            break
                        
                        if '\n' in raw_text:
                            raw_text = raw_text.split('\n')[0].strip()
                            
                        clean_name = raw_text.replace('/', '_').replace(' ', '_')
                        filename = f"{save_dir}/{clean_name}.txt"
                        
                        if os.path.exists(filename):
                            continue
                            
                        case_url = f"https://shuukh.mn{href}"
                        print(f"[{level_name} | {collected_count + 1}/{TARGET_COUNT}] Downloading: {clean_name}")
                        
                        try:
                            download_page = context.new_page()
                            download_page.goto(case_url, timeout=60000)
                            
                            download_page.wait_for_selector("body", timeout=30000)
                            content = download_page.locator("body").inner_text()
                            
                            with open(filename, "w", encoding="utf-8") as f:
                                f.write(content)
                                
                            download_page.close()
                            collected_count += 1
                            time.sleep(random.uniform(0.5, 1.5)) 
                        except Exception as e:
                            print(f"Error on {clean_name}: {e}")
                            try:
                                download_page.close()
                            except:
                                pass
                            continue

                    if collected_count >= TARGET_COUNT:
                        break

                    print("Clicking 'Дараах' (Next) button...")
                    try:
                        next_button = list_tab.get_by_text("Дараах")
                        next_button.wait_for(state="visible", timeout=5000)
                        next_button.click()
                        list_tab.wait_for_timeout(3000) 
                    except Exception as e:
                        print("No more pages found. Finishing this section.")
                        break
                
                list_tab.close()
            
        browser.close()
        print("\nSuccess! Data extraction complete.")

if __name__ == "__main__":
    scrape_court_cases()