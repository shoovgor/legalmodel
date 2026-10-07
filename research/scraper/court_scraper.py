from playwright.sync_api import sync_playwright
import os
import time
import random
import re

def scrape_court_cases():
    # Шатны шүүх тус бүрт тусдаа зорилтот тоо тогтоох
    court_levels = {
        "First_Instance": {"url": "https://shuukh.mn/cases/1/1", "target": 10000},
        "Appellate":      {"url": "https://shuukh.mn/cases/1/2", "target": 7000},
        "Supreme":        {"url": "https://shuukh.mn/cases/1/3", "target": 1000},
    }

    DATE_START = "2022/01/01"
    DATE_END   = "2027/07/01"
    
    # Easily add more dispute types from the dropdown here to expand your dataset
    DISPUTE_TYPES = [
        "Бусад зээл",
        # "Түрээсийн гэрээ",
        # "Худалдах-худалдан авах болон арилжааны гэрээ",
        # "Ажил гүйцэтгэх"
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True) 
        context = browser.new_context()
        
        for dispute_type in DISPUTE_TYPES:
            for level_name, level_cfg in court_levels.items():
                url = level_cfg["url"]
                TARGET_COUNT = level_cfg["target"]
                print(f"\n{'='*60}\nTarget: {dispute_type} | {level_name} Court ({TARGET_COUNT} зорилтот)\n{'='*60}")
                
                # Nesting folders by Dispute Type -> Court Level for clean dataset organization
                clean_dispute_name = dispute_type.replace('/', '_').replace(' ', '_')
                save_dir = f"case_details/{clean_dispute_name}/{level_name}"
                os.makedirs(save_dir, exist_ok=True)
                
                existing_files = [f for f in os.listdir(save_dir) if f.endswith(".txt")]
                collected_count = len(existing_files)
                print(f"Одоогийн явц: {collected_count}/{TARGET_COUNT}")
                
                if collected_count >= TARGET_COUNT:
                    print("Зорилтот тоонд хүрсэн. Дараагийнх руу шилжиж байна.")
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
                # APPLY FILTERS (Handling the Select2 Dropdown)
                # ---------------------------------------------------------
                print("Applying Date and Dispute Type filters...")
                try:
                    # Wait for Select2 to initialize on the page
                    list_tab.wait_for_selector(".select2-container", timeout=15000)
                    list_tab.wait_for_timeout(2000)  # Extra time for all JS to finish
                    
                    # 1. Set the date range via the daterangepicker jQuery plugin API.
                    #    .fill() only changes the visible text but does NOT update the
                    #    picker's internal startDate/endDate, so the server-side filter
                    #    never actually receives the custom range.  Using the plugin's
                    #    own setStartDate/setEndDate methods fixes that.
                    list_tab.evaluate(f'''() => {{
                        const $dp = $('#end_date');
                        $dp.data('daterangepicker').setStartDate('{DATE_START}');
                        $dp.data('daterangepicker').setEndDate('{DATE_END}');
                    }}''')
                    print(f"Set date range: {DATE_START} - {DATE_END}")
                    list_tab.wait_for_timeout(500)
                    
                    # 2. Set the dispute type via Select2 on <select id="hh">
                    # Look up the <option> value for the given dispute_type text
                    option_value = list_tab.evaluate(f'''() => {{
                        const sel = document.querySelector('#hh');
                        for (const opt of sel.options) {{
                            if (opt.textContent.trim() === "{dispute_type}") return opt.value;
                        }}
                        return null;
                    }}''')
                    
                    if option_value:
                        # Use Select2's jQuery API to set the value programmatically
                        list_tab.evaluate(f'''() => {{
                            const $el = $('#hh');
                            $el.val(['{option_value}']);
                            $el.trigger('change');
                        }}''')
                        print(f"Set dispute type '{dispute_type}' (value={option_value})")
                    else:
                        # Fallback: interact with the Select2 UI manually
                        print(f"Option value not found for '{dispute_type}', trying UI interaction...")
                        select2_container = list_tab.locator("#hh").locator("xpath=..").locator(".select2-container").first
                        select2_container.click()
                        list_tab.wait_for_timeout(1000)
                        
                        # Type into the Select2 search box
                        search_input = list_tab.locator(".select2-search__field").first
                        if search_input.is_visible():
                            search_input.fill(dispute_type)
                            list_tab.wait_for_timeout(1500)
                        
                        # Click the matching result
                        dropdown_item = list_tab.locator(f".select2-results__option:has-text('{dispute_type}')").first
                        if dropdown_item.is_visible():
                            dropdown_item.click()
                        else:
                            list_tab.keyboard.press("Enter")
                        
                        list_tab.keyboard.press("Escape")
                    
                    list_tab.wait_for_timeout(500)
                    
                    # 3. Click the Search / Хайх button (text varies between MN/EN)
                    search_btn = list_tab.locator("button[type='submit']").first
                    if not search_btn.is_visible():
                        search_btn = list_tab.locator("button:has-text('Хайх'), button:has-text('Search')").first

                    # Capture what's showing BEFORE search, so we can tell
                    # whether clicking actually changed anything - confirmed
                    # the site sometimes needs the button clicked more than
                    # once before results actually refresh.
                    try:
                        before_href = list_tab.locator('#contents a[href*="/single_case/"]').first.get_attribute("href", timeout=3000)
                    except Exception:
                        before_href = None

                    search_worked = False
                    for search_attempt in range(4):
                        search_btn.click()
                        list_tab.wait_for_timeout(3000)
                        try:
                            after_href = list_tab.locator('#contents a[href*="/single_case/"]').first.get_attribute("href", timeout=3000)
                        except Exception:
                            after_href = None
                        if after_href and after_href != before_href:
                            search_worked = True
                            break
                        print(f"Search click {search_attempt + 1}/4 didn't change results yet, retrying...")

                    if not search_worked:
                        print("Warning: results never appeared to change after multiple search clicks - "
                              "continuing anyway, but double check the filter actually applied "
                              "(e.g. check the daterange param on a downloaded case's URL).")

                    print("Filters applied! Waiting for results to load...")
                    list_tab.wait_for_timeout(2000)
                    
                except Exception as e:
                    print(f"Warning: Could not auto-fill filters. Error: {e}")
                    print("Please fill them manually in the browser right now. The script will wait 30 seconds...")
                    list_tab.wait_for_timeout(30000)

                # ---------------------------------------------------------
                # SCRAPING LOOP
                # ---------------------------------------------------------
                consecutive_empty_pages = 0
                MAX_EMPTY_RETRIES = 3
                
                while collected_count < TARGET_COUNT:
                    # Wait for case links to appear in the table
                    try:
                        list_tab.wait_for_selector('#contents a[href*="/single_case/"]', timeout=30000)
                    except:
                        consecutive_empty_pages += 1
                        if consecutive_empty_pages >= MAX_EMPTY_RETRIES:
                            print(f"No cases found after {MAX_EMPTY_RETRIES} attempts. Moving to next target.")
                            break
                        # Exponential backoff (10s, 30s, 90s...) rather than a
                        # flat 5s wait - repeated empty pages right after an
                        # ERR_HTTP_RESPONSE_CODE_FAILURE looks like the site
                        # pushing back under sustained load, not a one-off
                        # blip. A longer cooldown gives it room to recover
                        # instead of giving up on the whole section too fast.
                        backoff = 10 * (3 ** (consecutive_empty_pages - 1))
                        print(f"No cases found on this page (attempt {consecutive_empty_pages}/{MAX_EMPTY_RETRIES}). Waiting {backoff}s before retrying...")
                        list_tab.wait_for_timeout(backoff * 1000)
                        try:
                            # Confirmed via traceback: this exact call threw
                            # an unguarded playwright.TimeoutError ("Page.
                            # reload: Timeout 30000ms exceeded") and crashed
                            # the entire script. Longer timeout gives a slow
                            # site more room; the try/except means a second
                            # failure here costs a retry, not the whole run.
                            list_tab.reload(timeout=60000)
                        except Exception as reload_err:
                            print(f"Reload also failed/timed out: {reload_err}")
                        list_tab.wait_for_timeout(5000)
                        continue
                    
                    # Reset empty page counter on success
                    consecutive_empty_pages = 0
                        
                    elements = list_tab.locator('#contents a[href*="/single_case/"]').all()
                    
                    # Group all links by href to deduplicate — each table row has
                    # multiple <a> tags (court, date, case#, judge) all pointing to
                    # the same /single_case/ URL.  We collect every link text per href
                    # so we can pick the proper case-number text for the filename.

                    href_map = {}  # href -> list of link texts
                    for el in elements:
                        text = el.inner_text().strip()
                        href = el.get_attribute('href')
                        if href:
                            href_map.setdefault(href, []).append(text)
                    
                    valid_cases = []
                    case_num_pattern = re.compile(r'^\d+[/_][А-Яа-яЁёA-Za-z]{2,}\d{4}[/_]\d+$|^\d+[/_]\d{4}[А-Яа-яЁёA-Za-z]{2,}[/_]\d+$')
                    for href, texts in href_map.items():
                        # Try to find the case-number text among all link texts for this href
                        case_name = None
                        for t in texts:
                            clean = t.replace(' ', '_').replace('/', '_')
                            if case_num_pattern.match(clean):
                                case_name = clean
                                break
                        # Fallback: use the case ID from the URL itself
                        if not case_name:
                            url_match = re.search(r'/single_case/(\d+)', href)
                            case_name = f"case_{url_match.group(1)}" if url_match else texts[0].replace('/', '_').replace(' ', '_')
                        valid_cases.append((case_name, href))
                            
                    print(f"Found {len(valid_cases)} cases on this page.")
                    
                    if len(valid_cases) == 0:
                        print("Page loaded but no valid cases extracted. Moving to next target.")
                        break
                    
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
                            # small backoff instead of immediately firing the
                            # next request - ERR_HTTP_RESPONSE_CODE_FAILURE
                            # looks like the site pushing back under load, so
                            # hammering it right away likely makes it worse
                            time.sleep(random.uniform(3, 6))
                            continue

                    if collected_count >= TARGET_COUNT:
                        break

                    # ---------------------------------------------------------
                    # PAGINATION: Click the "Next" button
                    # ---------------------------------------------------------
                    print("Clicking 'Дараах' (Next) button...")
                    try:
                        # Use the pagination container to avoid matching unrelated "Дараах" text
                        next_button = list_tab.locator('#pagination_link a:has-text("Дараах")').first
                        next_button.wait_for(state="visible", timeout=10000)
                        
                        # Snapshot current page content to detect when it changes
                        old_first_link = list_tab.locator('#contents a[href*="/single_case/"]').first.get_attribute('href')
                        
                        next_button.click()
                        
                        # Wait for AJAX: the table content should refresh (new links appear)
                        for wait_attempt in range(10):
                            list_tab.wait_for_timeout(2000)
                            try:
                                new_first_link = list_tab.locator('#contents a[href*="/single_case/"]').first.get_attribute('href')
                                if new_first_link != old_first_link:
                                    break  # New page loaded successfully
                            except:
                                pass  # Links not yet rendered
                        else:
                            print("Warning: Page content may not have refreshed. Continuing anyway...")
                        
                        list_tab.wait_for_timeout(1000)
                        
                    except Exception as e:
                        print(f"No more pages found (pagination ended). Finishing this section. ({e})")
                        break
                
                list_tab.close()
            
        browser.close()
        print("\nSuccess! Data extraction complete.")

if __name__ == "__main__":
    scrape_court_cases()