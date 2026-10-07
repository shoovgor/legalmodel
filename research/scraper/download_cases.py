from playwright.sync_api import sync_playwright
import os

def download_all_cases():
    # 1. Prepare directory
    os.makedirs("case_details", exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # 2. Read your list of links
        with open("cases_found.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            if "| URL:" in line:
                # Extract URL and Case ID
                case_name = line.split("|")[0].replace("Case: ", "").strip()
                url = line.split("| URL: ")[1].strip()
                
                print(f"Downloading: {case_name}...")
                
                try:
                    page.goto(url, wait_until="domcontentloaded")
                    # Change this selector to match the part of the page you want to save
                    content = page.locator("body").inner_text() 
                    
                    # 3. Save as individual file
                    filename = f"case_details/{case_name.replace('/', '_')}.txt"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(content)
                        
                except Exception as e:
                    print(f"Failed to download {case_name}: {e}")

        browser.close()
        print("All downloads complete!")

if __name__ == "__main__":
    download_all_cases()