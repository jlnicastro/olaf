import requests
from bs4 import BeautifulSoup
import time


def scrape_all_links(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']

            # Skip anchors
            if '#' in href:
                continue

            # Skip news links
            if 'news' in href:
                continue

            absolute_url = requests.compat.urljoin(url, href)
            links.add(absolute_url)
        return list(links)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []


def scrape_text_from_url(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        main_tag = soup.find('main')
        if not main_tag:
            print(f"No element with <main> tag found on {url}")
            return ""

        # Get all text, clean whitespace
        text = main_tag.get_text(separator="\n", strip=True)

        return text

    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return ""

base_url = "https://torchtechnologies.com/"
all_links = set()

page_links = scrape_all_links(base_url)
print(f"Found {len(page_links)} links on page {base_url}")

all_links.update(page_links)
print(f"✅ Total unique article links found: {len(all_links)}")

# Now scrape each article
all_text = ""

for url in all_links:
    print(f"Scraping: {url}")
    page_text = scrape_text_from_url(url)
    all_text += f"\n\n--- Page: {url} ---\n\n{page_text}"
    time.sleep(1)

with open("torch_website_data.txt", "w", encoding="utf-8") as f:
    f.write(all_text)