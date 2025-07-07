import requests
from bs4 import BeautifulSoup
import time


def scrape_all_links(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        for card in soup.find_all(class_='card'):
            for a_tag in card.find_all('a', href=True):
                href = a_tag['href']

                absolute_url = requests.compat.urljoin('https://torchhelps.org', href)
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

        article_flex_content = soup.find(class_='article-flex_content')
        if not article_flex_content:
            print(f"No element with class 'article-flex_content' found on {url}")
            return ""

        # Get all text, clean whitespace
        text = article_flex_content.get_text(separator="\n", strip=True)

        return text

    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return ""


base_url = "https://torchhelps.org/news/"
all_links = set()

# Loop through all pages
for page in range(1, 3):
    if page == 1:
        page_url = base_url
    else:
        page_url = f"{base_url}?page={page}"
    print(f"Scraping links from: {page_url}")
    page_links = scrape_all_links(page_url)
    print(f"Found {len(page_links)} links on page {page}")
    all_links.update(page_links)
    time.sleep(1)

print(f"✅ Total unique article links found: {len(all_links)}")

# Now scrape each article
all_text = ""

for url in all_links:
    print(f"Scraping: {url}")
    page_text = scrape_text_from_url(url)
    all_text += f"\n\n--- Page: {url} ---\n\n{page_text}"
    time.sleep(1)

with open("torchhelps_news_data.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("✅ Done. All articles saved to torchhelps_news_data.txt")
