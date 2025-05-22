import requests
from bs4 import BeautifulSoup


def scrape_all_links(url):
    """
    Scrapes all links from a given URL and returns a list of unique, absolute URLs.
    """
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()  # Use a set to store unique links
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Handle relative URLs
            absolute_url = requests.compat.urljoin(url, href)  # Correct way to join URLs

            # Filter out URLs that don't start with the base domain
            if absolute_url.startswith("https://www.torchtechnologies.com") or absolute_url.startswith("https://torchtechnologies.com"):
                links.add(absolute_url)

        return list(links)  # Convert set back to list
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return []


def scrape_text_from_url(url):
    """
    Scrapes text from a single URL.
    """
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return "\n".join(p.get_text(strip=True) for p in soup.find_all('p'))
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return ""
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return ""



# Main execution
base_url = "https://www.torchtechnologies.com/"
all_links = scrape_all_links(base_url)

# Remove duplicate links.  This can happen in complex sites.
unique_links = list(set(all_links))

# Scrape and combine text from all URLs
all_text = ""
for url in unique_links:
    print(f"Scraping: {url}")
    page_text = scrape_text_from_url(url)
    all_text += f"\n\n--- Page: {url} ---\n\n{page_text}"


# Save to a single text file
with open("torch_website_data.txt", "w", encoding="utf-8") as f:
    f.write(all_text)