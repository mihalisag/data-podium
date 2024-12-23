import wikipediaapi
import pywikibot
from bs4 import BeautifulSoup

# Returns race names from Wikipedia (maybe no need?)
def race_names_gen(year: int):

    # Connect to the English Wikipedia site
    site = pywikibot.Site("en", "wikipedia")

    # Define the page title
    page_title = "2018 Formula One World Championship"
    page = pywikibot.Page(site, page_title)

    if page.exists():
        # Get the full HTML content of the page
        html_content = page.text

        # Parse the page's HTML using BeautifulSoup
        page_html = page.get_parsed_page()
        soup = BeautifulSoup(page_html, 'html.parser')

        # Find the table containing the 'Results and standings'
        tables = soup.find_all("table", {"class": "wikitable"})
        
        race_names = []
        for table in tables:
            # Search for "Report" links in the table
            links = table.find_all("a", string="Report")
            for link in links:
                href = str(link.get("href"))
                race_name = href.lstrip('/wiki/')
                # report_links.append(f"https://en.wikipedia.org{href}")
                race_names.append(race_name)

        # # Print the race names
        # print("Race names:")
        # for race in race_names:
        #     print(race)
    else:
        print(f"The page '{page_title}' does not exist on Wikipedia.")

    return race_names


# Function to retrieve and return the long text from either "Race" or "Report" sections
def get_race_report_text(race_name):

    # Initialize the Wikipedia API object
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')

    # Retrieve the page for the given race name
    page_py = wiki_wiki.page(race_name)
    
    # Function to retrieve a section's text by title
    def get_section_text(page, section_title):
        section = page.section_by_title(section_title)
        if section:
            return section
        else:
            return None

    # First, check for the "Report" section (older articles)
    report_section = get_section_text(page_py, 'Report')

    # If "Report" section exists, check for "Race" subsection inside it
    if report_section:
        race_section_in_report = report_section.section_by_title('Race')
        if race_section_in_report:
            return race_section_in_report.text  # Return the raw text of the "Race" subsection inside Report

    # If "Report" section doesn't exist, check for the "Race" section directly
    else:
        race_section = get_section_text(page_py, 'Race')
        
        # Try to retrieve the "Race report" subsection (if it exists)
        if race_section:
            race_report_section = race_section.section_by_title('Race report')
            if race_report_section:
                return race_report_section.text  # Return the raw text of the "Race report" subsection
            else:
                return race_section.text

    # If no relevant sections are found, return a message indicating so
    return "No relevant race report found for this article."


# Helper function to extract a short paragraph (first few sentences or characters)
def extract_short_paragraph(text, max_length=1500):
    # Limit the length of the text to the max_length (e.g., first 500 characters)
    if len(text) > max_length:
        text = text[:max_length]  # Cut the text to the first max_length characters

    # # Optionally, break at the last full sentence (a period, question mark, or exclamation mark).
    # if '.' in text:
    #     text = text.split('.')[0] + '.'  # Cut off after the first sentence

    return text.strip()
