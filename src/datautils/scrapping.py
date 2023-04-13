import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

# El rows = 1000 travieso és per aprofitar el bug
BASE_ARXIU = 'https://catalegarxiumunicipal.bcn.cat'
# QUERY_EIXAMPLE = BASE_ARXIU + '/ms-opac/mosaic?q=dreta+de+l%27eixample&start=0&rows=1000&sort=fecha%20asc&fq=norm&fv=*&fo=and&fq=norm&fv=*&fo=and&fq=msstored_doctype&fv=%22Fotogr%C3%A0fic%22&fo=and&fq=media&fv=*&fo=and'
QUERY_GLOBAL = BASE_ARXIU + f"/ms-opac/search?q=*%3A*&start=0&rows={10 * 104797}&sort=msstored_typology+asc&norm=*&fq=msstored_doctype&fv=%22Fotogr%C3%A0fic%22&fo=and"
QUERY_EIXAMPLE = QUERY_GLOBAL # Workaround im lazy today

DRIVERPATH = 'utilities/geckodriver'
OUPATH = 'utilities/links.txt'
DRIVER = webdriver.Firefox()

#### Scrapping Eixample Data ####
def get_collections():
    plain_html = requests.get(QUERY_EIXAMPLE).text
    soup = BeautifulSoup(plain_html, features="html.parser")
    return soup.find_all(class_ = 'media-object')

def get_images_from_collection_tag(tag):
    href = BASE_ARXIU + tag['href']

    # As the image carousel is loaded on the client side, we need a driver to process the JS
    DRIVER.get(href)
    try:
        gallery = DRIVER.find_element(By.CLASS_NAME, "es-carousel")
        images = gallery.find_elements(By.TAG_NAME, 'img')
        if not len(images): raise AssertionError
    except: return []

    return [BASE_ARXIU + img.get_attribute('data-large') for img in images]

def save_links():
    col = get_collections()
    whole_data = []
    for colection in tqdm(col): whole_data.extend(get_images_from_collection_tag(colection))
    open(OUPATH, 'w').writelines('\n'.join(whole_data))
    DRIVER.close()


if __name__ == '__main__': save_links()