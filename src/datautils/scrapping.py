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
QUERY_GLOBAL = BASE_ARXIU + '/ms-opac/search?q=*%3A*&start={}&rows=10&sort=msstored_typology+asc&norm=*&fq=msstored_doctype&fv="Fotogràfic"&fo=and'
QUERY_EIXAMPLE = QUERY_GLOBAL # Workaround im lazy today

DRIVERPATH = 'utilities/geckodriver'
OUPATH = 'utilities/links.txt'
DRIVER = webdriver.Firefox()

#### Scrapping Eixample Data ####
def get_collections(rows = 100):
    plain_html = requests.get(QUERY_EIXAMPLE.format(rows)).text
    soup = BeautifulSoup(plain_html, features="html.parser")
    return soup.find_all(class_ = 'media-object') + soup.find_all(class_ = 'media') + soup.find_all(class_ = 'cont_imagen')

def get_images_from_collection_tag(tag):
    try: href = BASE_ARXIU + tag['href']
    except KeyError: return []

    # As the image carousel is loaded on the client side, we need a driver to process the JS
    DRIVER.get(href)
    try:
        gallery = DRIVER.find_element(By.CLASS_NAME, "es-carousel")
        images = gallery.find_elements(By.TAG_NAME, 'img')
        if not len(images): raise AssertionError
    except Exception as _: return []

    return [BASE_ARXIU + img.get_attribute('data-large') for img in images]

def save_links():
    col = get_collections()
    whole_data = []
    for colection in tqdm(col): whole_data.extend(get_images_from_collection_tag(colection))
    open(OUPATH, 'w').writelines('\n'.join(whole_data))
    DRIVER.close()

def save_links_huge(max_rows = 104797):
    whole_data = []
    for idx in tqdm(range(0, max_rows, 10)):
        col = get_collections(idx)
        for colection in col: whole_data.extend(get_images_from_collection_tag(colection))

    open(OUPATH, 'w').writelines('\n'.join(list(set(whole_data))))
    DRIVER.close()

if __name__ == '__main__': save_links_huge()