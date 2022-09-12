# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:44:30 2022

@author: HP
"""
import csv
import time
from selenium import webdriver
import pandas as pd

from selenium.common.exceptions import ElementClickInterceptedException


driver = webdriver.Chrome()
driver.implicitly_wait(6)
driver.get("https://healthunlocked.com/vasculitis-uk/posts#mobile")
# click accept cookies
driver.find_element_by_id("ccc-notify-accept").click()
post_links = set()
while True:
    driver.get("https://healthunlocked.com/vasculitis-uk/posts#mobile")
    all_posts = [post for post in
                 driver.find_element_by_class_name("results-posts").find_elements_by_class_name("results-post") if
                 "results-post" == post.get_attribute("class")]
    # handle clicking more posts
    while len(all_posts) <= len(post_links):

        see_more_posts = [btn for btn in driver.find_elements_by_class_name("btn-secondary")
                          if btn.text == "See more posts"]
        try:
            see_more_posts[0].click()
        except ElementClickInterceptedException:
            # handle floating box covering "see more posts" button
            driver.execute_script("return document.getElementsByClassName('floating-box-sign-up')[0].remove();")
            see_more_posts[0].click()
        all_posts = [post for post in driver.find_element_by_class_name("results-posts").find_elements_by_class_name("results-post") if "results-post" == post.get_attribute("class")]
    # popoulate links
    start_from = len(post_links)
    for post in all_posts[start_from:]: # len(post_links): <-- to avoid visiting same links
        # save link
        link = post.find_element_by_tag_name("a").get_attribute("href")
        post_links.add(link)

    # visit the site and scrape info
    for post_site in list(post_links)[start_from:]:
        
        

        driver.get(post_site)
        post_text = driver.find_element_by_class_name("post-body").text
        for btn in driver.find_element_by_class_name("post-actions__buttons").find_elements_by_tag_name("button"):
            if "Like" in btn.text:
                with open('realvascu10.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)

    # write the header
                    writer.writerow([post_text])

    # write the data
                   #writer.writerow(data)
                    
                    post_like = btn.text.split()[1][1]
                