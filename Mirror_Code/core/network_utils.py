# core/network_utils.py

import logging
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

logger = logging.getLogger(__name__)


def get_page_load_time(url, headless=True):
    """Return load time in milliseconds for a URL using browser timing APIs."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=chrome_options)
    try:
        start_time = time.time()
        driver.get(url)
        load_time_script = """
        var performance = window.performance || window.webkitPerformance || window.mozPerformance || window.msPerformance;
        if (performance) {
            var timing = performance.timing;
            return timing.loadEventEnd - timing.navigationStart;
        }
        return 0;
        """
        load_time_ms = driver.execute_script(load_time_script)
        measured_time_ms = (time.time() - start_time) * 1000
        logger.info(
            "Page load for %s -> browser_timing=%sms measured_python=%.2fms",
            url,
            load_time_ms,
            measured_time_ms,
        )
        return load_time_ms
    finally:
        driver.quit()
