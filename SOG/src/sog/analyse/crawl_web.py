# import csv
# import time
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.chrome import ChromeDriverManager

# # === 配置 Chrome 启动参数 ===
# options = Options()
# options.add_argument('--no-sandbox')
# options.add_argument('--disable-dev-shm-usage')
# options.add_argument('--start-maximized')
# # options.add_argument('--headless')  # 如果不需要浏览器窗口可取消注释

# # 如果你知道实际 Chrome 路径可手动指定（可选）
# # options.binary_location = "/opt/google/chrome/google-chrome"

# # === 自动下载匹配版本的 ChromeDriver 并启动浏览器 ===
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# # === 打开网站 ===
# url = "https://honeybadger.uni.lu"
# driver.get(url)

# time.sleep(10)
# driver.save_screenshot("debug_screenshot.png")  # 看看页面是否真的加载成功

# # rows = driver.find_elements(By.CSS_SELECTOR, 'div.flex.flex-col > div')
# rows = driver.find_elements(By.CSS_SELECTOR, "table.table tbody tr")
# print(f"共找到 {len(rows)} 行")

# for row in rows:
#     cells = row.find_elements(By.TAG_NAME, "td")
#     print([cell.text for cell in cells])

# # === 等待数据区出现 ===
# WebDriverWait(driver, 200).until(
#     EC.presence_of_element_located((By.CSS_SELECTOR, "table.table tbody tr"))
# )

# all_data = []
# total_pages = 1  # 按需调整页数

# for page in range(1, total_pages + 1):
#     print(f"📄 正在处理第 {page} 页...")

#     time.sleep(20)

#     rows = driver.find_elements(By.CSS_SELECTOR, "table.table tbody tr")

#     for row in rows:
#         cells = row.find_elements(By.TAG_NAME, "td")
#         # lines = row.text.strip().split('\n')
        
#         block_height = cells[0].text
#         contract_address = cells[1].text
#         honeypot_creator = cells[2].text
#         honeypot_type = cells[3].text
#         all_data.append([
#             block_height, contract_address, honeypot_creator, honeypot_type
#         ])

#     # 点击下一页
#     try:
#         next_button = driver.find_element(By.XPATH, f"//div[contains(@class,'pagination-box')]//a[text()='{page + 1}']")
#         driver.execute_script("arguments[0].click();", next_button)
#     except Exception as e:
#         print(f"❌ 无法翻页到第 {page + 1} 页：{e}")
#         break

# # === 关闭浏览器 ===
# driver.quit()

# # === 保存数据为 CSV ===
# with open("SOG/dataset/honeybadger_contracts.csv", "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Block Height", "Contract Address", "Honeypot Creator", "Honeypot Type"])
#     writer.writerows(all_data)

# print("✅ 数据抓取完成，已保存为 honeybadger_contracts.csv")

import csv
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# === 配置 Chrome 启动参数 ===
options = Options()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--start-maximized')
# options.add_argument('--headless')  # 可选

# === 启动 ChromeDriver ===
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# === 打开网站 ===
url = "https://honeybadger.uni.lu/?page=81"
driver.get(url)

# 等待页面加载
WebDriverWait(driver, 20).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "table.table tbody tr"))
)

# === 准备 CSV 文件（写入标题） ===
output_path = "SOG/dataset/honeybadger_contracts3.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Block Height", "Contract Address", "Honeypot Creator", "Honeypot Type"])

# === 开始爬取分页数据 ===
page = 81
while True:
    print(f"📄 正在处理第 {page} 页...")

    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "table.table tbody tr"))
    )

    rows = driver.find_elements(By.CSS_SELECTOR, "table.table tbody tr")
    page_data = []

    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        if len(cells) >= 4:
            block_height = cells[0].text
            contract_address = cells[1].text
            honeypot_creator = cells[2].text
            honeypot_type = cells[3].text
            page_data.append([block_height, contract_address, honeypot_creator, honeypot_type])

    # 追加写入本页数据到 CSV
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(page_data)

    print(f"✅ 第 {page} 页保存 {len(page_data)} 条数据")

    # 点击下一页
    try:
        next_button = driver.find_element(By.XPATH, f"//div[contains(@class,'pagination-box')]//a[text()='{page + 1}']")
        driver.execute_script("arguments[0].click();", next_button)
        page += 1
        time.sleep(2)
    except Exception as e:
        print(f"❌ 无法翻页到第 {page + 1} 页：{e}")
        break

driver.quit()
print("🎉 数据抓取完成，已逐页保存至 CSV。")
