import json
import os
import re
import time
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

BASE_URL = "http://digimons.net/digimon/"
INDEX_URL = urljoin(BASE_URL, "chn.html")
DATA_DIR = "./train"
METADATA_PATH = os.path.join(DATA_DIR, "metadata.jsonl")


def ensure_data_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def fetch_soup(session: requests.Session, url: str) -> Optional[BeautifulSoup]:
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")
    except requests.RequestException as exc:
        print(f"请求失败: {url} ({exc})")
        return None


def normalize_href(value: object) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
        return value[0]
    return None


def extract_name_and_detail_url(digimon: Tag) -> tuple[Optional[str], Optional[str]]:
    link = digimon.find("a")
    if not isinstance(link, Tag):
        return None, None

    href = normalize_href(link.get("href"))
    if not href:
        return None, None

    name = href.strip("/").split("/")[0]
    if not name:
        return None, None

    detail_url = urljoin(BASE_URL, href)
    return name, detail_url


def collect_digimon_entries(index_soup: BeautifulSoup) -> list[tuple[str, str]]:
    # 新版页面不再提供统一的 digimon-list 容器，改为从 href 规则提取条目。
    href_pattern = re.compile(r"^[a-z0-9][a-z0-9_-]*/index\.html$", re.IGNORECASE)
    entries: list[tuple[str, str]] = []
    seen_names: set[str] = set()

    for link in index_soup.select("a[href]"):
        if not isinstance(link, Tag):
            continue

        href = normalize_href(link.get("href"))
        if not href:
            continue

        cleaned_href = href.strip()
        if not href_pattern.match(cleaned_href):
            continue

        name = cleaned_href.split("/")[0].strip()
        if not name or name in seen_names:
            continue

        seen_names.add(name)
        detail_url = urljoin(INDEX_URL, cleaned_href)
        entries.append((name, detail_url))

    return entries


def main() -> None:
    ensure_data_dir(DATA_DIR)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; dataset-crawler/1.0)",
        }
    )

    soup = fetch_soup(session, INDEX_URL)
    if soup is None:
        raise SystemExit("入口页获取失败，程序结束。")

    digimon_entries = collect_digimon_entries(soup)
    if not digimon_entries:
        raise SystemExit("未提取到数码兽详情链接，网页结构可能已变化。")

    next_index = (
        sum(1 for filename in os.listdir(DATA_DIR) if filename.lower().endswith(".jpg")) + 1
    )

    with open(METADATA_PATH, "a", encoding="utf-8") as meta_file:
        for name, detail_url in digimon_entries:
            print(f"正在处理：{detail_url}")
            detail_soup = fetch_soup(session, detail_url)
            if detail_soup is None:
                continue

            profile_eng = detail_soup.find("div", class_="profile_eng")
            if not isinstance(profile_eng, Tag):
                print(f"跳过 {name}: 未找到英文简介块")
                continue

            caption_p = profile_eng.find("p")
            if not isinstance(caption_p, Tag):
                print(f"跳过 {name}: 未找到简介段落")
                continue

            caption = caption_p.get_text(strip=True)
            if not caption:
                print(f"跳过 {name}: 简介为空")
                continue

            img_url = urljoin(BASE_URL, f"{name}/{name}.jpg")
            try:
                img_response = session.get(img_url, timeout=15)
                img_response.raise_for_status()
                content_type = img_response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    print(f"跳过 {name}: 响应非图片 ({content_type})")
                    continue
            except requests.RequestException as exc:
                print(f"跳过 {name}: 图片下载失败 ({exc})")
                continue

            file_name = f"{next_index:04d}.jpg"
            with open(os.path.join(DATA_DIR, file_name), "wb") as img_file:
                img_file.write(img_response.content)

            metadata = {"file_name": file_name, "text": f"{name}. {caption}"}
            meta_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            next_index += 1
            time.sleep(0.2)


if __name__ == "__main__":
    main()