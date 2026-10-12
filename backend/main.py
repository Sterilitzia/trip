from typing import Optional
import os
import re
import json
import math
import queue
import threading
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import urllib.request
import urllib.parse
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pypinyin import lazy_pinyin
from attractions import detect_attraction, wants_ticket, booking_answer, booking_info
from brands import detect_brands, brand_for_query, brand_prompt, is_brand_store, unverified_brands, STORE_CATEGORIES
from menu import (MENU_OCR_MODELS_DEFAULT, MENU_TRANSLATE_MODEL_DEFAULT, TRANSLATE_BATCH, TRANSLATE_FIRST_BATCH, TRANSLATE_WORKERS, MenuReadError,
                  ocr_menu_items, translate_menu_items, format_price_en, price_options, decode_image)

# 加载 .env 环境变量
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化大模型客户端（智谱兼容 OpenAI SDK）
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY") or "local-time-only",
    base_url=os.getenv("LLM_BASE_URL")
)

# ============================================================
# 高德地图 REST API：真实公交/地铁换乘路线
# 线路和站点使用地图接口结果，模型仅负责解析地名。
# ============================================================
AMAP_KEY = os.getenv("AMAP_KEY")
AMAP_BASE = "https://restapi.amap.com/v3"


def _amap_get(path: str, params: dict) -> Optional[dict]:
    """统一的高德 REST GET 请求：带 key、超时保护，失败时返回 None（调用方负责降级到纯跳转链接）。"""
    if not AMAP_KEY:
        return None
    query = {**params, "key": AMAP_KEY}
    url = f"{AMAP_BASE}/{path}?{urllib.parse.urlencode(query)}"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[高德API错误] {path}: {e}")
        return None
    if data.get("status") != "1":
        print(f"[高德API失败] {path}: {data.get('info')}")
        return None
    return data


def amap_reverse_geocode(lon: float, lat: float) -> Optional[str]:
    """经纬度 → 城市名。用来给目的地搜索加“当前城市”偏好，否则“南京路”这种地名在全国范围搜索时
    可能匹配到完全不相关的城市（实测：不加城市偏好会匹配到新疆石河子市的同名路）。
    北京/上海/天津/重庆等直辖市的 city 字段是空数组，用 province 兜底。"""
    data = _amap_get("geocode/regeo", {"location": f"{lon},{lat}"})
    if not data:
        return None
    comp = (data.get("regeocode") or {}).get("addressComponent") or {}
    city = comp.get("city")
    if not city or isinstance(city, list):
        city = comp.get("province") or ""
    return city or None


_PAREN_NOTE = re.compile(r"[(（].*?[)）]")


def _core_name(name: str) -> str:
    """去掉括号备注和空白，用于名称比对：复旦大学(地铁站) → 复旦大学。"""
    return re.sub(r"\s+", "", _PAREN_NOTE.sub("", name or ""))


def place_matches(query: str, name: str) -> bool:
    """搜索结果是不是用户要的那个地方：名称互相包含才算（天安门 ↔ 天安门广场 ✓，天安门 ↔ 天安中心 ✗）。
    高德在限定城市里搜不到时不会返回空，而是返回字面相近的别的地方（实测：人在上海搜“北京理工大学良乡校区”
    → 上海理工大学），所以不能盲信第一条结果。英文/拼音关键词没法和中文名比对，不做判断。"""
    q, n = _core_name(query), _core_name(name)
    if not q or not n or not _HAN.search(q):
        return True
    return q in n or n in q


def amap_search_place(keywords: str, city: Optional[str] = None) -> Optional[dict]:
    """地名/地标关键词 → 经纬度。优先用高德“兴趣点搜索”（按真实相关度/人气排序），
    而不是结构化地址编码 geocode/geo——后者是给完整门牌地址用的，对“南京路”这种裸地标名排序很差
    （实测排到一条无关小路前面）。带 city 时限定在该城市搜索，找不到再退回地址编码兜底。
    返回的 matched 表示名称是否和关键词对得上（见 place_matches），由调用方决定要不要换个城市再搜。"""
    params = {"keywords": keywords, "offset": 5, "page": 1}
    if city:
        params = {**params, "city": city, "citylimit": "true"}
    data = _amap_get("place/text", params)
    places = []
    for p in (data or {}).get("pois") or []:
        location = p.get("location", "")
        if "," in location:
            lon, lat = location.split(",")
            places.append({"lon": float(lon), "lat": float(lat), "city": p.get("cityname") or city or "", "formatted_address": p.get("name") or keywords})
    if places:
        # 前几条里第一个名称对得上的（排序靠前但名字对不上的多半是字面相近的别处）
        best = next((pl for pl in places if place_matches(keywords, pl["formatted_address"])), places[0])
        return {**best, "matched": place_matches(keywords, best["formatted_address"])}

    # 兴趣点搜索没结果（比如用户输入的是完整门牌地址）→ 退回结构化地址编码
    geo_params = {"address": keywords}
    if city:
        geo_params["city"] = city
    data = _amap_get("geocode/geo", geo_params)
    geocodes = (data or {}).get("geocodes") or []
    if not geocodes:
        return None
    g = geocodes[0]
    location = g.get("location", "")
    if "," not in location:
        return None
    lon, lat = location.split(",")
    # 门牌地址的 formatted_address 是“省市区 + 原地址”，正常情况下包含关键词，同样可以比对
    name = g.get("formatted_address") or keywords
    return {"lon": float(lon), "lat": float(lat), "city": g.get("city") or g.get("province") or city or "", "formatted_address": name, "matched": place_matches(keywords, name)}


def find_place(keywords: str, preferred_cities: list[Optional[str]]) -> Optional[dict]:
    """按城市优先级依次搜索，最后全国搜索，返回第一个名称对得上的结果。
    限定城市是为了“中山公园/人民广场”这类全国有很多个的地名落在用户所在的城市；
    但用户说的是别的城市的地方（人在上海，说“我在北京理工大学”）时，限定城市只会搜出字面相近的错地方，
    这时要继续往后搜。都对不上时退回最优先城市的结果（如“上海火车站”→“上海站”这类简称）。"""
    fallback = None
    tried = set()
    for city in [*preferred_cities, None]:
        if city in tried:
            continue
        tried.add(city)
        place = amap_search_place(keywords, city)
        if not place:
            continue
        if place["matched"]:
            return place
        fallback = fallback or place
    return fallback


def amap_gps_to_gcj02(lat: float, lng: float) -> Optional[dict]:
    """浏览器 Geolocation 给的是 GPS(WGS84) 坐标，高德用 GCJ02——不转换起点会偏移几百米。"""
    data = _amap_get("assistant/coordinate/convert", {"locations": f"{lng},{lat}", "coordsys": "gps"})
    if not data:
        return None
    location = data.get("locations", "")
    if "," not in location:
        return None
    lon, lat2 = location.split(",")
    return {"lon": float(lon), "lat": float(lat2)}


def amap_transit_directions(origin: dict, destination: dict, city: str) -> Optional[dict]:
    """公交/地铁换乘方案：解析成前端可直接渲染的时间线步骤（哪一站上车/换乘/下车）。
    高德默认按 transits[0] 排的“推荐”方案不是按耗时排序的——实测同一对起终点里，
    排第一的公交方案要 42 分钟，而列表里第 4 条地铁方案只要 29 分钟。这里改成从所有候选方案里
    直接选耗时最短的一条，不盲信第一条。"""
    data = _amap_get("direction/transit/integrated", {
        "origin": f"{origin['lon']},{origin['lat']}",
        "destination": f"{destination['lon']},{destination['lat']}",
        "city": city,
        "extensions": "all",
    })
    if not data:
        return None
    transits = (data.get("route") or {}).get("transits") or []
    if not transits:
        return None
    best = min(transits, key=lambda tr: float(tr.get("duration") or float("inf")))

    steps = []
    for seg in best.get("segments", []):
        walking = seg.get("walking")
        if walking and float(walking.get("distance") or 0) >= 50:
            steps.append({
                "type": "walk",
                "distance_m": round(float(walking.get("distance") or 0)),
                "duration_min": max(1, round(float(walking.get("duration") or 0) / 60)),
            })
        buslines = ((seg.get("bus") or {}).get("buslines")) or []
        if buslines:
            bl = buslines[0]
            line_name = bl.get("name", "")
            steps.append({
                "type": "subway" if ("地铁" in line_name or "地铁" in bl.get("type", "")) else "bus",
                "line": line_name.split("(")[0].strip(),
                "from": (bl.get("departure_stop") or {}).get("name", ""),
                "to": (bl.get("arrival_stop") or {}).get("name", ""),
                "via_num": int(bl.get("via_num") or 0),
                "duration_min": max(1, round(float(bl.get("duration") or 0) / 60)),
            })

    if not steps:
        return None

    cost = best.get("cost")
    if isinstance(cost, dict):
        cost = cost.get("transit_fee")

    total_distance = best.get("distance") or (data.get("route") or {}).get("distance") or 0

    return {
        "duration_min": max(1, round(float(best.get("duration") or 0) / 60)),
        "distance_km": round(float(total_distance) / 1000, 1),
        "cost_cny": cost,
        "steps": steps,
        "destination_lon": destination["lon"],
        "destination_lat": destination["lat"],
    }


def amap_walking_directions(origin: dict, destination: dict) -> Optional[dict]:
    """步行路线。近距离（附近的商场/便利店）时公交换乘接口直接返回 0 个方案，只能走路。"""
    data = _amap_get("direction/walking", {
        "origin": f"{origin['lon']},{origin['lat']}",
        "destination": f"{destination['lon']},{destination['lat']}",
    })
    paths = ((data or {}).get("route") or {}).get("paths") or []
    if not paths:
        return None
    distance = float(paths[0].get("distance") or 0)
    duration_min = max(1, round(float(paths[0].get("duration") or 0) / 60))
    return {
        "duration_min": duration_min,
        "distance_km": round(distance / 1000, 1),
        "cost_cny": None,
        "steps": [{"type": "walk", "distance_m": round(distance), "duration_min": duration_min}],
        "destination_lon": destination["lon"],
        "destination_lat": destination["lat"],
    }


def straight_distance_m(a: dict, b: dict) -> float:
    """两点直线距离（米），用来判断是不是步行距离。"""
    lat1, lat2 = math.radians(a["lat"]), math.radians(b["lat"])
    dlat, dlon = lat2 - lat1, math.radians(b["lon"] - a["lon"])
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


# 直线距离在这以内直接给步行路线：否则公交接口会给出“走 300 米、坐 1 站、再走 300 米”这种比走路还慢的方案
WALKING_THRESHOLD_M = 1200


def amap_route(origin: dict, destination: dict, city: str) -> Optional[dict]:
    """近距离步行；远距离公交/地铁换乘，没有换乘方案时退回步行。
    直线近但实际要绕路的（天安门周边封闭区域：直线 1.1 公里、步行 2.3 公里）再和公交比一下，取快的。"""
    if straight_distance_m(origin, destination) <= WALKING_THRESHOLD_M:
        walk = amap_walking_directions(origin, destination)
        if walk and walk["distance_km"] * 1000 <= WALKING_THRESHOLD_M * 1.5:
            return walk
        transit = amap_transit_directions(origin, destination, city)
        if transit and (not walk or transit["duration_min"] < walk["duration_min"]):
            return transit
        return walk
    return amap_transit_directions(origin, destination, city) or amap_walking_directions(origin, destination)


# 附近搜索时排除的高德 POI 大类：用户问“最近的商场/药店”时，名字里带“商业广场”的写字楼、公司、小区都不是他要的
_NEARBY_EXCLUDED_CATEGORIES = ("商务住宅", "公司企业", "地名地址信息")


def amap_search_nearby(keywords: str, center: dict, limit: int = 3) -> list[dict]:
    """以 center 为圆心按距离由近到远搜某类地点（商场/药店/便利店…）或某个连锁品牌，5 公里内。
    搜品牌时只保留门店名里真的带这个品牌名的店（见 is_brand_store），宁可显示“附近没有”，也不带去别家。"""
    brand = brand_for_query(keywords)
    data = _amap_get("place/around", {
        "location": f"{center['lon']},{center['lat']}",
        "keywords": keywords,
        "radius": 5000,
        "sortrule": "distance",
        "offset": 20,
        "page": 1,
    })
    places = []
    for p in (data or {}).get("pois") or []:
        location = p.get("location", "")
        if "," not in location or (p.get("type") or "").startswith(_NEARBY_EXCLUDED_CATEGORIES):
            continue
        if brand and not is_brand_store(brand, p.get("name") or ""):
            continue
        lon, lat = location.split(",")
        address = p.get("address")
        places.append({
            "name": p.get("name") or keywords,
            "address": address if isinstance(address, str) else "",  # 高德没有地址时给的是空数组
            "distance_m": round(float(p.get("distance") or 0)),
            "lon": float(lon),
            "lat": float(lat),
        })
        if len(places) >= limit:
            break
    return places


# ============================================================
# 站名/地名英译：英文界面下站名保留中文（和站牌、问路时对得上），括号里附英文；线路名直接给英文。
# 路线本身仍然只来自高德——这里只翻译高德返回的名字，中文原名始终保留作为依据。
# ============================================================
TRANSLATE_PROMPT = """
你是中国地名英译助手。把用户给的 JSON 数组里每个中文名字翻译成外国游客看得懂的英文，严格输出 JSON（不要 Markdown 标记）：
{"translations": {"中文原名": "English name"}}

规则：
1. 地铁站、景点、商圈有官方英文名的，用地铁站牌/官方使用的英文名，如：南京东路 → East Nanjing Road，人民广场 → People's Square，豫园 → Yuyuan Garden，陆家嘴 → Lujiazui。
2. 没有官方英文名的：专名用拼音，通名意译（路 Road，街 Street，大道 Avenue，广场 Square，公园 Park，大学 University，校区 Campus，地铁站 Metro Station，公交站 Bus Stop）。如：国权路 → Guoquan Road。
3. 两条路名拼成的公交站名（路口站）用 & 连接：国权路邯郸路 → Guoquan Road & Handan Road。
4. 括号里的补充说明也要翻译：复旦大学(地铁站) → Fudan University (Metro Station)。
5. 线路名：地铁N号线 → Metro Line N；N路 → Bus N；快线/区间/夜宵线等按含义意译，如 300路快外 → Bus 300 Express (Outer Loop)。
6. 每个输入都必须出现在 translations 里，键与输入完全一致；只输出英文，不要附中文。
"""

_METRO_LINE = re.compile(r"^地铁(\d+)号线")
_BUS_LINE = re.compile(r"^(\d+)路")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_NAME_EN_CACHE: dict[str, str] = {}  # 站名译名不会变，跨请求缓存，同一条路线第二次查询不再调模型


def line_name_en(name: str) -> Optional[str]:
    """最常见的线路名格式（地铁N号线 / N路）用代码确定性翻译，不经过模型。"""
    for pattern, template in ((_METRO_LINE, "Metro Line {}"), (_BUS_LINE, "Bus {}")):
        m = pattern.match(name)
        if m and m.end() == len(name):
            return template.format(m.group(1))
    return None


def line_name_fallback_en(name: str) -> str:
    """模型翻译失败时的线路名兜底：保留线路号，丢掉“快外/区间”这类后缀（站名仍中英对照，不影响认路）。"""
    for pattern, template in ((_METRO_LINE, "Metro Line {}"), (_BUS_LINE, "Bus {}")):
        m = pattern.match(name)
        if m:
            return template.format(m.group(1))
    if name.endswith("线") and len(name) > 1:  # 浦江线 → Pujiang Line，地铁房山线 → Fangshan Line
        return f"{place_name_pinyin_en(name[:-1].removeprefix('地铁'))} Line"
    return place_name_pinyin_en(name)


# ---- 模型不可用（限流/超时）时的确定性兜底：拼音 + 通名意译，遵循上海路牌/站牌的英文规范 ----
# 顺序即优先级：长的通名在前，避免“步行街”被当成“街”
_GENERIC_NAMES = [
    ("购物中心", "Shopping Mall"), ("便利店", "Convenience Store"), ("大药房", "Pharmacy"),
    ("药房", "Pharmacy"), ("药店", "Pharmacy"), ("超市", "Supermarket"), ("商场", "Mall"), ("商城", "Mall"),
    ("博物院", "Museum"), ("医院", "Hospital"), ("酒店", "Hotel"), ("饭店", "Restaurant"), ("餐厅", "Restaurant"), ("商店", "Store"),
    ("步行街", "Pedestrian Street"), ("地铁站", "Metro Station"), ("公交站", "Bus Stop"),
    ("火车站", "Railway Station"), ("博物馆", "Museum"), ("大学", "University"), ("校区", "Campus"),
    ("广场", "Square"), ("公园", "Park"), ("大道", "Avenue"), ("机场", "Airport"),
    ("路", "Road"), ("街", "Street"), ("寺", "Temple"), ("桥", "Bridge"), ("站", "Station"),
]
# 只在多字通名和“路”处切分（路口站名“国权路邯郸路”要切成两条路）；单字通名（桥/站/街/寺）常出现在专名中间，
# 只在结尾生效——否则“虹桥火车站”会被切成 Hong Bridge Railway Station
_GENERIC_SPLIT = re.compile("(.*?(?:" + "|".join(zh for zh, _ in _GENERIC_NAMES if len(zh) > 1 or zh == "路") + "))")
_ROAD_DIRECTIONS = {"东": "East", "西": "West", "南": "South", "北": "North", "中": "Middle"}
_ROAD_ORDINALS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_PARENS = re.compile(r"^(.+?)[(（](.+?)[)）]$")


def _pinyin_word(zh: str) -> str:
    """专名转拼音连写，首字母大写；以 a/e/o 开头的非首音节加隔音符号（静安 → Jing'an）。"""
    out = ""
    for syl in lazy_pinyin(zh):
        if out and out[-1].isalpha() and syl[:1] in ("a", "e", "o"):
            out += "'"
        out += syl
    return out[:1].upper() + out[1:]


def _road_en(stem: str) -> str:
    """路名（不含“路”字）→ 英文，按上海路牌规范：南京东路 → East Nanjing Road，中山东二路 → East Zhongshan No.2 Road。
    方位字/序数字去掉后专名至少要剩两个字，否则它们是专名的一部分（山东路 → Shandong Road，纬一路 → Weiyi Road）。"""
    if not stem:
        return "Road"
    number = ""
    if len(stem) >= 3 and stem[-1] in _ROAD_ORDINALS:
        number, stem = f"No.{_ROAD_ORDINALS[stem[-1]]} ", stem[:-1]
    if len(stem) >= 3 and stem[-1] in _ROAD_DIRECTIONS:
        return f"{_ROAD_DIRECTIONS[stem[-1]]} {_pinyin_word(stem[:-1])} {number}Road"
    return f"{_pinyin_word(stem)} {number}Road"


def _token_en(token: str) -> tuple[str, bool]:
    """一个“专名+通名”片段 → 英文。返回 (译名, 是否是一条路)。"""
    for zh, en in _GENERIC_NAMES:
        if token.endswith(zh):
            stem = token[: -len(zh)]
            if zh == "路":
                return _road_en(stem), True
            return (f"{_pinyin_word(stem)} {en}" if stem else en), False
    # 方位后缀的站名：天安门西 → Tian'anmen West，广州东 → Guangzhou East（“山西/广东”这类两字专名不拆）
    if len(token) >= 3 and token[-1] in _ROAD_DIRECTIONS and token[-1] != "中":
        return f"{_pinyin_word(token[:-1])} {_ROAD_DIRECTIONS[token[-1]]}", False
    return _pinyin_word(token), False


def place_name_pinyin_en(name: str) -> str:
    """站名/地名的确定性英译兜底：复旦大学邯郸校区 → Fudan University Handan Campus，
    国权路邯郸路（路口公交站）→ Guoquan Road & Handan Road，复旦大学(地铁站) → Fudan University (Metro Station)。"""
    name = name.strip()
    m = _PARENS.match(name)
    if m:
        return f"{place_name_pinyin_en(m.group(1))} ({place_name_pinyin_en(m.group(2))})"
    # 连锁店分店：益丰大药房国定路店 → Yifeng Pharmacy Guoding Road Branch，全家便利店(政肃路店) → … (Zhengsu Road Branch)
    branch = name.endswith("店") and not any(name.endswith(zh) for zh, _ in _GENERIC_NAMES)
    if branch and len(name) > 1:
        return f"{place_name_pinyin_en(name[:-1])} Branch"
    parts = _name_parts_en(name)
    if len(parts) > 1 and all(is_road for _, is_road in parts):
        return " & ".join(en for en, _ in parts)
    return " ".join(en for en, _ in parts)


def _name_parts_en(name: str) -> list[tuple[str, bool]]:
    return [_token_en(t) for t in _GENERIC_SPLIT.split(name) if t]


def is_road_name(name: str) -> bool:
    """纯路名/路口名（国权路、南京东路、中山东二路金陵东路）。这类名字的英文完全由路牌规范决定，
    规则译名就是官方译名；模型反而会译得不统一（实测：中山东二路 → Zhongshan East 2nd Road）。"""
    parts = _name_parts_en(name.strip())
    return bool(parts) and all(is_road for _, is_road in parts)


def translate_names(names: list[str]) -> dict[str, str]:
    """批量把站名/地名/线路名译成英文（一次模型调用），只返回模型给出的译名；
    模型失败的名字由调用方用确定性兜底补上。"""
    todo = sorted({n for n in names if n and _HAN.search(n) and n not in _NAME_EN_CACHE})
    if todo:
        try:
            # 只在后台异步调用（路线早已用兜底译名显示），超时可以放宽；限流时不重试，免得加重限流
            response = client.with_options(timeout=20, max_retries=0).chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=[
                    {"role": "system", "content": TRANSLATE_PROMPT},
                    {"role": "user", "content": json.dumps(todo, ensure_ascii=False)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            translations = json.loads(response.choices[0].message.content).get("translations") or {}
            for zh, en in translations.items():
                # 只收下确实是请求过的名字、且确实是英文的译名，防止模型夹带中文或多编条目
                if zh in todo and isinstance(en, str) and re.search(r"[A-Za-z]", en) and not _HAN.search(en):
                    _NAME_EN_CACHE[zh] = en.strip()
        except Exception as e:
            print(f"[翻译失败] {e}")
    return {n: _NAME_EN_CACHE[n] for n in names if n in _NAME_EN_CACHE}


def attach_english_names(result: dict) -> dict:
    """给路线结果补上英文名：站名/地名 → *_en 字段（前端显示为“中文 (English)”），线路名 → line_en。
    不调模型，不拖慢路线返回：优先用模型以前翻译过的缓存，否则用确定性兜底。
    英文界面会再异步调 /api/translate_names 拿模型的官方译名替换兜底译名。"""
    def place_en(zh: Optional[str]) -> Optional[str]:
        if not zh:
            return None
        # 只能网上预约的景点（故宫博物院…）有核对过的英文名，直接用，不走拼音兜底（“Gugongbowuyuan”）
        attraction = detect_attraction(zh)
        if attraction and zh in (attraction["zh"], *attraction.get("zh_names", [])):
            return attraction["en"]
        return _NAME_EN_CACHE.get(zh) or place_name_pinyin_en(zh)

    for key in ("origin_name", "destination_name"):
        result[f"{key}_en"] = place_en(result.get(key))
    for s in result.get("steps") or []:
        if s.get("line"):
            s["line_en"] = line_name_en(s["line"]) or _NAME_EN_CACHE.get(s["line"]) or line_name_fallback_en(s["line"])
            s["from_en"] = place_en(s.get("from"))
            s["to_en"] = place_en(s.get("to"))
    return result

# ============================================================
# 核心 Prompt：所有场景共享（输出格式 + 通用规则）
# 保持 CORE_PROMPT 内容稳定不变，有利于 LLM 服务商的上下文缓存命中
# ============================================================
CORE_PROMPT = """
你是旅游消费场景的意图澄清与行动助手：弄清外国游客的需求，生成可直接出示给中国店员的双语执行卡，或在用户问路时帮他规划出行方式。

【输出】严格 JSON（不要 Markdown 标记）：
{
  "status": "success" 或 "clarify",
  "intent_type": "drink" 或 "food" 或 "ticket" 或 "retail" 或 "recommend" 或 "travel" 或 "direction" 或 "other",
  "question": "clarify 时的提问；success 时可写一句推荐理由或默认值说明，没有则空字符串",
  "quick_replies": ["快捷选项1", "快捷选项2", "快捷选项3"],
  "card_content": {"zh": "中文执行卡", "en": "English translation"},
  "action": "show_to_staff" 或 "search_map" 或 "trip_com" 或 "navigate",
  "map_query": "仅当 action 是 search_map 或 navigate 时填地图搜索关键词（如：便利店 / 故宫），否则填空字符串",
  "origin_query": "仅当 action 是 navigate 或 search_map，且用户明确说了自己在哪/从哪出发时填该地点名称（如：复旦大学），否则填空字符串",
  "city": "仅当 action 是 navigate 或 search_map 时填所在城市的中文名（如：北京），判断不了填空字符串"
}

【通用规则】：
1. 默认行动，追问是例外：只要能给出一个合理、用户可纠正的方案，直接 success。只有缺少决定性信息（完全无法给出合理方案）才 clarify，且同一段对话最多连续追问两次（每次都必须比上一次更接近可执行方案），绝不连环追问。
2. 用户在任何一轮说过的偏好必须继承，绝不重复追问；绝不替用户编造他没说的决定性信息（如具体场次）。
3. clarify 时：提一个针对性问题 + 3-4 个 quick_replies（最后一项必须是“其他”）。success 卡片里绝不提问；success 时 quick_replies 为空数组。
4. card_content：除“问路/导航”场景外都是出示给店员的，必须第一人称（我/I），禁止“您/请问/是否”反问句。如：“你好，我要一杯伯牙绝弦，少糖，谢谢。”“问路/导航”场景的 card_content 规则见该场景专属说明。
5. action 取值：点单/点餐/购票/推荐 → "show_to_staff"；找店购物/找最近的某类地方（最近的商场/药店/便利店） → "search_map" 且必须填 map_query（地点类别）；机票/酒店 → "trip_com"；问路/导航（去某地怎么走） → "navigate" 且必须填 map_query。clarify 状态下 action 和 map_query 都填空字符串。不要输出 next_action 字段，系统会自动生成。
6. 在品类已确认的前提下，用户用模糊描述委托你选具体项时（如“fruity”“随便”“都可以”“推荐一个”），不要继续反问细节——直接替他选最受欢迎的具体单品，success，question 写推荐理由（用户可纠正）。
"""

# ============================================================
# 场景定义：关键词（路由用）+ 槽位 schema + 场景专属 Prompt
# 按场景添加规则，避免不同业务的约束互相干扰。
# ============================================================
SCENARIOS = {
    "direction": {
        "keywords": [
            "怎么走", "怎么去", "怎么坐车", "怎么坐地铁", "地铁怎么换乘", "地铁换乘", "换乘",
            "路线怎么", "导航去", "导航到", "打车去", "坐地铁去", "坐地铁到", "公交到", "怎么到",
            "从这里到", "路怎么走", "想去", "我要去", "带我去",
            "how do i get to", "how to get to", "how can i get to", "way to get to",
            "route to", "directions to", "navigate to", "how to reach", "how do i reach",
            "take the subway to", "take the metro to", "by metro to", "by subway to",
            "want to go to", "on my way to", "take me to", "heading to",
        ],
        "prompt": """
【当前场景：路线导航】
用户想知道怎么去/怎么走到某个具体地点（地铁换乘、打车等），这不是找店购物，也不是点单——他已经知道要去哪，只是不知道怎么去。
直接 success（无需追问路线细节，具体换乘方案由系统调用地图 API 实时生成）：
- 地名里的校区/分店/分馆/出入口等限定词必须保留，绝不能省略——同一所大学不同校区可能相隔几十公里。如 "Beijing Institute of Technology (Liangxiang Campus)" → "北京理工大学良乡校区"，"Fudan University Jiangwan Campus" → "复旦大学江湾校区"。
- action="navigate"，map_query 只填目的地这一个地名的标准中文名称（地图能搜到的名字，英文地名要翻译成中文，如 "Forbidden City" → "故宫"，"Nanjing Road" → "南京路"）。map_query 绝不能包含出发地，绝不能写成“A到B”“A to B”这种整句——它会被直接拿去地图搜索，写成整句会搜到出发地。
- origin_query：用户明确说了自己在哪/从哪出发（如 "I'm at Fudan University"、“我在复旦大学”）→ 填出发地的标准中文名称（如 "复旦大学"）；没说 → 空字符串（系统会用手机定位）。
- city：用户明确说了城市（如 "Zhongshan Park in Beijing"）→ 填该城市中文名（如 "北京"）；没说城市但提到的地点是全国唯一、众所周知属于某城市的（如 天安门、北京理工大学 → "北京"；外滩、复旦大学 → "上海"）→ 填该城市；地名在很多城市都有（如 中山公园、火车站、南京路）且用户没说城市 → 空字符串（系统会用手机定位所在城市）。服务范围是全中国，绝不要默认成上海。
- 你不知道实时的地铁线路和换乘信息，绝不能自己编路线：question 必须是空字符串，card_content 里也绝不能出现任何具体线路号、站名或耗时。
- card_content 在此场景不是出示给店员的，而是给用户看的行程摘要：zh/en 各一句话，说明可以坐地铁（含换乘）或打车前往，例如 {"zh": "为你规划前往故宫的路线：可乘地铁换乘前往，也可以直接打车。", "en": "Here's how to get to the Forbidden City: take the metro (with a line change) or just grab a car."}
- 目的地是“一类地方”而不是某个具体地点（如“最近的商场”“附近的药店”“找个便利店”）→ 这是找最近的店，不是去某个地方：action="search_map"，map_query 填地点类别（如 "商场"），origin_query/city 照常填，card_content 两个字段都填空字符串。
- 只有当用户完全没说清要去哪（如只说“怎么走”但没提地点，或句子说到一半没说完，如“我在复旦大学，想去……”后面没有下文）才 clarify，quick_replies 给 2-3 个热门地点示例 + “其他”。
- 绝对不能把用户说的出发地/当前位置当成目的地，也绝对不能在真实目的地缺失时凭空编一个目的地——用户没说清就 clarify，宁可多问一句，也不能编错方向。
示例：用户：“How do I get to the Forbidden City?” → {"status":"success","intent_type":"direction","question":"","quick_replies":[],"card_content":{"zh":"为你规划前往故宫的路线：可乘地铁换乘前往，也可以直接打车。","en":"Here's how to get to the Forbidden City: take the metro (with a line change) or just grab a car."},"action":"navigate","map_query":"故宫","origin_query":"","city":"北京"}
示例：用户：“I'm at Fudan University and want to go to Nanjing Road, give me directions” → {"status":"success","intent_type":"direction","question":"","quick_replies":[],"card_content":{"zh":"为你规划从复旦大学前往南京路的路线：可乘地铁换乘前往，也可以直接打车。","en":"Here's how to get from Fudan University to Nanjing Road: take the metro (with line changes) or just grab a car."},"action":"navigate","map_query":"南京路","origin_query":"复旦大学","city":"上海"}
示例：用户：“I'm at Fudan University and I want to go to”（句子没说完，没提目的地） → {"status":"clarify","intent_type":"direction","question":"Where would you like to go from Fudan University?","quick_replies":["City Center","Shopping Mall","Restaurant","其他"],"card_content":{"zh":"","en":""},"action":"","map_query":"","origin_query":"","city":""}
""",
    },
    "travel": {
        "keywords": [
            "机票", "航班", "酒店", "订房", "住宿", "机场", "高铁", "火车票",
            "flight", "hotel", "hostel", "airbnb", "airport", "train ticket", "book a room",
        ],
        "prompt": """
【当前场景：机票/酒店】
直接 success：引导用户前往 Trip.com 查看和预订，action="trip_com"，card_content 写引导语（第一人称，含用户已提供的日期/目的地信息）。不追问缺失的日期——用户会在 Trip.com 上自行选择。
""",
    },
    "retail": {
        "keywords": [
            "哪里买", "哪有卖", "购买", "超市", "便利店", "药店", "药房", "商场", "商城", "购物中心", "免税店", "附近", "最近的",
            "where can i buy", "where to buy", "buy", "purchase", "store", "supermarket", "pharmacy", "convenience store", "shop",
            "mall", "shopping center", "shopping centre", "nearest", "nearby", "near me", "closest",
        ],
        "prompt": """
【当前场景：找店购物】
用户想知道“哪里能买到X”或“最近的某类店/地方在哪”→ 绝不追问细节，直接 success：action="search_map"，map_query 填店铺/地点类别（如：便利店、超市、药店、商场），系统会搜出离用户最近的几家并规划路线。找店 ≠ 进店下单。
- card_content：用户要买具体东西 → 写用户想买的东西（到店后可出示给店员，第一人称）；用户只是找地方、没说要买什么（如“最近的商场在哪”）→ zh/en 都填空字符串，绝不写“我想找最近的商场”这种没法给店员看的话。
- origin_query：用户说了自己在哪（如 "I'm at Fudan University (Jiangwan Campus)"）→ 填该地点标准中文名，校区/分店等限定词必须保留（如 "复旦大学江湾校区"）；没说 → 空字符串（系统用手机定位）。
- city：用户说了城市，或提到的地点众所周知属于某城市（如 复旦大学 → "上海"）→ 填城市中文名；否则空字符串。绝不要默认成上海。
示例：用户：“I need to buy candy, where can I buy?” → {"status":"success","intent_type":"retail","question":"","quick_replies":[],"card_content":{"zh":"你好，我想买糖果，谢谢。","en":"Hello, I'd like to buy some candy."},"action":"search_map","map_query":"便利店","origin_query":"","city":""}
示例：用户：“I'm at Fudan University (Jiangwan Campus), where's the nearest mall?” → {"status":"success","intent_type":"retail","question":"","quick_replies":[],"card_content":{"zh":"","en":""},"action":"search_map","map_query":"商场","origin_query":"复旦大学江湾校区","city":"上海"}
""",
    },
    "ticket": {
        "keywords": [
            "门票", "票", "学生票", "景点", "博物馆", "故宫", "护照", "身份证", "学生",
            "ticket", "student", "passport", "museum", "attraction", "admission", "student id",
        ],
        "prompt": """
【当前场景：票务/景点】
槽位：票种（学生/成人/儿童）；证件（护照/身份证）；场次/时间段（上午/下午/晚上 —— 决定性！）。
完整性：场次/时间段是出票的必要信息，用户未提供 → 必须 clarify（quick_replies：上午/下午/晚上/其他），绝不假设当天任意时间。票种、证件、场次都已知 → success。
""",
    },
    "recommend": {
        "keywords": [
            "霸王茶姬", "chagee", "蜜雪", "mixue", "星巴克", "starbucks", "瑞幸", "luckin",
            "喜茶", "heytea", "奈雪", "川菜", "sichuan", "火锅", "hotpot", "粤菜", "湘菜",
            "北京烤鸭", "peking duck", "推荐", "招牌", "特色", "recommend", "signature",
        ],
        "prompt": """
【当前场景：招牌推荐】
用户提到店家/品牌/菜系并想要建议：利用你的真实知识直接推荐一个具体单品/菜品，success，intent_type:"recommend"，question 写一句推荐理由，card_content 为第一人称点单语句。连锁品牌推招牌单品；路边摊/菜馆推品类经典菜。用户已说的偏好（如不太甜）必须体现在卡片参数里。
示例：用户：“我在霸王茶姬，想要一杯不太甜的奶茶” → {"status":"success","intent_type":"recommend","question":"推荐伯牙绝弦：茉莉茶香浓、甜度天然偏低，少糖更清爽。","quick_replies":[],"card_content":{"zh":"你好，我要一杯伯牙绝弦，少糖，谢谢。","en":"Hello, one Boya Juexian (jasmine green milk tea), low sugar, please."},"action":"show_to_staff","map_query":""}
示例：用户：“I want some spicy Sichuan food” → {"status":"success","intent_type":"recommend","question":"回锅肉是川菜之魂，一定要试试。","quick_replies":[],"card_content":{"zh":"你好，我要一份回锅肉，谢谢。","en":"Hello, one Twice-Cooked Pork (回锅肉), please."},"action":"show_to_staff","map_query":""}
""",
    },
    "drink": {
        "keywords": [
            "奶茶", "咖啡", "果茶", "果汁", "可乐", "气泡水", "柠檬茶", "绿茶", "红茶", "乌龙茶", "花茶",
            "milk tea", "coffee", "tea", "juice", "bubble tea", "boba", "latte", "smoothie", "drink", "beverage", "coke",
        ],
        "prompt": """
【当前场景：饮品点单】
槽位：品类（茶/果茶/奶茶/咖啡等，必需）；糖度（无糖/30%糖/50%糖/70%糖，可选）；冰量（去冰/少冰/常温/多冰，可选）。
完整性：确认品类即完整 → success。糖度/冰量：用户说过→继承映射（“不太甜”→少糖/30%糖）；没说过→用热门默认（少糖、少冰），question 注明“已为您选择…”。
示例：用户：“我要一杯果茶，不太甜” → {"status":"success","intent_type":"drink","question":"已为您选择少糖、少冰，可告诉我调整。","quick_replies":[],"card_content":{"zh":"你好，我要一杯水果茶，少糖，少冰，谢谢。","en":"Hello, one fruit tea, low sugar, less ice, please."},"action":"show_to_staff","map_query":""}
""",
    },
    "food": {
        "keywords": [
            "汉堡", "披萨", "面条", "包子", "饺子", "烤串", "沙拉", "三明治", "吃",
            "burger", "cheeseburger", "hamburger", "pizza", "noodle", "rice", "dumpling", "sandwich", "salad", "food", "eat", "hungry",
        ],
        "prompt": """
【当前场景：点餐】
槽位：菜品（必需）；加料/辣度/份量（可选，高价值选项最多问一次）。
完整性：确认菜品即完整 → success。
菜名必须标准，绝不张冠李戴：回锅肉(Twice-Cooked Pork)、水煮牛肉(Boiled Beef in Chili Oil)、麻婆豆腐(Mapo Tofu)、宫保鸡丁(Kung Pao Chicken)、鱼香肉丝(Yu-Shiang Shredded Pork)、小笼包(Xiaolongbao)、北京烤鸭(Peking Duck)。拿不准 → clarify。
示例：用户：“I want a cheeseburger.”（已回答不加料） → {"status":"success","intent_type":"food","question":"","quick_replies":[],"card_content":{"zh":"你好，我要一个芝士汉堡，谢谢。","en":"Hello, one cheeseburger, please."},"action":"show_to_staff","map_query":""}
""",
    },
}

GENERAL_PROMPT = """
【当前场景：通用/未分类】
先判断用户需求属于哪类：问路导航/饮品/点餐/票务/找店购物/机酒/求推荐。
- 用户已经知道要去哪，只是问怎么走/怎么坐车/怎么换乘 → 问路导航场景，action="navigate"，map_query 只填目的地的中文名（绝不写成“A到B”整句），用户说了出发地就填 origin_query，question 为空字符串，绝不自己编线路/站名。但目的地必须是用户明确说出来的；句子说到一半没说完、或只提到了出发地/当前位置（如“我在复旦大学，想去……”后面没有下文）→ 目的地缺失，必须 clarify 问清楚要去哪，绝不能把出发地当成目的地，也绝不能凭空编一个目的地。
- 品类本身未定时（如只说“fruity / 水果味的”，全程未提“喝/饮品/茶”）→ clarify 品类（quick_replies：饮品/食物/其他）。
- 一旦品类确认，按该品类常识处理：饮品确认品类即可出卡（糖冰用热门默认并注明）；找店直接 search_map；票务缺场次必须 clarify；机酒引导 Trip.com；提到品牌/菜系求建议→直接推荐具体单品。
- 拿不准具体菜名/单品时 → clarify，绝不张冠李戴。常见标准菜名：回锅肉(Twice-Cooked Pork)、麻婆豆腐(Mapo Tofu)、宫保鸡丁(Kung Pao Chicken)、小笼包(Xiaolongbao)、北京烤鸭(Peking Duck)。
"""

# 场景优先级：多场景关键词同时命中时，排前面的优先
# 例：“哪里能买到奶茶”同时命中 retail 和 drink → retail 胜出（找店 ≠ 点单）
# direction 排最前：“怎么去故宫”应走问路导航，而不是被“故宫”命中成 ticket
SCENARIO_PRIORITY = ["direction", "travel", "retail", "ticket", "recommend", "drink", "food"]

PROMPTS = {name: CORE_PROMPT + cfg["prompt"] for name, cfg in SCENARIOS.items()}
PROMPTS["general"] = CORE_PROMPT + GENERAL_PROMPT


def _kw_match(keyword: str, text: str) -> bool:
    """英文关键词按单词边界匹配（避免 eat 误命中 great），容忍复数 s；中文按子串匹配。"""
    if keyword.isascii():
        return re.search(rf"\b{re.escape(keyword)}s?\b", text) is not None
    return keyword in text


# 提到连锁品牌时的路由：问路/机酒/找店关键词优先（“最近的德克士在哪”是找店），其次才是在店里点单/求推荐
_BEFORE_BRAND = ("direction", "travel", "retail")


def route_scenario(text: str, context: Optional[dict] = None) -> str:
    """零成本路由：关键词命中 → 对应场景；提到连锁品牌 → 推荐（便利店 → 找店）；
    未命中 → 继承上一场场景（追问环节中简短回答没有关键词）；都没有 → general。"""
    t = (text or "").lower()
    brands = detect_brands(text)
    for name in SCENARIO_PRIORITY:
        if brands and name not in _BEFORE_BRAND:
            # 只提到便利店/超市/药店（“最近的全家”）是找店；否则是在这家店点单/求推荐（“我在茶百道，喝什么好”）
            return "retail" if all(b["category"] in STORE_CATEGORIES for b in brands) else "recommend"
        if any(_kw_match(k, t) for k in SCENARIOS[name]["keywords"]):
            return name
    sticky = (context or {}).get("intent_type")
    if sticky in SCENARIOS:
        return sticky
    return "general"


# 根据 action 补齐下一步提示。
ACTION_INSTRUCTIONS = {
    "show_to_staff": {
        "zh": "请将上方卡片出示给店员看",
        "en": "Please show the card above to the staff."
    },
    "trip_com": {
        "zh": "请前往 Trip.com 查看和预订",
        "en": "Please go to Trip.com to view and book."
    },
}

# 某些场景的 action 只能是这几个，模型给了别的就覆盖成第一个，进一步消除漂移
# 问路场景也可能是“最近的商场怎么走”（目的地是一类地方）→ 允许 search_map
SCENARIO_ALLOWED_ACTIONS = {
    "travel": ("trip_com",),
    "retail": ("search_map",),
    "direction": ("navigate", "search_map"),
}

FALLBACK_CLARIFY = {
    "status": "clarify",
    "intent_type": "other",
    "question": "抱歉，我刚才没太听明白。请问您具体想要什么？（例如：水果茶、汉堡、学生票）",
    "quick_replies": ["饮料", "食物", "门票", "其他"]
}

# 票务的两个决定性槽位：哪一天 + 哪个时段，分开判断。
# （原来“今天/明天”也算作场次，导致“明天的票”不再追问上午/下午，“下午的票”也不问是哪天）
_WEEKDAYS_EN = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend"
_MONTHS_EN = r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
DATE_HINT = re.compile(
    r"今天|明天|后天|今晚|本周|这周|下周|周末|(?:周|星期|礼拜)[一二三四五六日天]|\d{1,2}\s*月\s*\d{1,2}|\d{1,2}\s*[号日]"
    r"|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
    rf"|\b(?:today|tonight|tomorrow|day after tomorrow|{_WEEKDAYS_EN})\b"
    rf"|\b(?:{_MONTHS_EN})[a-z]*\.?\s*\d{{1,2}}\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS_EN})"
    r"|\b\d{1,2}/\d{1,2}\b",
    re.IGNORECASE,
)
TIME_SLOT_HINT = re.compile(
    r"上午|下午|晚上|早上|中午|凌晨|今晚|夜场|日场|\d{1,2}\s*[:：点时]|\b(?:morning|afternoon|evening|tonight|night)\b|\d{1,2}\s*(?:am|pm)\b",
    re.IGNORECASE,
)


def _ticket_clarify(question: str, replies: list[str]) -> dict:
    return {
        "status": "clarify",
        "intent_type": "ticket",
        "question": question,
        "quick_replies": replies + ["其他"],
        "card_content": {"zh": "", "en": ""},
        "action": "",
        "map_query": "",
    }


def ticket_date_options(today: date) -> list[str]:
    """追问日期时的快捷选项：直接给出具体日期，游客不用自己换算。"""
    labels = [("今天", "Today"), ("明天", "Tomorrow"), ("后天", "Day after tomorrow")]
    return [
        f"{zh} {en} ({d.month}/{d.day} {d.strftime('%a')})"
        for (zh, en), d in ((lbl, today + timedelta(days=i)) for i, lbl in enumerate(labels))
    ]


def ticket_session_guardrail(result: dict, scenario: str, convo_text: str) -> dict:
    """票务场景：模型说 success 但对话里没说哪天/哪个时段 → 代码确定性拦截，强制 clarify（先问日期，再问时段）。
    这是决定性槽位的最后防线，不依赖模型自觉：日期或时段错了，买到的就是张用不了的票。"""
    if scenario != "ticket" or result.get("status") != "success":
        return result
    if not DATE_HINT.search(convo_text):
        print("[护栏] 票务日期护栏触发：success 被拦截 → clarify")
        return _ticket_clarify("请问您想要哪一天的票？/ Which day is the ticket for?", ticket_date_options(china_today()))
    if not TIME_SLOT_HINT.search(convo_text):
        print("[护栏] 票务场次护栏触发：success 被拦截 → clarify")
        return _ticket_clarify("请问您想要哪个时间段？/ Which time slot would you like?", ["上午 Morning", "下午 Afternoon", "晚上 Evening"])
    return result


# ============================================================
# 问日期/时间（“今天几号”“What day is it?”）：服务器时钟就是准确答案，不经过模型
# ============================================================
_DATETIME_QUESTION = re.compile(
    r"(今天|明天|后天|今儿)(是)?(几月)?几[号日]|(今天|明天|后天|今儿)(是)?(星期|周|礼拜)几|现在几点|几点了|(今天|明天)(是)?(什么|啥)日子|今天(的)?日期"
    r"|\bwhat(?:'s| is)?\s+(?:the\s+)?date(?:\s+(?:today|tomorrow|now))?\s*\??\s*$"
    r"|\bwhat\s+day\s+(?:is\s+(?:it|today|tomorrow)|of\s+the\s+week)\b"
    r"|\btoday'?s\s+date\b|\bwhat\s+is\s+today\b|\bwhat\s+time\s+is\s+it\b|\bwhat(?:'s| is)\s+the\s+time\b"
    r"|\bwhat(?:'s| is)?\s+(?:the\s+)?date\s+(?:the\s+)?day after tomorrow\s*\??\s*$",
    re.IGNORECASE,
)


def datetime_answer(text: str, now: Optional[datetime] = None) -> Optional[dict]:
    """用户在问今天/明天几号、星期几、现在几点 → 直接给出答案卡（intent_type/action 为 info）；不是这类问题返回 None。"""
    t = (text or "").strip()
    if not _DATETIME_QUESTION.search(t):
        return None
    now = now.astimezone(CHINA_TZ) if now is not None else datetime.now(CHINA_TZ)
    low = t.lower()
    if "后天" in t or "day after tomorrow" in low:
        offset, zh_word, en_word = 2, "后天", "The day after tomorrow"
    elif "明天" in t or "tomorrow" in low:
        offset, zh_word, en_word = 1, "明天", "Tomorrow"
    else:
        offset, zh_word, en_word = 0, "今天", "Today"
    d = now.date() + timedelta(days=offset)
    zh = f"{zh_word}是{d.year}年{d.month}月{d.day}日，星期{_WEEKDAYS_ZH[d.weekday()]}。"
    en = f"{en_word} is {d.strftime('%A, %B')} {d.day}, {d.year}."
    if offset == 0:
        zh += f"现在是北京时间 {now:%H:%M}。"
        en += f" It's {now:%H:%M} China time (UTC+8)."
    return {
        "status": "success",
        "intent_type": "info",
        "question": "",
        "quick_replies": [],
        "card_content": {"zh": zh, "en": en},
        "action": "info",
        "map_query": "",
        "next_action": {"type": "info", "query": None, "origin": None, "city": None, "instruction": {"zh": "日期和时间来自运行本项目的电脑时钟（北京时间）。", "en": "Date and time come from this project's server clock, in China time (UTC+8)."}},
    }


# “A到B / A to B / A→B”——模型把整段行程塞进 map_query 时的分隔符。
# 英文 to 要求单词边界：“观夏to summer”这类店名里的 to 前面紧贴汉字，不会被误切。
_ROUTE_SEPARATOR = re.compile(r"\s*(?:到|→|->|\bto\b)\s*", re.IGNORECASE)
_ROUTE_FROM_PREFIX = re.compile(r"^\s*(?:从|from\b)\s*", re.IGNORECASE)


def split_route_query(query: str) -> tuple[str, str]:
    """把 “复旦大学到南京路” / “Fudan University to Nanjing Road” 拆成 (出发地, 目的地)。
    不是整句行程时返回 ("", 原文)。"""
    query = (query or "").strip()
    parts = [p for p in _ROUTE_SEPARATOR.split(query) if p.strip()]
    if len(parts) < 2:
        return "", query
    return _ROUTE_FROM_PREFIX.sub("", parts[0]).strip(), parts[-1].strip()


def direction_places_guardrail(result: dict, scenario: str) -> dict:
    """问路导航：确定性清洗模型给出的出发地/目的地，并去掉模型自己编的路线说明。
    实测：模型经常把 map_query 写成“复旦大学到南京路”整句，高德搜这个整句会命中“复旦大学(地铁站)”，
    结果把用户导航回出发地。map_query 会被直接拿去地图搜索，所以这里必须用代码兜底，不能只靠 prompt。"""
    is_navigate = scenario == "direction" or result.get("action") == "navigate"
    if not is_navigate or result.get("status") != "success":
        return result
    origin = (result.get("origin_query") or "").strip()
    split_origin, destination = split_route_query(result.get("map_query") or "")
    origin = origin or split_origin
    # 目的地和出发地是同一个地方 = 模型把出发地当成了目的地 → 清空，交给目的地护栏强制 clarify
    if origin and destination.lower() == origin.lower():
        print("[护栏] 问路出发地护栏触发：目的地与出发地相同")
        destination = ""
    result["map_query"] = destination
    result["origin_query"] = origin
    result["city"] = (result.get("city") or "").strip() if isinstance(result.get("city"), str) else ""
    # 换乘路线只来自高德实时数据；模型写的线路/站名/耗时没有依据，不下发
    result["question"] = ""
    return result


def direction_destination_guardrail(result: dict, scenario: str) -> dict:
    """问路导航：模型说 success 但没给出目的地 → 代码确定性拦截，强制 clarify。
    不只看 scenario==direction——general 场景的模型也可能输出 action=navigate，一样要拦。
    没有目的地就没法生成路线，绝不能让空的 navigate 结果流出去。"""
    is_navigate = scenario == "direction" or result.get("action") == "navigate"
    if not is_navigate or result.get("status") != "success":
        return result
    if (result.get("map_query") or "").strip():
        return result
    print("[护栏] 问路目的地护栏触发：success 被拦截 → clarify")
    return {
        "status": "clarify",
        "intent_type": "direction",
        "question": "请问您想去哪里？/ Where would you like to go?",
        "quick_replies": ["景点 Attraction", "商场 Shopping mall", "地铁站 Metro station", "其他"],
        "card_content": {"zh": "", "en": ""},
        "action": "",
        "map_query": ""
    }


def _clean_str(value) -> Optional[str]:
    """模型输出的可选字符串字段：非字符串/空白 → None。"""
    return value.strip() or None if isinstance(value, str) else None


def attach_next_action(result: dict, scenario: str = "general") -> dict:
    """根据模型输出的 action / map_query，用代码补齐双语的 next_action。"""
    if result.get("status") != "success":
        result.pop("next_action", None)
        return result

    action = result.get("action") or "show_to_staff"
    # 场景级约束：机酒/找店/问路的 action 只能是固定几种
    allowed = SCENARIO_ALLOWED_ACTIONS.get(scenario)
    if allowed and action not in allowed:
        action = allowed[0]
    query = (result.get("map_query") or "").strip()

    if action == "search_map" and query:
        instruction = {
            "zh": f"为你找到附近的“{query}”，并规划前往最近一家的路线",
            "en": f"Here are the nearest places for '{query}', with a route to the closest one."
        }
    elif action == "navigate" and query:
        origin = (result.get("origin_query") or "").strip()
        instruction = {
            "zh": f"为你规划{f'从“{origin}”' if origin else ''}前往“{query}”的路线：地铁换乘或打车",
            "en": f"Here's how to get {f'from {origin!r} ' if origin else ''}to '{query}': by metro (with line changes) or by car."
        }
    elif action == "trip_com":
        instruction = ACTION_INSTRUCTIONS["trip_com"]
    elif action in ("search_map", "navigate") and not query:
        # search_map / navigate 必须有 query，否则退化为展示卡
        action = "show_to_staff"
        instruction = ACTION_INSTRUCTIONS["show_to_staff"]
    else:
        action = "show_to_staff"
        instruction = ACTION_INSTRUCTIONS["show_to_staff"]

    # 导航去只能网上预约的景点：路线卡片上提醒游客先预约（现场不卖票）
    attraction = detect_attraction(query) if action == "navigate" and query else None
    result["next_action"] = {
        "type": action,
        "booking": booking_info(attraction) if attraction else None,
        "query": query or None,
        "origin": _clean_str(result.get("origin_query")) if action in ("navigate", "search_map") else None,
        "city": _clean_str(result.get("city")) if action in ("navigate", "search_map") else None,
        "instruction": instruction
    }
    return result


def brand_query_guardrail(result: dict, scenario: str, user_text: str) -> dict:
    """地图搜索词必须是品牌的真实中文店名：实测模型会把 “Auntea Jenny” 直译成不存在的“阿姨珍妮奶茶”，地图搜不到。
    - 搜索词本身就是某品牌的英文名/拼音（Dicos）→ 换成中文店名
    - 找店（search_map）且用户这句话只提到一个品牌 → 搜的就是这家店"""
    if result.get("status") != "success":
        return result
    query = (result.get("map_query") or "").strip()
    if not query:
        return result
    brand = brand_for_query(query)
    is_search = result.get("action") == "search_map" or scenario == "retail"
    mentioned = detect_brands(user_text)
    if not brand and is_search and len(mentioned) == 1:
        brand = mentioned[0]
    if brand and query != brand["zh"]:
        print(f"[护栏] 品牌搜索词护栏触发：{query} → {brand['zh']}")
        result["map_query"] = brand["zh"]
    return result


_MENU_NOTE = "（具体菜名以店内菜单为准 / Check the in-store menu for the exact name.）"


def unverified_brand_note_guardrail(result: dict, convo_text: str) -> dict:
    """在没有核实过招牌的连锁店里推荐菜：模型只能推这类店的常见菜（如烤鱼店推烤鱼），菜名不一定和店里一字不差。
    实测模型会把它说成“这家店的招牌之一”、也不加提醒——这里用代码固定加上提醒，不依赖模型自觉。"""
    if result.get("status") != "success" or result.get("action") not in ("show_to_staff", None, ""):
        return result
    if not unverified_brands(detect_brands(convo_text)):
        return result
    question = (result.get("question") or "").strip()
    if _MENU_NOTE not in question:
        result["question"] = f"{question}{_MENU_NOTE}"
    return result


def finalize_result(content: str, scenario: str = "general", convo_text: str = "", user_text: str = "") -> dict:
    """把大模型的原始输出解析、兜底、护栏校验、补齐 next_action。两个接口共用同一套后处理。"""
    # 清理可能的 Markdown 代码块标记
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        result_json = json.loads(content)
    except json.JSONDecodeError:
        print(f"[警告] 大模型返回了非 JSON 内容，已自动降级处理。原始内容：{content}")
        return dict(FALLBACK_CLARIFY)

    # 决定性槽位护栏（票务场次 / 问路出发地与目的地）
    result_json = ticket_session_guardrail(result_json, scenario, convo_text)
    result_json = direction_places_guardrail(result_json, scenario)
    result_json = direction_destination_guardrail(result_json, scenario)
    result_json = brand_query_guardrail(result_json, scenario, user_text)
    result_json = unverified_brand_note_guardrail(result_json, convo_text)

    # clarify 状态下强制保证有“其他”选项
    if result_json.get("status") == "clarify":
        replies = result_json.get("quick_replies") or []
        if not any(str(r).lower() in ["other", "其他", "其它"] for r in replies):
            replies.append("其他")
        result_json["quick_replies"] = replies

    return attach_next_action(result_json, scenario)


# 1. 升级数据结构，接收历史对话 + 前端维护的紧凑会话状态
class UserInput(BaseModel):
    text: str = ""
    image_base64: Optional[str] = None
    history: list[dict] = []
    context: Optional[dict] = None  # 如：{"answering": "上一个追问的问题", "intent_type": "drink"}
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    ui_language: Optional[str] = None


SUPPORTED_LANGUAGES = {
    "en": "English", "zh": "Chinese", "es": "Spanish", "hi": "Hindi",
    "ar": "Arabic", "pt": "Portuguese", "fr": "French", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean",
}


def language_prompt(data: UserInput) -> str:
    # 未指定沟通语言时，沿用中英双语返回格式。
    source = SUPPORTED_LANGUAGES.get(data.source_language)
    target = SUPPORTED_LANGUAGES.get(data.target_language)
    interface = SUPPORTED_LANGUAGES.get(data.ui_language) or source
    if not source or not target:
        return ""
    return (
        f"\n【用户语言偏好】输入语言：{source}；沟通卡目标语言：{target}。"
        "理解用户输入的任何语言，不要将中文示例提示词误认为要求切换语言。"
        f"question 和 quick_replies 必须使用界面语言 {interface}，不受输入语言或示例英语的影响。仍保留 card_content.zh/en 和原有 action/map_query/origin_query 字段。"
        "地图关键词和地点名称仍用中文；绝不改动路线护栏规则。"
        "success 时额外输出 localized_card 对象："
        f'{{"source_language":"{data.source_language}","target_language":"{data.target_language}",'
        f'"source":"与 card_content 相同含义的 {source} 表达",'
        f'"target":"与 card_content 相同含义的 {target} 表达"}}。'
        "保持同一组事实、数量、偏好和第一人称口吻，不要补充新信息；两种语言相同时仍填写两个字段。"
        f'额外输出 localized_ui: {{"language":"{data.ui_language or data.source_language}","instruction":"使用 {interface} 描述下一步操作"}}。'

    )


def ensure_localized_card(result: dict, data: UserInput) -> dict:
    """Some streaming models omit extension fields. Fill translations without changing actions."""
    source, target = data.source_language, data.target_language
    if source not in SUPPORTED_LANGUAGES or target not in SUPPORTED_LANGUAGES:
        return result
    if result.get("status") != "success":
        return result
    card = result.get("card_content") or {}
    original = _clean_str(card.get("zh")) or _clean_str(card.get("en"))
    if not original:
        return result
    localized = result.get("localized_card")
    if isinstance(localized, dict) and localized.get("source_language") == source and localized.get("target_language") == target and _clean_str(localized.get("source")) and _clean_str(localized.get("target")):
        return result
    result.pop("localized_card", None)
    try:
        response = client.with_options(timeout=20, max_retries=0).chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {"role": "system", "content": (
                    "You are a precise translator. Translate the supplied action card, keeping every quantity, preference, place and first-person meaning. "
                    "Do not add advice or change the request. Return JSON only with two string fields: source and target. "
                    f"source must be in {SUPPORTED_LANGUAGES[source]}; target must be in {SUPPORTED_LANGUAGES[target]}."
                )},
                {"role": "user", "content": original},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        translated = json.loads(response.choices[0].message.content)
        if _clean_str(translated.get("source")) and _clean_str(translated.get("target")):
            result["localized_card"] = {"source_language": source, "target_language": target, "source": translated["source"], "target": translated["target"]}
    except Exception:
        # The UI explicitly reports the original bilingual fallback; never relabel it.
        result["translation_warning"] = "selected_translation_unavailable"
    return result


# 前端在每次出卡（success）后往历史里记一条以此开头的助手消息：它标志着上一个任务已经结束
_TASK_DONE_PREFIX = "生成执行卡"


def current_task_history(history: list) -> list[dict]:
    """只取当前这个任务的对话（上一次出卡之后的部分）。
    历史里还留着之前已完成的任务：实测先问了“今天几号”（答案里有“10月3日”），
    再买票时日期护栏扫到这个旧日期，就不再追问票是哪天的，卡片上没有日期。品牌检测同理。"""
    turns = [h for h in (history or []) if isinstance(h, dict)]
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].get("role") == "assistant" and str(turns[i].get("content", "")).startswith(_TASK_DONE_PREFIX):
            return turns[i + 1:]
    return turns


def convo_text_of(data: UserInput) -> str:
    """只扫描当前任务中用户确认的内容，追问里的日期/时段选项不算用户选择。"""
    parts = [data.text or ""]
    parts += [str(h.get("content", "")) for h in current_task_history(data.history) if h.get("role") == "user"]
    return " ".join(parts)


# ============================================================
# 相对日期使用北京时间日历换算。每次请求附上未来两周日期，固定场景提示词放在前面。
# ============================================================
try:
    CHINA_TZ = ZoneInfo("Asia/Shanghai")
except ZoneInfoNotFoundError:
    # Windows 首次安装离线时也能读取当前北京时间。
    CHINA_TZ = timezone(timedelta(hours=8), "Asia/Shanghai")
_WEEKDAYS_ZH = "一二三四五六日"
_RELATIVE_DAY_LABELS = {0: "今天 / today", 1: "明天 / tomorrow", 2: "后天 / the day after tomorrow"}
# 周一作为一周的起点，供“下周一”等表达查表。
_WEEK_LABELS = {0: "本周 / this week", 1: "下周 / next week", 2: "下下周 / the week after next"}
CALENDAR_DAYS = 14


def china_today() -> date:
    return datetime.now(CHINA_TZ).date()


def calendar_prompt(today: date) -> str:
    """给模型的日历：今天起两周内每天的日期和星期，外加相对日期的换算规则。"""
    rows = []
    this_monday = today - timedelta(days=today.weekday())
    for i in range(CALENDAR_DAYS):
        d = today + timedelta(days=i)
        week = _WEEK_LABELS[(d - this_monday).days // 7]
        label = _RELATIVE_DAY_LABELS.get(i, "")
        rows.append(f"{d.isoformat()} 周{_WEEKDAYS_ZH[d.weekday()]} {d.strftime('%a, %b')} {d.day} | {week} | {label}".rstrip(" |"))
    return f"""
【日历（中国时间，查表用，绝不自己推算日期或星期）】
今天是 {today.year}年{today.month}月{today.day}日 星期{_WEEKDAYS_ZH[today.weekday()]}（{today.strftime('%A, %B')} {today.day}, {today.year}）。
""" + "\n".join(rows) + """
日期规则：
1. 用户说了相对日期（今天/明天/后天/周六/这周六/下周一/tomorrow/this Saturday/next Monday）→ 必须对照上表换算成具体日期，写进 card_content：中文写“10月3日（周六）”这种格式，英文写“Sat, Oct 3”这种格式。出示给店员的卡片里绝不能只写“明天”“周六”这种相对说法，也绝不能把用户说的日期丢掉。
2. 换算时直接查表里的“本周/下周/下下周”列：“这周X / 本周X / this X”→ 本周那一行的周X（已经过去的话就是从今天起最近的周X）；“下周X / next X”→ 标着“下周 / next week”那一行的周X；单说“周X / on X”→ 从今天起（含今天）最近的周X。
3. 用户没说日期 → 不要替他编日期，卡片里不写日期。
"""


def build_messages(data: UserInput, scenario: str) -> list:
    """组装 system（场景专属 prompt + 当天日历）+ 历史对话。历史只保留最近 6 条，避免 token 随轮次无限增长。"""
    # 品牌资料按整段对话检测：追问环节里用户只回答“少糖”，也要记得他在哪家店
    brands = detect_brands(convo_text_of(data))
    messages = [{"role": "system", "content": PROMPTS[scenario] + calendar_prompt(china_today()) + brand_prompt(brands) + language_prompt(data)}]
    if data.history:
        for h in current_task_history(data.history)[-6:]:
            if isinstance(h, dict) and "role" in h and "content" in h:
                messages.append({"role": h["role"], "content": h["content"]})
    return messages


def build_user_text(data: UserInput) -> str:
    """若用户正在回答追问，把回答显式绑定到问题上，避免“饮料”这类简短回答被当成全新需求。"""
    answering = (data.context or {}).get("answering")
    if answering:
        return (
            f"（用户正在回答上一个追问：“{answering}”）\n"
            f"用户的回答：{data.text}\n"
            f"请结合对话历史中已确认的信息，判断现在是否可以生成执行卡；若仍缺核心参数，继续针对性追问。"
        )
    return data.text


def localize_direct_answer(answer: dict, data: UserInput) -> dict:
    """确定性中英答案沿用当前 UI 的语言字段；其他语种沿用原翻译补全。"""
    card = answer.get("card_content") or {}
    if data.source_language in ("zh", "en") and data.target_language in ("zh", "en"):
        answer["localized_card"] = {
            "source_language": data.source_language, "target_language": data.target_language,
            "source": card.get(data.source_language, ""), "target": card.get(data.target_language, ""),
        }
        return answer
    if not os.getenv("LLM_API_KEY"):
        if data.source_language in SUPPORTED_LANGUAGES and data.target_language in SUPPORTED_LANGUAGES:
            answer["translation_warning"] = "selected_translation_unavailable"
        return answer
    return ensure_localized_card(answer, data)


@app.get("/api/health")
def health():
    return {"code": 200, "data": {"service": "travel-ai-v3-merged", "revision": os.getenv("TRAVEL_DEMO_REVISION", "development"), "project": os.getenv("TRAVEL_DEMO_ROOT", "development"), "llm_configured": bool(os.getenv("LLM_API_KEY")), "timezone": "Asia/Shanghai"}}


@app.get("/api/time")
def local_time():
    now = datetime.now(CHINA_TZ)
    return {"code": 200, "data": {"datetime": now.isoformat(), "date": now.date().isoformat(), "time": now.strftime("%H:%M:%S"), "weekday": now.isoweekday(), "timezone": "Asia/Shanghai", "utc_offset": "+08:00", "source": "server_clock"}}


def attraction_booking_answer(data: UserInput) -> Optional[dict]:
    """要买票/参观只能网上实名预约的景点（故宫、天安门广场、兵马俑…）→ 直接给“提前网上预约”卡片，不经过模型。
    这些景点现场不卖票，出一张“我要一张门票”的购票卡让游客去窗口只会白跑。问怎么走的照常走路线卡（卡片上附预约提醒）。"""
    convo = convo_text_of(data)
    attraction = detect_attraction(convo)
    if not attraction or not wants_ticket(convo):
        return None
    if route_scenario(data.text, data.context) == "direction":
        return None
    return booking_answer(attraction)


@app.post("/api/parse_intent")
def parse_intent(data: UserInput):
    answer = None if data.image_base64 else (datetime_answer(data.text) or attraction_booking_answer(data))
    if answer:
        return {"code": 200, "data": localize_direct_answer(answer, data)}
    if not os.getenv("LLM_API_KEY"):
        return {"code": 503, "data": {"status": "clarify", "question": "请先在 hackathon/backend/.env 配置模型密钥。日期和时间查询可直接使用。 / Configure your model key to use AI; date and time queries work without a key.", "quick_replies": ["今天几号", "现在几点", "其他"]}}
    scenario = route_scenario(data.text, data.context)
    print(f"[路由] scenario: {scenario} | text: {data.text[:40]}")
    try:
        messages = build_messages(data, scenario)

        # 图片请求使用视觉模型。
        if data.image_base64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": data.text if data.text else "请识别图片中的需求并生成执行卡。"},
                    {"type": "image_url", "image_url": {"url": data.image_base64}}
                ]
            })
            response = client.chat.completions.create(
                model="glm-4v-flash",
                messages=messages,
                temperature=0.1
            )
        else:
            messages.append({"role": "user", "content": build_user_text(data)})
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}}
            )

        content = response.choices[0].message.content
        print(f"[usage] {response.usage}")
        return {"code": 200, "data": ensure_localized_card(finalize_result(content, scenario, convo_text_of(data), data.text), data)}

    except Exception as e:
        print(f"[错误] LLM Error: {e}")
        return {
            "code": 500,
            "data": {
                "status": "clarify",
                "question": f"系统内部错误：{str(e)}",
                "quick_replies": ["重试"]
            }
        }


class OriginInput(BaseModel):
    origin: Optional[str] = None  # 用户在对话里明确说的出发地（如“复旦大学”）；优先于手机定位
    city: Optional[str] = None    # 用户在对话里明确提到的城市（如“北京”），地名搜索最优先在这个城市
    origin_lat: Optional[float] = None
    origin_lng: Optional[float] = None


class DirectionsInput(OriginInput):
    destination: str
    # 目的地已经确定到具体地点（如附近搜索选中的那家商场）时直接给坐标，不再按名字搜
    destination_lon: Optional[float] = None
    destination_lat: Optional[float] = None


class NearbyInput(OriginInput):
    query: str  # 地点类别，如“商场”“药店”


def _same_city(a: Optional[str], b: Optional[str]) -> bool:
    """高德各接口返回的城市名格式不完全一致（“北京市”/“北京”），比较前去掉“市”。缺失时不判为跨城。"""
    if not a or not b:
        return True
    return a.removesuffix("市") == b.removesuffix("市")


def resolve_origin(data: OriginInput) -> dict:
    """确定起点：用户说了出发地就用出发地（游客可能在提前规划，人不在那里），否则用手机定位。
    返回 {"ok": True, "origin", "origin_name", "origin_city", "gps_city", "city_hint"} 或 {"ok": False, "reason"}。"""
    has_gps = data.origin_lat is not None and data.origin_lng is not None
    origin_text = (data.origin or "").strip()
    city_hint = (data.city or "").strip() or None
    if not origin_text and not has_gps:
        return {"ok": False, "reason": "no_origin"}

    gps = amap_gps_to_gcj02(data.origin_lat, data.origin_lng) if has_gps else None
    gps_city = amap_reverse_geocode(gps["lon"], gps["lat"]) if gps else None

    if origin_text:
        origin_place = find_place(origin_text, [city_hint, gps_city])
        if not origin_place:
            return {"ok": False, "reason": "origin_geocode_failed"}
        return {
            "ok": True,
            "origin": {"lon": origin_place["lon"], "lat": origin_place["lat"]},
            "origin_name": origin_place["formatted_address"],
            "origin_city": origin_place["city"] or city_hint or gps_city,
            "gps_city": gps_city,
            "city_hint": city_hint,
        }
    if not gps:
        return {"ok": False, "reason": "convert_failed"}
    return {"ok": True, "origin": gps, "origin_name": None, "origin_city": gps_city, "gps_city": gps_city, "city_hint": city_hint}


@app.post("/api/directions")
def directions(data: DirectionsInput):
    """真实的地铁/公交换乘路线（哪一站上车、在哪换乘、哪一站下车），近距离给步行路线。全程确定性地图 API 调用，不经过大模型。
    地名搜索的城市优先级：用户说的城市 → 出发地所在城市 → 手机定位所在城市 → 全国（见 find_place）。"""
    start = resolve_origin(data)
    if not start["ok"]:
        return {"code": 200, "data": start}
    origin, origin_city = start["origin"], start["origin_city"]

    if data.destination_lon is not None and data.destination_lat is not None:
        dest = {"lon": data.destination_lon, "lat": data.destination_lat, "formatted_address": data.destination,
                "city": amap_reverse_geocode(data.destination_lon, data.destination_lat) or origin_city}
    else:
        dest = find_place(data.destination, [start["city_hint"], origin_city, start["gps_city"]])
    if not dest:
        return {"code": 200, "data": {"ok": False, "reason": "geocode_failed"}}

    # 跨城（上海 → 北京天安门）：市内公交/地铁换乘接口给不出靠谱方案，不硬凑路线，引导订火车/机票
    if not _same_city(origin_city, dest["city"]):
        return {"code": 200, "data": {
            "ok": False, "reason": "cross_city",
            "origin_name": start["origin_name"], "origin_city": origin_city,
            "origin_city_en": place_name_pinyin_en(origin_city.removesuffix("市")),
            "destination_name": dest["formatted_address"], "destination_city": dest["city"],
            "destination_city_en": place_name_pinyin_en(dest["city"].removesuffix("市")),
        }}

    city = origin_city or dest["city"]
    route = amap_route(origin, dest, city)
    if not route:
        return {"code": 200, "data": {"ok": False, "reason": "no_route"}}

    return {"code": 200, "data": attach_english_names({
        "ok": True,
        "origin_name": start["origin_name"],
        "origin_lon": origin["lon"],
        "origin_lat": origin["lat"],
        "destination_name": dest["formatted_address"],
        "city": city,
        **route,
    })}


@app.post("/api/nearby")
def nearby(data: NearbyInput):
    """“最近的商场/药店在哪”：以用户说的位置（或手机定位）为圆心，按距离由近到远返回最近的三个地点。
    只给名称和距离，不规划路线；前端的地图/打车按钮直接指向用户选中的那一个。"""
    start = resolve_origin(data)
    if not start["ok"]:
        return {"code": 200, "data": start}
    places = amap_search_nearby(data.query, start["origin"])
    if not places:
        return {"code": 200, "data": {"ok": False, "reason": "none_nearby"}}
    for pl in places:
        has_latin = re.search(r"[A-Za-z]", pl["name"])
        pl["name_en"] = _NAME_EN_CACHE.get(pl["name"]) or (None if has_latin else place_name_pinyin_en(pl["name"]))
    return {"code": 200, "data": {
        "ok": True,
        "origin_name": start["origin_name"],
        "origin_name_en": (_NAME_EN_CACHE.get(start["origin_name"]) or place_name_pinyin_en(start["origin_name"])) if start["origin_name"] else None,
        "origin_lon": start["origin"]["lon"],
        "origin_lat": start["origin"]["lat"],
        "city": start["origin_city"],
        "places": places,
    }}


class MenuInput(BaseModel):
    image_base64: str


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/menu")
def menu_photo(data: MenuInput):
    """拍菜单翻译（SSE）：逐个推送转写出的菜（中文原文 + 价格 + 英文价格单位），同时在后台边读边翻译，分批推送英文菜名和说明。
    事件：{"stage": "reading"}（转写中心跳）、{"item": {...}}、{"translations": {菜名: {en, desc}}}，
         最后 {"done": true, "count": N}；照片不是菜单时 count 为 0；转写模型全部失败时 {"error": "read_failed"}。"""
    if len(data.image_base64) > 16 * 1024 * 1024 or decode_image(data.image_base64) is None:
        raise HTTPException(status_code=422, detail="Please upload a valid menu image under 12 MB.")
    if not os.getenv("LLM_API_KEY"):
        return StreamingResponse(iter([_sse({"error": "model_not_configured"}), _sse({"done": True, "count": 0})]), media_type="text/event-stream")
    env_models = tuple(m.strip() for m in (os.getenv("MENU_OCR_MODELS") or "").split(",") if m.strip())
    ocr_models = env_models or MENU_OCR_MODELS_DEFAULT
    translate_model = os.getenv("MENU_TRANSLATE_MODEL") or MENU_TRANSLATE_MODEL_DEFAULT
    translate_models = tuple(dict.fromkeys(model for model in (translate_model, os.getenv("LLM_MODEL")) if model))
    events: "queue.Queue[dict]" = queue.Queue()
    to_translate: "queue.Queue[Optional[list[str]]]" = queue.Queue()
    # 客户端断开后停止转写和翻译，释放上游请求。
    cancelled = threading.Event()

    def translator():
        """从队列取菜名批次，优先用菜单翻译模型，失败后交给主对话模型。"""
        while True:
            batch = to_translate.get()
            if batch is None or cancelled.is_set():
                break
            for model in translate_models:
                if cancelled.is_set():
                    break
                try:
                    for zh, tr in translate_menu_items(client, model, batch):
                        if cancelled.is_set():
                            break
                        events.put({"translations": {zh: tr}})
                    break
                except Exception as exc:
                    # 已翻译的菜会命中缓存；都失败时仍保留中文和价格。
                    print(f"[菜单翻译失败] {model}: {type(exc).__name__}")
        events.put({"_translator_done": True})

    def reader():
        """转写线程：读出的菜立刻推给前端，攒够一批就交给翻译线程。"""
        count = 0
        pending: list[str] = []
        queued = set()
        queued_batches = 0
        last_beat = 0.0
        try:
            for ev in ocr_menu_items(client, ocr_models, data.image_base64):
                if cancelled.is_set():
                    break
                if ev.get("thinking"):
                    now = time.monotonic()
                    if now - last_beat > 2:  # 备选模型先思考几十秒才开始输出菜名：每 2 秒一次心跳
                        last_beat = now
                        events.put({"stage": "reading"})
                    continue
                if "part" in ev:
                    # 横幅照片切成几条逐条读：告诉前端读到第几条了；上一条剩下不够一批的菜先送去翻译，不用等下一条读满
                    events.put({"stage": "reading", "part": ev["part"], "parts": ev["parts"]})
                    if pending:
                        queued_batches += 1
                        to_translate.put(pending)
                        pending = []
                    continue
                events.put({"item": {"id": count, **ev, "price_en": format_price_en(ev["price"]), "options": price_options(ev["price"])}})
                count += 1
                if ev["zh"] not in queued:
                    queued.add(ev["zh"])
                    pending.append(ev["zh"])
                if len(pending) >= (TRANSLATE_BATCH if queued_batches else TRANSLATE_FIRST_BATCH):
                    queued_batches += 1
                    to_translate.put(pending)
                    pending = []
        except MenuReadError:
            events.put({"error": "read_failed"})
        except Exception as exc:
            print(f"[菜单读取错误] {type(exc).__name__}")
            events.put({"error": "read_failed"})
        finally:
            if pending and not cancelled.is_set():
                to_translate.put(pending)
            for _ in range(TRANSLATE_WORKERS):
                to_translate.put(None)
            events.put({"_reader_done": True, "count": count})

    def event_stream():
        for _ in range(TRANSLATE_WORKERS):
            threading.Thread(target=translator, daemon=True).start()
        threading.Thread(target=reader, daemon=True).start()
        reader_done = False
        translators_left = TRANSLATE_WORKERS
        count = 0
        try:
            while not reader_done or translators_left:
                try:
                    ev = events.get(timeout=1)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if ev.get("_reader_done"):
                    reader_done, count = True, ev["count"]
                    if count:
                        yield _sse({"stage": "translating"})
                elif ev.get("_translator_done"):
                    translators_left -= 1
                else:
                    yield _sse(ev)
            yield _sse({"done": True, "count": count})
        finally:
            # 正常结束或客户端断开（生成器被关闭）都会走到这里
            cancelled.set()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class TranslateNamesInput(BaseModel):
    names: list[str]


@app.post("/api/translate_names")
def translate_names_api(data: TranslateNamesInput):
    """英文界面在路线出来后异步调用：用模型把站名/地名/线路名换成官方英文名（如 外滩 → The Bund）。
    只返回模型译出的名字；没译出的，前端继续用路线接口给的兜底译名。"""
    names = [n for n in data.names[:50] if isinstance(n, str) and n.strip()]
    # 常见线路名、纯路名的确定性译名就是官方译名，不送模型、也不让模型覆盖
    translations = translate_names([n for n in names if not line_name_en(n) and not is_road_name(n)])
    return {"code": 200, "data": {"translations": translations}}


@app.post("/api/parse_intent/stream")
async def parse_intent_stream(data: UserInput):
    """文字场景的流式接口：SSE 逐块推送 delta 用于实时预览，
    流结束后推送一个 final 事件（已解析 + 兜底 + 护栏 + 补齐 next_action 的完整结果）。"""
    answer = None if data.image_base64 else (datetime_answer(data.text) or attraction_booking_answer(data))
    if answer:
        def answer_stream():
            yield f"data: {json.dumps({'final': localize_direct_answer(answer, data)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(answer_stream(), media_type="text/event-stream")
    if not os.getenv("LLM_API_KEY"):
        def unconfigured_stream():
            yield _sse({"final": {"status": "clarify", "question": "请先在 hackathon/backend/.env 配置模型密钥。 / Configure your model key to use AI.", "quick_replies": ["今天几号", "现在几点", "其他"]}})
            yield "data: [DONE]\n\n"
        return StreamingResponse(unconfigured_stream(), media_type="text/event-stream")
    scenario = route_scenario(data.text, data.context)
    print(f"[路由][stream] scenario: {scenario} | text: {data.text[:40]}")
    messages = build_messages(data, scenario)
    messages.append({"role": "user", "content": build_user_text(data)})
    convo_text = convo_text_of(data)

    def event_stream():
        full_text = ""
        try:
            stream = client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    full_text += delta
                    yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
            # 后处理统一在后端完成：解析、清洗、兜底、护栏、补齐 next_action
            final = ensure_localized_card(finalize_result(full_text, scenario, convo_text, data.text), data)
            yield f"data: {json.dumps({'final': final}, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"[错误] LLM Stream Error: {e}")
            yield f"data: {json.dumps({'final': {'status': 'clarify', 'question': f'系统内部错误：{e}', 'quick_replies': ['重试']}}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
