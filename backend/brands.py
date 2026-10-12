"""匹配品牌的中文名、英文名和拼音别名，并为提示词与地图搜索补充本地资料。"""
import re
from typing import Optional

from pypinyin import lazy_pinyin

from brands_data import BRANDS

CATEGORY_ZH = {
    "tea": "奶茶/果茶店", "coffee": "咖啡店", "fast_food": "西式快餐店（汉堡/炸鸡/披萨）",
    "chinese_fast_food": "中式快餐/小吃店", "noodles": "面馆", "dumplings": "饺子/馄饨店",
    "malatang": "麻辣烫店（自选称重）", "hotpot": "火锅店", "restaurant": "中餐馆",
    "bakery": "面包甜点店", "braised_snacks": "卤味熟食店",
    "convenience": "便利店", "supermarket": "超市", "pharmacy": "药店", "retail": "生活百货店",
}
# 这些品类的店是“去买东西/找店”，不是“在店里点单”：路由到找店场景
STORE_CATEGORIES = {"convenience", "supermarket", "pharmacy", "retail"}

# 有歧义的中文叫法，后面跟着这些词才算是在说这家店（“一点点店”✓，“一点点辣”✗）
_SHOP_SUFFIX = r"(?:店|门店|分店|餐厅|饭店|超市|药房|奶茶|咖啡)"
_HAN = re.compile(r"[一-鿿]")


def _normalize(text: str) -> str:
    """小写、统一撇号和连字符，方便匹配 “McDonald’s” “Auntea-Jenny” 这类写法。"""
    t = (text or "").lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"[-_]+", " ", t)


def _alias_pattern(alias: str) -> re.Pattern:
    """英文/拼音别名按单词边界匹配（coco 不能命中 coconut），别名内部空格可有可无（auntea jenny / aunteajenny）。"""
    parts = [re.escape(p) for p in _normalize(alias).split()]
    return re.compile(r"(?<![a-z0-9])" + r"\s*".join(parts) + r"(?![a-z0-9])")


def _pinyin_alias(zh: str) -> Optional[str]:
    """中文店名 → 拼音别名（“杨国福” → “yang guo fu”，匹配时空格可有可无，也就认 “yangguofu”）。
    至少两个音节、五个字母，太短的拼音容易撞上普通英文。
    中文里夹着字母/数字的店名（“Manner咖啡”“7分甜”“1点点”）不生成：只取汉字会得到 “ka fei”（咖啡）这种泛称，
    它们的英文名已经在 aliases 里了。多音字读错的（呷哺 ga→xia、大参林 can→shen）在 brands_data 里手写正确拼音。"""
    if re.search(r"[A-Za-z0-9]", zh):
        return None
    han = "".join(_HAN.findall(zh))
    syllables = lazy_pinyin(han)
    if len(syllables) < 2 or len("".join(syllables)) < 5:
        return None
    return " ".join(syllables)


def _build_matchers():
    matchers = []
    for brand in BRANDS:
        ambiguous = brand.get("ambiguous_zh", [])
        zh_plain = [brand["zh"], *brand.get("zh_names", [])]
        zh_plain = [z for z in zh_plain if z not in ambiguous]
        aliases = list(brand["aliases"])
        for zh in zh_plain + ambiguous:
            py = _pinyin_alias(zh)
            if py:
                aliases.append(py)
        patterns = [_alias_pattern(a) for a in dict.fromkeys(aliases) if a.isascii()]
        # 中文名大小写不敏感（“CoCo都可”“Manner咖啡”）：比较前和用户文本一样转小写
        patterns += [re.compile(re.escape(z.lower())) for z in zh_plain if _HAN.search(z)]
        patterns += [re.compile(re.escape(z.lower()) + _SHOP_SUFFIX) for z in ambiguous]
        matchers.append((brand, patterns))
    return matchers


_MATCHERS = _build_matchers()


def detect_brands(text: str) -> list[dict]:
    """找出文本里提到的连锁品牌（英文名/拼音/中文名都认），按出现顺序返回，不重复。"""
    t = _normalize(text)
    found = []
    for brand, patterns in _MATCHERS:
        positions = [m.start() for p in patterns for m in [p.search(t)] if m]
        if positions:
            found.append((min(positions), brand))
    return [b for _, b in sorted(found, key=lambda x: x[0])]


def brand_for_query(query: str) -> Optional[dict]:
    """地图搜索词如果其实是某个品牌的别名（Dicos / dicos），返回该品牌，用于换成中文店名。"""
    hits = detect_brands(query)
    return hits[0] if len(hits) == 1 else None


def is_brand_store(brand: dict, poi_name: str) -> bool:
    """地图搜到的店是不是这个品牌的门店。高德关键词搜索是模糊的：实测搜“幸运咖”最近的结果是
    “WE LUCKY COFFEE 小幸运咖啡”“Hubers Coffee”这些别家的店，不能把游客带过去。
    连锁门店在高德上都以品牌名开头（“幸运咖(xx店)”“西贝(xx店)”“7-ELEVEn(xx店)”），所以品牌名必须出现在
    店名开头或分隔符之后——“小幸运咖啡”“沪办大厦幸运咖啡店”里的“幸运咖”前面紧挨着别的字，是别家的店。"""
    name = poi_name.lower()
    names = [brand["zh"], *brand.get("zh_names", []), *brand.get("ambiguous_zh", []), brand["en"], *brand["aliases"]]
    for n in (x.lower() for x in names if len(x) >= 2):
        for m in re.finditer(re.escape(n), name):
            before = name[m.start() - 1] if m.start() else ""
            if not before or not (before.isalnum() or _HAN.match(before)):
                return True
    return False


def brand_prompt(brands: list[dict]) -> str:
    """命中品牌时追加到 system prompt 的真实资料。"""
    if not brands:
        return ""
    lines = []
    for b in brands:
        said = "、".join(b["aliases"][:3])
        line = f"- {b['zh']}（英文名 {b['en']}{f'；用户可能说：{said}' if said else ''}）：{CATEGORY_ZH[b['category']]}。"
        if b["signatures"]:
            line += "招牌：" + "；".join(f"{zh}（{en}）" for zh, en in b["signatures"]) + "。"
        if b.get("how_to_order"):
            line += "点单方式：" + b["how_to_order"]
        lines.append(line)
    return """
【用户提到的连锁品牌（真实资料，以此为准，绝不自己编店名或单品）】
""" + "\n".join(lines) + """
品牌规则：
1. 卡片和 question 里提到店名时用上面的中文店名；需要地图搜索这家店时（action=search_map/navigate），map_query 必须填上面的中文店名，绝不能把英文名直译成别的中文（如 Auntea Jenny 必须是“沪上阿姨”，绝不是“阿姨珍妮”）。
2. 用户在这家店点单/求推荐 → 用户没指定具体单品时，从上面的招牌里选最符合用户偏好的一款，question 写推荐理由（用户可纠正）；用户只说了品类（如“汉堡”“奶茶”），从招牌里选该品类的具体单品，绝不能只写“一个汉堡”这种店员没法下单的泛称。
3. 没有列出招牌的店：用户点了具体单品就照写；用户求推荐时，按“点单方式”提示生成卡片，或推荐这类店最常见的经典款（如面馆推牛肉面），绝不编这家店的专属菜名，也绝不能说它是这家店的招牌。
4. 用户点了招牌列表以外的单品，照用户说的写，不要替换。
"""


def unverified_brands(brands: list[dict]) -> list[dict]:
    """在店里点单、但没有核实过招牌单品的品牌（找店类品牌不算）：推荐出来的是同类店的常见菜，菜名不一定和店里一致。"""
    return [b for b in brands if not b["signatures"] and b["category"] not in STORE_CATEGORIES]
