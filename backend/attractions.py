"""景点官方渠道和预约提示。规则按景点分别处理，不统一断言现场不售票。
核验来源见项目根目录 修改文档.md；未核验的放票时刻不作为确定事实输出。
"""
import re
from typing import Optional

from brands import _alias_pattern, _normalize

# min_days_ahead=1：不能约当天的；release：每天几点放新一天的票（北京时间）；url：核对过能打开的官方预约网站
ATTRACTIONS = [
    {"zh": "故宫博物院", "en": "Forbidden City (Palace Museum)", "city": "北京",
     "aliases": ["forbidden city", "palace museum", "gugong"], "zh_names": ["故宫"],
     "days_ahead": 7, "release": "20:00",
     "url": "https://bookingticket.dpm.org.cn/", "wechat": "故宫博物院",
     "note_en": "The museum has not authorised any third-party ticket sellers. Book only through the official site or mini-program.",
     "note_zh": "故宫博物院未授权任何第三方代理门票，请只通过官方网站或小程序预约。"},
    {"zh": "天安门广场", "en": "Tiananmen Square", "city": "北京",
     "aliases": ["tiananmen square", "tiananmen"], "zh_names": ["天安门"],
     "days_ahead": 7, "min_days_ahead": 1, "wechat": "天安门广场预约参观",
     "note_en": "Visitors without a reservation cannot enter the square.",
     "note_zh": "未预约的游客无法进入广场参观。"},
    {"zh": "中国国家博物馆", "en": "National Museum of China", "city": "北京",
     "aliases": ["national museum of china", "national museum"], "zh_names": ["国家博物馆", "国博"],
     "days_ahead": 7, "release": "17:00", "url": "https://www.chnmuseum.cn/", "wechat": "中国国家博物馆"},
    {"zh": "毛主席纪念堂", "en": "Chairman Mao Memorial Hall", "city": "北京",
     "aliases": ["chairman mao memorial hall", "mao memorial hall", "mao mausoleum", "mao's mausoleum"],
     "days_ahead": 6, "min_days_ahead": 1, "release": "12:30", "wechat": "毛主席纪念堂"},
    {"zh": "天坛公园", "en": "Temple of Heaven", "city": "北京",
     "aliases": ["temple of heaven", "tiantan"], "zh_names": ["天坛"],
     "days_ahead": 7, "wechat": "天坛公园"},
    {"zh": "颐和园", "en": "Summer Palace", "city": "北京",
     "aliases": ["summer palace", "yiheyuan"],
     "days_ahead": 7, "wechat": "颐和园"},
    {"zh": "八达岭长城", "en": "Badaling Great Wall", "city": "北京",
     "aliases": ["badaling", "badaling great wall"], "zh_names": ["八达岭"],
     "days_ahead": 10, "wechat": "八达岭长城"},
    {"zh": "上海博物馆", "en": "Shanghai Museum", "city": "上海",
     "aliases": ["shanghai museum"], "zh_names": ["上博"],
     "url": "https://www.shanghaimuseum.net/mu/frontend/pg/en/index", "wechat": "上海博物馆",
     "note_en": "Overseas visitors register on the official website with an email address; the entry QR code is emailed to you.",
     "note_zh": "境外游客用邮箱在官网注册预约，入场二维码会发到邮箱。"},
    {"zh": "陕西历史博物馆", "en": "Shaanxi History Museum", "city": "西安",
     "aliases": ["shaanxi history museum"], "zh_names": ["陕历博"],
     "url": "https://www.sxhm.com/guide.html", "wechat": "陕西历史博物馆"},
    {"zh": "秦始皇帝陵博物院", "en": "Terracotta Warriors (Emperor Qinshihuang's Mausoleum Site Museum)", "city": "西安",
     "aliases": ["terracotta warriors", "terracotta army", "terracotta warrior", "bingmayong"], "zh_names": ["兵马俑"],
     "days_ahead": 7, "url": "https://www.bmy.com.cn/", "wechat": "秦始皇帝陵博物院"},
]

# 2026-10-05 合并核验：其余景点保留渠道和原数据，时刻/预售天数待官方复核。
_VERIFIED = {
    "故宫博物院": {"policy_verified": True, "min_days_ahead": 1,
        "entry_note_zh": "故宫不售当日票，请通过官方渠道提前实名预约。",
        "entry_note_en": "The Palace Museum does not sell same-day tickets. Make an advance reservation through official channels."},
    "天安门广场": {"policy_verified": True,
        "url": "https://yuyue2026.tamgw.beijing.gov.cn/web/",
        "entry_note_zh": "广场免费参观，个人和旅游团须提前1—7日实名预约。",
        "entry_note_en": "The square is free to visit; individuals and groups must make a real-name reservation 1–7 days ahead."},
    "中国国家博物馆": {"policy_verified": True,
        "entry_note_zh": "参观需按官方渠道实名预约，并按预约时段入馆。",
        "entry_note_en": "Make a real-name reservation through official channels and enter during your reserved time slot."},
    "上海博物馆": {"url": "https://www.shanghaimuseum.cn/mu/frontend/pg/en/index",
        "entry_note_zh": "东馆散客常设展免预约；互动空间和特展、人民广场馆的规则请分别查看官网。",
        "entry_note_en": "Individual visitors to the East Museum do not need a reservation for regular galleries. Check the official site for interactive spaces, special exhibitions and the People's Square Museum.",
        "note_zh": "请先确认馆址和展览，不要把不同馆址的规则混用。",
        "note_en": "Confirm the museum location and exhibition before planning your visit."},
}
for _attraction in ATTRACTIONS:
    _attraction.update(_VERIFIED.get(_attraction["zh"], {}))


def _build_matchers():
    matchers = []
    for a in ATTRACTIONS:
        patterns = [_alias_pattern(x) for x in a["aliases"]]
        patterns += [re.compile(re.escape(z)) for z in [a["zh"], *a.get("zh_names", [])]]
        matchers.append((a, patterns))
    return matchers


_MATCHERS = _build_matchers()


def detect_attraction(text: str) -> Optional[dict]:
    """返回文本中首个命中的已收录景点。"""
    t = _normalize(text)
    found = [(min(m.start() for p in pats for m in [p.search(t)] if m), a)
             for a, pats in _MATCHERS if any(p.search(t) for p in pats)]
    return min(found, key=lambda x: x[0])[1] if found else None


# 用户是要买票/进去参观（而不是问怎么走、问开放时间）：这时才出“去网上预约”卡片
_TICKET_INTENT = re.compile(
    r"门票|买票|购票|订票|票|预约|参观|进去|入场|入园|游览"
    r"|\b(?:tickets?|book(?:ing)?|reserv(?:e|ation)|admission|entry|entrance|visit(?:ing)?|get in|go in)\b",
    re.IGNORECASE,
)


def wants_ticket(text: str) -> bool:
    return bool(_TICKET_INTENT.search(text or ""))


def booking_info(a: dict) -> dict:
    """前端渲染“提前网上预约”卡片 / 路线卡片提醒所需的全部信息（中英双语，代码生成，不经过模型）。"""
    facts_en = [a.get("entry_note_en") or "Check official channels for ticketing and reservation requirements before visiting."]
    facts_zh = [a.get("entry_note_zh") or "参观前请通过官方渠道查看购票和预约要求。"]
    if a.get("policy_verified") and a.get("days_ahead"):
        when_en = f"Book up to {a['days_ahead']} days ahead"
        when_zh = f"最多可提前 {a['days_ahead']} 天预约"
        if a.get("release"):
            when_en += f"; new tickets are released daily at {a['release']} Beijing time"
            when_zh += f"，每天北京时间 {a['release']} 放票"
        if a.get("min_days_ahead"):
            when_en += f". Same-day booking is not possible"
            when_zh += "，不能预约当天"
        facts_en.append(when_en + ". Popular dates sell out fast, so book as early as you can.")
        facts_zh.append(when_zh + "。热门日期很快约满，请尽早预约。")
    facts_en.append("Bring your original passport for identity checks; follow the official channel’s document requirements.")
    facts_zh.append("请携带护照原件，并按官方渠道要求填写和出示身份证件。")
    if a.get("note_en"):
        facts_en.append(a["note_en"])
        facts_zh.append(a["note_zh"])
    return {
        "name_zh": a["zh"], "name_en": a["en"], "url": a.get("url"), "wechat": a.get("wechat"),
        "facts_en": facts_en + ["Current rules and availability are shown on the official channel."], "facts_zh": facts_zh + ["当前规则和余票以官方渠道为准。"],
    }


def booking_answer(a: dict) -> dict:
    """“我要一张故宫的票” → 不出购票卡，直接给“提前网上预约”卡片（intent_type=ticket，action=book_online）。"""
    info = booking_info(a)
    return {
        "status": "success",
        "intent_type": "ticket",
        "question": "",
        "quick_replies": [],
        "card_content": {"zh": a["zh"] + "：" + info["facts_zh"][0], "en": a["en"] + ": " + info["facts_en"][0]},
        "action": "book_online",
        "map_query": "",
        "next_action": {"type": "book_online", "query": None, "origin": None, "city": None, "instruction": {"zh": "查看官方渠道，确认适用的参观和预约规则。", "en": "Check official channels for the rules that apply to your visit."},
                        "booking": info},
    }
