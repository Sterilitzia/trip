"""菜单识别分为图片转写、菜名翻译和价格单位处理；结果逐项推送给前端。"""
import base64
import io
import json
import re
from typing import Iterator, Optional

from PIL import Image, ImageOps

# 按配置顺序尝试视觉模型；:fast 表示请求关闭思考模式。
MENU_OCR_MODELS_DEFAULT = ("glm-4.5v:fast", "glm-4.6v-flash:fast", "glm-4.1v-thinking-flash")

# 送给视觉模型的图片最长边：太小读错字/编菜，太大没有收益
OCR_IMAGE_SIZE = 2048
# 宽高比 ≥ 这个值的横幅照片（一整面墙的菜单板、多栏价目表）竖切成 STRIP_COUNT 条分别读：
# 每条的字更大，模型也不会只读中间一块。竖排的单张菜单（一行一个菜，菜名和价格隔得很远）不切，切了每行都会断开
LANDSCAPE_RATIO = 1.2
STRIP_COUNT = 3
STRIP_OVERLAP = 0.2  # 相邻两条重叠 20%：被一条切断的那一栏，在另一条里是完整的
# 切条读出来的菜少于这个数 → 多半是一整张横向表格被切断了，整张再读一遍
MIN_ITEMS_FROM_STRIPS = 8

OCR_PROMPT = """你是菜单转写员。照片里可能有好几块菜单板/好几栏，按从左到右、从上到下逐块逐栏转写，每一块都不能漏。
规则：
1. 只转写菜品（能点的东西），不要转写分类标题（如“汤类”“炒粉类”）、广告语、店名、营业执照、加料说明。
2. 菜名逐字照抄照片上的字，一个字都不能改：即使写法不常见、像错别字或是方言叫法，也要照抄，绝不能替换成你觉得更常见的写法（例：照片写“云香面”就写“云香面”，不能改成“云吞面”；写“蕃茄”就写“蕃茄”）。看不清的字用“?”代替，不要猜。
3. 竖排、字间有大空格的菜名要拼成一个词（“清 汤 面”→“清汤面”）。
4. 价格照抄原文，含单位和规格（如“一两/3元 二两/4元”“38元/位 88元/例”“10-12-15元”）；有划掉的原价和特价时写“原价138元/位 特价98元/位”。
5. 照片不是菜单/价目表（没有能点的菜）就什么都不输出。{strip_note}
输出格式：每行一个菜，每行是一个完整的 JSON 对象，不要 Markdown，不要数组，不要别的文字：
{"zh": "菜名", "price": "价格原文"}"""

TRANSLATE_PROMPT = """你是中国菜单的英文翻译，读者是第一次来中国的外国游客。用户给你一个编了号的菜名列表，照片转写来的，可能有个别错字，“?”表示看不清的字。
为每个菜给出：
- en：英文菜名，表达这道菜实际是什么，不要逐字直译（“拆鱼羹”是 Shunde Shredded Fish Soup，不是 dismantled fish；“鱼香肉丝”是 Fish-Fragrant Shredded Pork，菜里没有鱼）。有通行英文名的用通行名（回锅肉 Twice-Cooked Pork，麻婆豆腐 Mapo Tofu，宫保鸡丁 Kung Pao Chicken，扬州炒饭 Yangzhou Fried Rice）。地名照拼音写（顺德 Shunde，大良 Daliang，均安 Jun'an，邛崃 Qionglai）。
- desc：一句很短的英文说明，帮游客决定点不点：主要食材、做法、口味；辣的写 spicy；含内脏、猪肉、牛肉等饮食禁忌相关的要写出来（如 "contains pork intestines"）。不确定就写空字符串，绝不编。

地方叫法和容易译错的词（按真实含义翻译）：
- 云吞、云香（潮汕/深圳）、抄手（四川）、扁肉（福建/沙县）、馄饨 都是 wonton；铰子/饺子 = dumplings
- 粉 = rice noodles；河粉 = flat rice noodles；米粉 = thin rice noodles；米线 = rice vermicelli；面 = wheat noodles；公仔面 = instant noodles；油面 = yellow egg noodles；粉丝 = glass noodles；只有菜名里真的写着“粉/面”或“面/粉”（带斜杠）才是米粉和面条二选一，en 里写 Rice Noodles or Wheat Noodles（如 鱼香肉丝面/粉 = Fish-Fragrant Pork with Rice Noodles or Wheat Noodles），绝不能写成 Rice or Noodles（外国人会以为是米饭）；单独的“面”就是 wheat noodles，不是二选一（云香面 = Wonton Noodles）
- 盖浇 / 盖浇饭 = rice topped with ...（牛肉盖浇 = Rice Topped with Beef，不是 Beef Rice Topped）；汤饭 = rice in soup；净 = plain (on its own)；绍子 = minced pork topping
- 炆 = braised；煲 = clay pot；焗 = baked；炝 = flash-fried with dried chili；干煸 = dry-fried；红烧 = red-braised (soy)；炖 = slow-stewed；扣 = braised and plated (dome)；酿 = stuffed
- 翅（粤菜宴席、和鲍鱼花胶燕窝同列时）= shark fin；鸡翅 = chicken wings；花胶 = fish maw；鱼唇 = fish lips；海参/辽参 = sea cucumber；燕窝 = bird's nest；8头/6头鲍 = abalone (8 / 6 per catty, i.e. size)
- 走地鸡 = free-range chicken（海参炆走地鸡 = Free-Range Chicken Braised with Sea Cucumber）；乳鸽 = squab (young pigeon)；鲮鱼 = mud carp；蚬 = clams；猪横脷 = pig spleen；火腩 = roast pork belly；柚皮 = pomelo peel；香水莲 = water lily (flower)
- 猪杂/鸡杂 = pork/chicken offal；肥肠 = pork intestines；牛腩 = beef brisket；筒骨 = pork leg bone；蹄花/猪蹄 = pig trotter；雪豆 = white kidney beans
- 钵钵鸡 = Bobo chicken (cold chicken skewers in chili oil)；虎皮海椒 = blistered green chilies；凤尾 = romaine lettuce tips；莲白 = cabbage；油菜尖 = rapeseed greens；盐煎肉 = salt-fried pork slices；烂肉粉丝 = glass noodles with minced pork
- 素椒 = with chili sauce, no meat；红汤 = spicy chili-oil broth；清汤 = clear broth；奶汤 = milky bone broth；烫蛋 = with a poached egg；三鲜 = three-delicacy (pork, shrimp, mushroom etc.)；大良炒牛奶 = Daliang stir-fried milk (silky egg-white custard)；脆皮炸牛奶 = deep-fried milk custard

输入是编了号的菜名列表。输出格式：每个菜一行，按输入顺序，格式为“编号|en|desc”，不要输出菜名本身，不要 JSON，不要 Markdown，不要别的文字。例如：
1|Shunde Shredded Fish Soup|Fish soup with shredded fish
2|Clay Pot Braised Shark Fin|Shark fin braised in soy sauce"""

TRANSLATE_BATCH = 20  # 边转写边翻译：攒够 20 个菜就送去翻译，游客不用等整张菜单读完才看到英文
TRANSLATE_FIRST_BATCH = 8  # 第一批少一点：读出前几个菜就开始翻译，英文尽早出现
MENU_TRANSLATE_MODEL_DEFAULT = "glm-4.5-air"
# 限制翻译并发，减少上游限流。
TRANSLATE_WORKERS = 2

_HAN = re.compile(r"[\u4e00-\u9fff]")
_BOX_TAGS = re.compile(r"<\|/?(?:begin|end)_of_box\|>")


# ============================================================
# 第一步：转写（视觉模型，流式，一行一个菜）
# ============================================================
def parse_ocr_line(line: str) -> Optional[dict]:
    """解析视觉模型输出的一行 {"zh": ..., "price": ...}；不是菜（没有汉字/解析失败）返回 None。"""
    line = _BOX_TAGS.sub("", line).strip().rstrip(",")
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    zh = re.sub(r"\s+", "", str(obj.get("zh") or ""))
    if not _HAN.search(zh):
        return None
    price = re.sub(r"\s+", " ", str(obj.get("price") or "")).strip()
    return {"zh": zh, "price": price, "unclear": "?" in zh or "？" in zh}


STRIP_NOTE = "\n6. 这张图是从一整张菜单照片里竖着裁下来的一条。被图片左右边缘切断、显示不完整的菜名或价格不要输出（相邻的另一条里有完整的）。"


class MenuReadError(Exception):
    """所有转写模型都失败（高峰期限流/超时），且一个菜都没读出来。"""


def decode_image(image_url: str) -> Optional[Image.Image]:
    """前端传来的 data URL → 图片；解析失败返回 None（调用方退回整张原图直接交给模型）。"""
    try:
        data = image_url.split(",", 1)[1] if image_url.startswith("data:") else image_url
        img = Image.open(io.BytesIO(base64.b64decode(data)))
        if img.width * img.height > 24_000_000:
            return None
        img.load()
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return None


def encode_image(img: Image.Image) -> str:
    """按 OCR_IMAGE_SIZE 缩放图片，小图最多放大四倍。"""
    scale = min(4.0, OCR_IMAGE_SIZE / max(img.size))
    if abs(scale - 1) > 0.01:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def menu_image_parts(img: Image.Image) -> list[Image.Image]:
    """横幅照片竖切成重叠的几条，其余整张。"""
    if img.width / img.height < LANDSCAPE_RATIO:
        return [img]
    n = STRIP_COUNT
    width = (1 + STRIP_OVERLAP * (n - 1)) / n  # 每条占整张宽度的比例
    step = (1 - width) / (n - 1)
    return [img.crop((round(i * step * img.width), 0, round((i * step + width) * img.width), img.height)) for i in range(n)]


def _ocr_stream(client, model_spec: str, image_url: str, prompt: str):
    """model_spec 形如 “glm-4.5v” 或 “glm-4.5v:fast”；:fast 表示关闭思考。"""
    model, _, mode = model_spec.partition(":")
    extra = {"extra_body": {"thinking": {"type": "disabled"}}} if mode == "fast" else {}
    return client.with_options(timeout=180, max_retries=1).chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=16000,  # 沙县小吃这类 200 多个菜的大菜单板，输出很长
        stream=True,
        **extra,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
    )


def _read_part(client, models: tuple[str, ...], image_url: str, prompt: str) -> Iterator[dict]:
    """读一张图（或一条）。按顺序尝试 models：只有在这张图一个菜都还没读出来时才换下一个模型
    （中途失败就保留已读出的菜，不混用两个模型的结果）；全部失败抛 MenuReadError。"""
    last_error = None
    for model in models:
        produced = False
        buf = ""
        stream = None
        try:
            stream = _ocr_stream(client, model, image_url, prompt)
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "reasoning_content", None):
                    yield {"thinking": True}
                buf += delta.content or ""
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    item = parse_ocr_line(line)
                    if item:
                        produced = True
                        yield item
            item = parse_ocr_line(buf)
            if item:
                yield item
            return
        except Exception as e:
            print(f"[菜单转写失败] {model}: {e}")
            last_error = e
            if produced:
                return
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
    raise MenuReadError(str(last_error))


def ocr_menu_items(client, models: tuple[str, ...], image_url: str) -> Iterator[dict]:
    """流式转写：横幅照片切条逐条读，每读出一个新菜就 yield（按菜名去重，重叠区域会读到两次）。
    还会 yield {"thinking": True}（备选模型思考中的心跳）和 {"part": i, "parts": n}（开始读第 i 条）。
    第一条就全部失败时抛 MenuReadError；后面某条失败则保留已读出的菜，停止读取并抛出，由调用方提示“可能漏了菜”。"""
    img = decode_image(image_url)
    parts = menu_image_parts(img) if img else []
    urls = [encode_image(p) for p in parts] or [image_url]
    seen: set[str] = set()

    def read(url: str, prompt: str) -> Iterator[dict]:
        for ev in _read_part(client, models, url, prompt):
            if "zh" in ev:
                if ev["zh"] in seen:
                    continue
                seen.add(ev["zh"])
            yield ev

    whole_prompt = OCR_PROMPT.replace("{strip_note}", "")
    if len(urls) == 1:
        yield from read(urls[0], whole_prompt)
        return
    strip_prompt = OCR_PROMPT.replace("{strip_note}", STRIP_NOTE)
    for i, url in enumerate(urls, start=1):
        yield {"part": i, "parts": len(urls)}
        yield from read(url, strip_prompt)
    if len(seen) < MIN_ITEMS_FROM_STRIPS:
        # 切条几乎没读出菜：多半是一整张横向表格（菜名在左、价格在右）被切断了每一行 → 整张重读
        yield {"part": len(urls) + 1, "parts": len(urls) + 1}
        yield from read(encode_image(img), whole_prompt)


# ============================================================
# 第二步：翻译（文本模型，分批）
# ============================================================
# 菜名 → 译文的进程内缓存：同一张菜单再拍一次、不同菜单里的常见菜（蛋炒饭、云吞面）都不用再调模型
_TRANSLATION_CACHE: dict[str, dict] = {}


def _parse_translation_line(line: str, names: list[str]) -> Optional[tuple[str, dict]]:
    """“3|Wonton Noodles|Wontons with noodles” → (第 3 个菜名, {en, desc})；格式不对或编号越界返回 None。"""
    parts = [p.strip() for p in line.strip().split("|")]
    if len(parts) < 2 or not parts[0].rstrip(".").isdigit():
        return None
    idx = int(parts[0].rstrip(".")) - 1
    en, desc = parts[1], "|".join(parts[2:]).strip()
    if not (0 <= idx < len(names)) or not en or _HAN.search(en):
        return None
    return names[idx], {"en": en, "desc": desc}


def _translate_stream(client, model: str, names: list[str]) -> Iterator[tuple[str, dict]]:
    """逐行解析译文，返回 (菜名, {en, desc})。"""
    # max_retries：并发时偶尔会撞上限流（429），SDK 会退避后重试
    stream = client.with_options(timeout=120, max_retries=3).chat.completions.create(
        model=model,
        temperature=0.1,
        stream=True,
        extra_body={"thinking": {"type": "disabled"}},
        messages=[
            {"role": "system", "content": TRANSLATE_PROMPT},
            {"role": "user", "content": "\n".join(f"{i}. {n}" for i, n in enumerate(names, 1))},
        ],
    )
    try:
        buf = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            buf += chunk.choices[0].delta.content or ""
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                parsed = _parse_translation_line(line, names)
                if parsed:
                    yield parsed
        parsed = _parse_translation_line(buf, names)
        if parsed:
            yield parsed
    finally:
        close = getattr(stream, "close", None)
        if close:
            close()


def translate_menu_items(client, model: str, names: list[str]) -> Iterator[tuple[str, dict]]:
    """翻译一批菜名，逐个 yield (菜名, {en, desc})：先给缓存里有的，其余流式翻译；模型漏掉的菜再补翻一次。"""
    todo = []
    for name in names:
        if name in _TRANSLATION_CACHE:
            yield name, _TRANSLATION_CACHE[name]
        else:
            todo.append(name)
    for attempt in range(2):
        if not todo:
            return
        done = set()
        for name, tr in _translate_stream(client, model, todo):
            if name not in done:
                done.add(name)
                if len(_TRANSLATION_CACHE) > 2000:
                    _TRANSLATION_CACHE.clear()
                _TRANSLATION_CACHE[name] = tr
                yield name, tr
        todo = [n for n in todo if n not in done]


# ============================================================
# 第三步：价格单位（代码确定性换算）
# ============================================================
_CN_DIGITS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "半": 0.5}
_UNIT_EN = {
    "位": "per person", "人": "per person", "例": "per dish", "份": "per portion", "条": "per whole fish",
    "只": "each", "个": "each", "碗": "per bowl", "杯": "per cup", "串": "per skewer", "斤": "per jin (500 g)",
    "两": "per liang (50 g)", "盘": "per plate", "锅": "per pot", "打": "per dozen", "瓶": "per bottle", "笼": "per steamer basket",
}
_SIZE_EN = {"小份": "small", "中份": "medium", "大份": "large", "小": "small", "中": "medium", "大": "large", "例": "regular"}
_NUM = r"\d+(?:\.\d+)?"


def _money(n: str) -> str:
    return f"¥{n}"


def _liang_label(word: str) -> str:
    """一两 → 1 liang ≈ 50 g（面条、抄手按生重计价）。"""
    n = _CN_DIGITS.get(word[:-1], None)
    if n is None:
        return word
    grams = int(n * 50)
    n_str = f"{n:g}"
    return f"{n_str} liang ≈ {grams} g"


def _normalize_price(price: str) -> str:
    p = (price or "").strip().replace("：", ":").replace("／", "/").replace("－", "-").replace("—", "-")
    # 转写出来的价格常带多余空格（“38元 / 位 88元/例”），斜杠和横杠两边的空格去掉，空格只用来分隔不同规格
    return re.sub(r"\s*([/-])\s*", r"\1", p)


def format_price_en(price: str) -> str:
    """把菜单价格原文换成外国人看得懂的英文，单位必须写清楚。看不懂的格式保留原文（只把“元”换成 ¥）。"""
    p = _normalize_price(price)
    if not p:
        return ""
    if "?" in p or "？" in p:
        return "price unclear, ask staff"

    # 原价/特价：只报特价，附原价
    m = re.search(rf"原价:?\s*({_NUM})\s*元?(?:/(\S+?))?\s+特价:?\s*({_NUM})\s*元?(?:/(\S+))?", p)
    if m:
        unit = m.group(4) or m.group(2)
        unit_en = f" {_unit_phrase(unit)}" if unit else ""
        return f"{_money(m.group(3))}{unit_en} (special price, was {_money(m.group(1))})"

    # 大中小份价格档：10-12-15元
    m = re.fullmatch(rf"({_NUM}(?:\s*-\s*{_NUM}){{1,3}})\s*元?", p)
    if m:
        prices = [x.strip() for x in m.group(1).split("-")]
        labels = {2: "small / large", 3: "small / medium / large", 4: "4 sizes, small → extra large"}[len(prices)]
        return " / ".join(_money(x) for x in prices) + f" ({labels})"

    parts = []
    for seg in re.split(r"\s+", p):
        parts.append(_segment_en(seg))
    return " · ".join(x for x in parts if x)


def _unit_phrase(unit: str) -> str:
    """“位” → per person；“1只” → for 1；“2只” → for 2。"""
    m = re.fullmatch(rf"({_NUM})(\S)", unit)
    if m:
        return f"for {m.group(1)}"
    return _UNIT_EN.get(unit, f"/ {unit}")


def _segment_en(seg: str) -> str:
    # 一两/3元、二两4元、3元/一两
    m = re.fullmatch(rf"([一二两三四五半]两)/?({_NUM})元?", seg) or None
    if m:
        return f"{_money(m.group(2))} ({_liang_label(m.group(1))})"
    m = re.fullmatch(rf"({_NUM})元?/([一二两三四五半]两)", seg)
    if m:
        return f"{_money(m.group(1))} ({_liang_label(m.group(2))})"
    # 小份5元、大份10元
    m = re.fullmatch(rf"(小份|中份|大份|小|中|大)/?({_NUM})元?", seg)
    if m:
        return f"{_money(m.group(2))} {_SIZE_EN[m.group(1)]}"
    # 38元/位、48元/1只、8元
    m = re.fullmatch(rf"({_NUM})元?(?:/(\S+))?", seg)
    if m:
        unit = m.group(2)
        return _money(m.group(1)) + (f" {_unit_phrase(unit)}" if unit else "")
    return seg.replace("元", "").strip() and re.sub(rf"({_NUM})元", r"¥\1", seg)



# ---- 份量/规格选项：点菜时让游客选，点单卡上写清楚（不然店员还得问“要大份还是小份”“要几两”）----
_TIER_LABELS = {
    2: [("小份", "Small"), ("大份", "Large")],
    3: [("小份", "Small"), ("中份", "Medium"), ("大份", "Large")],
    4: [("小份", "Small"), ("中份", "Medium"), ("大份", "Large"), ("特大份", "Extra large")],
}
_UNIT_ZH = {"位": "每位", "人": "每位", "例": "例牌", "份": "一份", "条": "整条", "只": "一只", "个": "一个", "碗": "一碗",
            "杯": "一杯", "串": "一串", "斤": "一斤", "两": "一两", "盘": "一盘", "锅": "一锅", "打": "一打", "瓶": "一瓶", "笼": "一笼"}


def price_options(price: str) -> list[dict]:
    """一道菜有几种规格可选时返回 [{"zh": 写在点单卡上的规格, "en": 给游客看的选项}]；只有一个价格时返回空列表。
    10-12-15元 → 小份/中份/大份；一两/3元 二两/4元 → 一两/二两；38元/位 88元/例 → 每位/例牌；48元/1只 88元/2只 → 1只/2只。"""
    p = _normalize_price(price)
    if not p or "?" in p or "？" in p or "原价" in p:
        return []
    m = re.fullmatch(rf"({_NUM}(?:-{_NUM}){{1,3}})元?", p)
    if m:
        prices = m.group(1).split("-")
        return [{"zh": zh, "en": f"{en} · {_money(x)}"} for (zh, en), x in zip(_TIER_LABELS[len(prices)], prices)]
    options = []
    for seg in p.split():
        m = re.fullmatch(rf"([一二两三四五半]两)/?({_NUM})元?", seg) or re.fullmatch(rf"({_NUM})元?/([一二两三四五半]两)", seg)
        if m:
            liang, money = (m.group(1), m.group(2)) if m.group(1).endswith("两") else (m.group(2), m.group(1))
            label = _liang_label(liang).replace(" ≈ ", " (≈ ") + ")"
            options.append({"zh": liang, "en": f"{label} · {_money(money)}"})
            continue
        m = re.fullmatch(rf"(小份|中份|大份|小|中|大)/?({_NUM})元?", seg)
        if m:
            size = m.group(1) if m.group(1).endswith("份") else m.group(1) + "份"
            options.append({"zh": size, "en": f"{_SIZE_EN[m.group(1)].capitalize()} · {_money(m.group(2))}"})
            continue
        m = re.fullmatch(rf"({_NUM})元?/(\S+)", seg)
        if m:
            unit = m.group(2)
            zh = unit if re.match(rf"{_NUM}", unit) else _UNIT_ZH.get(unit, unit)
            options.append({"zh": zh, "en": f"{_unit_phrase(unit).capitalize()} · {_money(m.group(1))}"})
            continue
        return []  # 有看不懂的规格：不给选项，免得点单卡上写错
    return options if len(options) >= 2 else []
