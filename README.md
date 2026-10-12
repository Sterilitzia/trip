# 中国旅行搭子 · China Companion
---

## 项目简介

**中国旅行搭子** 是一个专为境外游客设计的 AI 旅行助手 Web 应用。你只需像和朋友聊天一样说出需求——查路线、找餐厅、翻译菜单、问景点预约——系统就能自动识别意图，调用高德地图、智谱大模型等能力，给出可直接执行的结果卡片。

**本项目是 Trip.com 2025 Hackathon 参赛项目。**

---

## 在线体验

| 环境 | 地址 | 状态 |
|------|------|------|
| Render 部署 | `https://china-travel-companion.onrender.com` | 🟢 Live |

> 免费实例闲置 15 分钟后会休眠，首次访问可能需要约 1 分钟唤醒。

---

## 功能一览

| 功能 | 说明 | 示例 |
|------|------|------|
| ** 智能路线规划** | 公交 / 地铁 / 步行换乘方案，自动选择最短耗时路线 | "从外滩到东方明珠怎么走" |
| ** 附近搜索** | 基于高德地图 POI，查找最近的商场、药店、便利店等 | "最近的药店在哪里" |
| ** 菜单拍照翻译** | 上传菜单照片，AI 逐条转写并翻译成英文，附带食材与口味说明 | 拍照中文菜单 → 英文菜名 |
| ** 景点预约提醒** | 针对故宫、天安门、兵马俑等需提前预约的景点，直接给出官方预约渠道 | "我想去故宫" |
| ** 自然语言对话** | 支持多轮上下文，日期、时间、品牌门店等直接作答 | "今天几号" / "一点点在哪" |
| ** 图片理解** | 上传照片即可让 AI 识别场景、物品或文字信息 | 上传地铁线路图提问 |
| ** 中英双语** | 界面与回复根据用户语言偏好自动切换中英文 | English / 中文 |

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (Frontend)                       │
│              SPA (静态构建) → FastAPI StaticFiles            │
├─────────────────────────────────────────────────────────────┤
│                        后端 (Backend)                        │
│   FastAPI  +  智谱 AI (GLM-4)  +  高德地图 REST API         │
│   · 意图解析 / 多轮对话                                      │
│   · 公交/地铁/步行 路线规划                                  │
│   · 菜单 OCR + 翻译 (SSE 流式推送)                           │
│   · 附近 POI 搜索 + 英文名称生成                             │
│   · 景点预约规则引擎（免模型直接输出）                        │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | 静态 SPA (JS + CSS)，响应式设计 |
| 后端 | Python 3.11, FastAPI, Uvicorn |
| AI 模型 | 智谱 GLM-4 / GLM-4V (兼容 OpenAI SDK) |
| 地图服务 | 高德地图 REST API (路径规划、POI 搜索、逆地理编码) |
| 部署 | Render (Web Service, Free Tier) |
| 其他 | Pillow (图像处理), PyPinyin (拼音生成), python-dotenv |

---

## 项目结构

```
.
├── app.py                 # Render 入口：挂载 FastAPI + 静态文件
├── render.yaml            # Render Blueprint 配置（一键部署）
├── requirements.txt       # Python 依赖
├── backend/
│   ├── main.py            # FastAPI 主应用 & 业务逻辑
│   ├── attractions.py     # 景点预约数据与规则
│   ├── menu.py            # 菜单 OCR + 翻译引擎
│   └── brands.py          # 品牌门店识别与别名匹配
├── static/                # 前端构建产物 & 图片资源
│   ├── index.html         # 入口 HTML
│   ├── assets/            # JS/CSS bundle
│   ├── images/            # 景点/美食配图
│   └── photo-credits.json # 图片来源与授权信息
└── README.md
```

---

## 快速开始

### 方式一：Render 一键部署（推荐）

1. Fork 本仓库到自己的 GitHub 账号
2. 登录 [Render](https://render.com) → 点击 **New → Blueprint**
3. 选择你 Fork 的仓库，`render.yaml` 会自动完成服务配置
4. 在 Environment 中填入以下密钥（见「环境变量」章节）
5. 等待部署完成，访问 Render 提供的域名即可

### 方式二：手动创建 Web Service

| 配置项 | 值 |
|--------|-----|
| Language | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/api/health` |

---

## 环境变量

> **密钥仅填写在 Render Environment 页面，请勿提交到 GitHub！**

| 变量名 | 必填 | 说明 |
|--------|------|------|
| `LLM_API_KEY` | ✅ | 智谱 AI API Key ([获取](https://open.bigmodel.cn/)) |
| `AMAP_KEY` | ✅ | 高德地图 Web Service Key ([获取](https://lbs.amap.com/)) |
| `LLM_BASE_URL` | 可选 | 模型服务地址，默认 `https://open.bigmodel.cn/api/paas/v4/` |
| `LLM_MODEL` | 可选 | 对话模型，默认 `glm-4.5-flash` |
| `MENU_OCR_MODELS` | 可选 | 菜单识别模型列表，默认 `glm-4.5v:fast,glm-4.6v-flash:fast,glm-4.1v-thinking-flash` |
| `MENU_TRANSLATE_MODEL` | 可选 | 菜单翻译模型，默认 `glm-4.5-air` |

---

## 🔌 API 接口概览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/time` | 获取北京时间 |
| `POST` | `/api/parse_intent` | 意图解析（含直接回答 / LLM 对话） |
| `POST` | `/api/parse_intent/stream` | 流式意图解析（SSE） |
| `POST` | `/api/directions` | 路线规划（公交/地铁/步行） |
| `POST` | `/api/nearby` | 附近 POI 搜索 |
| `POST` | `/api/menu` | 菜单拍照翻译（SSE 流式） |
| `POST` | `/api/translate_names` | 批量中文地名 → 英文拼音 |

---

## 📝 开源与授权

- 本项目代码以 MIT 协议开源。
- 图片素材来源及授权信息详见 [`static/photo-credits.json`](static/photo-credits.json)，部分图片来自 Wikimedia Commons 与 Unsplash。
- 地图数据来源于 [高德地图开放平台](https://lbs.amap.com/)。
- AI 能力由 [智谱 AI](https://open.bigmodel.cn/) 提供。

---

## 关于本项目

本项目诞生于 **Trip.com 2025 Hackathon**，旨在通过 AI 降低外国游客在中国旅行的信息门槛——语言不通、菜单看不懂、景点不知道怎么预约、公交看不懂……我们希望让这些问题变成「说一句就行」的简单事。

如果你有任何建议或发现问题，欢迎提交 Issue。

---

> Made with ❤️ for travelers in China.
