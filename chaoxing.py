import base64
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import uuid
from functools import partial
from urllib.parse import parse_qsl, quote, urljoin, urlparse

# print 立即刷新：管道/沙箱下 stdout 默认块缓冲会攒着一次性输出，重定义后任何环境都逐行实时
print = partial(print, flush=True)

# 脚本所在目录：fonts/cxsecret_map.json 都相对它定位，
# 这样无论从哪个工作目录（命令行/PyCharm 运行配置）启动都不会找不到文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from bs4 import BeautifulSoup
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


class ChaoXing:
    # 视频播放器页面地址，心跳上报需要带上这个 Referer
    VIDEO_REFERER = 'https://mooc1.chaoxing.com/ananas/modules/video/index.html?v=2026-0828-1810'

    # cxsecret 人工核对的兜底映射（优先级最高，覆盖自动识别结果）
    CXSECRET_MAP = {
        '懙': '效', '懚': '被', '懛': '预', '懜': '有', '懝': '害', '懞': '防',
        '懠': '学', '懡': '中', '懢': '杀', '懤': '是', '懥': '国', '懧': '给',
        '懩': '尾', '懪': '己', '懫': '对', '懬': '来', '懭': '路', '穤': '谋',
    }
    # 字形比对基准字体：思源黑体 CN Normal（cxsecret 源字体即它改的 cmap）
    CXSECRET_BASE_FONT = os.path.join(BASE_DIR, 'fonts', 'SourceHanSansCN-Normal.otf')
    CXSECRET_BASE_FONT_URL = ('https://cdn.jsdelivr.net/gh/adobe-fonts/source-han-sans@release/'
                              'SubsetOTF/CN/SourceHanSansCN-Normal.otf')
    # 密文映射全局缓存文件（与脚本同目录）：{"密字": "真字", ...}，全部字体共用一张表
    CXSECRET_CACHE = os.path.join(BASE_DIR, 'cxsecret_map.json')
    # 9010 风控验证码识别引擎（懒加载，进程内复用）
    __ocr = None

    # Agnes AI（答题用），OpenAI 兼容接口
    LLM_API_URL = 'https://apihub.agnes-ai.com/v1/chat/completions'
    LLM_API_KEY = 'sk-kvOgUeRuTbuX4dkjG7Whe1s5gUCQ6hwx7Vx6AxPjXvQoyizZ'
    # 账户额度为 0，Pro 系列返回 403；flash 系列免费可用
    LLM_MODEL = 'agnes-2.5-flash'
    # 测验最多作答轮数：每轮提交后判分，错的题带上历次错误答案重新问，直到全对
    QUIZ_MAX_ROUNDS = 3
    # 错题报告统一输出目录（脚本目录下，一个文件夹不细分子目录）
    REPORT_DIR = os.path.join(BASE_DIR, '错题报告')
    # 考试答题日志输出目录（save_log=True 且不满分才写入，一场考试一个文件）
    EXAM_LOG_DIR = os.path.join(BASE_DIR, '考试答题日志')
    # 考试滑块验证码版本（cx_captcha 组件，与页面加载的 load-i.min.js 一致）
    EXAM_CAPTCHA_VERSION = '1.1.22'
    # 考试满分线：已完成但最终成绩低于此分的自动重考刷分（取最高成绩规则，重考不亏）
    EXAM_FULL_SCORE = 100.0
    # 考试滑块缺口识别引擎（utils/SliderCaptchaOcr，进程内懒加载复用）
    __gap_detector = None

    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.session = requests.session()
        self.public_headers = {
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Pragma': 'no-cache',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Chromium";v="152", "Not?A_Brand";v="24", "Google Chrome";v="152"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        }
        self.session.headers.update(self.public_headers)
        # 传输层自动重试：服务端/中间设备会回收闲置的 keep-alive 连接，长挂机的心跳
        # 正好撞上旧连接就 RemoteDisconnected/10054——requests 默认 0 重试，异常一路
        # 穿透把整个脚本炸掉（单视频挂得短难触发，多视频连挂十几分钟必踩）。
        # GET/HEAD 幂等可安全换新连接重发；POST 交卷/保存类不在传输层重试，由
        # __req_with_retry 统一兜底
        retry = requests.adapters.Retry(total=3, connect=3, read=2, backoff_factor=0.5,
                                        allowed_methods=frozenset({'GET', 'HEAD', 'OPTIONS'}))
        adapter = requests.adapters.HTTPAdapter(max_retries=retry)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        self.course_list = []
        self.capter_list = []
        # 章节树自带的数据（studentcourse 页面一次请求全都有，无需逐章进详情页查）：
        # chapter_status: {章节id: 待完成任务点数}（已完成章为0）；course_pts: 课程总进度(已完成x, 总y)
        self.chapter_status = {}
        self.chapter_done = set()
        self.course_pts = None
        # 实时任务点计数：main 里以 course_pts 为基线初始化，每个任务点完成时+1并即时输出
        # 进度条；all 模式章节结束后用服务端刷新校正回权威值
        self.pts_done = 0
        self.pts_all = 0
        self._stu_params = None  # studentcourse 页请求参数，refresh_chapter_status 重拉时复用
        self.fid = None
        self.realname = None

    def get_cards_url(self, clazzid, courseid, knowledgeid, cpi, num=0):
        return (
            "https://mooc1.chaoxing.com"
            f"/mooc-ans/knowledge/cards?"
            f"clazzid={clazzid}"
            f"&courseid={courseid}"
            f"&knowledgeid={knowledgeid}"
            f"&num={num}"
            f"&ut=s"
            f"&cpi={cpi}"
            f"&v=2025-0424-1038-4"
            f"&mooc2=1"
            f"&isMicroCourse=false"
            f"&editorPreview=0"
        )

    # 登录账号/密码 AES 加密（与超星登录页 login.js 的 encryptByAES 一致）：
    # 固定 key "u2oh6Vu^HWe4_AES"，AES-CBC，iv=key，Pkcs7，输出 Base64
    @staticmethod
    def get_aes_encrypt(word):
        key = b'u2oh6Vu^HWe4_AES'
        cipher = AES.new(key, AES.MODE_CBC, iv=key)
        return base64.b64encode(cipher.encrypt(pad(word.encode('utf-8'), AES.block_size))).decode()

    # 1、登录首页获取源数据
    def __get_login_meta(self):
        login_meta_url = 'https://passport2.chaoxing.com/login'
        login_meta_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Referer': 'https://v200255.mooc.chaoxing.com/',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-site',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
        }

        response = self.session.get(login_meta_url, headers=login_meta_headers)
        return response

    # 2、登录
    def login(self):
        login_url = 'https://passport2.chaoxing.com/fanyalogin'
        meta_response = self.__get_login_meta()
        soup = BeautifulSoup(meta_response.content, 'html.parser')
        fid = soup.select_one('#fid').get('value')
        refer = soup.select_one('#refer').get('value')
        t = soup.select_one('#t').get('value')
        forbidotherlogin = soup.select_one('#forbidotherlogin').get('value')
        validate = soup.select_one('#validate').get('value', '')
        doubleFactorLogin = soup.select_one('#doubleFactorLogin').get('value')
        independentId = soup.select_one('#independentId').get('value')
        username_encrypt = self.get_aes_encrypt(self.username)
        password_encrypt = self.get_aes_encrypt(self.password)
        login_headers = {
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Origin': 'https://passport2.chaoxing.com',
            'Referer': 'https://passport2.chaoxing.com/login?newversion=true&loginType=4&fid=200255&refer=http://i.mooc.chaoxing.com',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'X-Requested-With': 'XMLHttpRequest',
        }
        data = {
            'fid': fid,
            'uname': username_encrypt,
            'password': password_encrypt,
            'refer': refer,
            't': t,
            'forbidotherlogin': forbidotherlogin,
            'validate': validate,
            'doubleFactorLogin': doubleFactorLogin,
            'independentId': independentId,
        }
        response = self.session.post(login_url, headers=login_headers, data=data)
        self.fid = fid
        try:
            result = response.json()
        except ValueError:
            result = {}
        if result.get('status') is True:
            print('✅ 登录成功')
            self.realname = self.__fetch_realname()
            print('👤 当前用户：%s' % (self.realname or self.username))
        else:
            print('❌ 登录失败: %s' % result.get('mesg', response.text[:200]))
            # 登录失败不再返回：带着未登录会话继续跑只会在后续接口上崩出难看的 traceback
            raise SystemExit(1)
        return response

    # 2.1、登录成功后从 i.mooc 空间页提取实际昵称（服务端渲染，fanyalogin 响应里没有）
    def __fetch_realname(self):
        try:
            response = self.__risky_req('GET', 'http://i.mooc.chaoxing.com/space/index')
            soup = BeautifulSoup(response.content, 'html.parser')
            name_tag = soup.select_one('span.zt_u_name') or soup.select_one('p.personalName')
            if name_tag is not None:
                return name_tag.get_text(strip=True)
        except Exception:
            pass
        return None

    # 3、所有课程首页获取源数据
    def __get_course_list_meta(self):
        url = 'https://i.chaoxing.com/base'
        course_list_meta_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Referer': 'https://i.chaoxing.com/',
            'Sec-Fetch-Dest': 'iframe',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-site',
            'Upgrade-Insecure-Requests': '1',
        }
        ts = str(int(time.time() * 1000))
        params = {
            "t": ts
        }
        meta_response = self.session.get(url, params=params)
        soup = BeautifulSoup(meta_response.content, 'html.parser')
        course_list_meta_url = soup.select_one('[name="课程"]').get('dataurl')
        response = self.session.get(course_list_meta_url, headers=course_list_meta_headers)
        return response

    # 4、获取课程列表
    def get_course_list(self):
        course_list_url = 'https://mooc2-ans.chaoxing.com/mooc2-ans/visit/courselistdata'
        course_list_headers = {
            'Accept': 'text/html, */*; q=0.01',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Origin': 'https://mooc2-ans.chaoxing.com',
            'Referer': 'https://mooc2-ans.chaoxing.com/visit/interaction?s=3a89767141e1a86d253c54a2eca633bc',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'X-Requested-With': 'XMLHttpRequest',
        }
        meta_response = self.__get_course_list_meta()
        soup = BeautifulSoup(meta_response.content, 'html.parser')
        courseType = soup.select_one('#myLearn').get('coursetype')
        courseFolderId = soup.select_one('#courseFolderId').get('value')
        query = soup.select_one('#searchInput').get('value', '')
        pageHeader = soup.select_one('#tchPageHeader').get('value')
        single = soup.select_one('#single').get('value')
        superstarClass = soup.select_one('#superstarClass').get('value')
        isFirefly = soup.select_one('#isFirefly').get('value')
        fid = soup.select_one('#filterFid').get('value', '')
        data = {
            'courseType': courseType,
            'courseFolderId': courseFolderId,
            'query': query,
            'pageHeader': pageHeader,
            'single': single,
            'superstarClass': superstarClass,
            'isFirefly': isFirefly,
            'fid': fid,
        }
        response = self.session.post(course_list_url, headers=course_list_headers, data=data)
        soup = BeautifulSoup(response.content, 'html.parser')
        course_list_origin = soup.select('#stuNormalCourseListDiv > div')
        for item in course_list_origin:
            course_infos = item.select_one('div .course-info .inlineBlock a')
            course_url = course_infos.get('href')
            title = course_infos.select_one('span').get_text()
            self.course_list.append((title, course_url))

    # 5、单个课程首页获取源数据
    def __get_course_meta(self, index):
        meta_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Referer': 'https://mooc2-ans.chaoxing.com/visit/interaction?s=2fb658e658dd997dea0dd7057ec66e1f',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
        }
        target_course = self.course_list[index]
        target_course_url = target_course[-1]
        response = self.__risky_req('GET', target_course_url, headers=meta_headers)
        return response

    def get_course(self, index):
        # 课程大标题（如「大学生安全教育（入学篇）」），供答题提示词动态引用
        self.course_name = (self.course_list[index][0] or '').strip()
        course_url = 'https://mooc2-ans.chaoxing.com/mooc2-ans/mycourse/studentcourse'
        course_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Referer': 'https://mooc2-ans.chaoxing.com/visit/interaction?s=2fb658e658dd997dea0dd7057ec66e1f',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
        }
        meta_response = self.__get_course_meta(index)
        soup = BeautifulSoup(meta_response.content, 'html.parser')
        if soup.select_one('#courseid') is None:
            raise RuntimeError('课程页被 9010 风控拦截（自动过码未成功），稍后重试')
        self.courseid = soup.select_one('#courseid').get('value')
        self.clazzid = soup.select_one('#clazzid').get('value')
        self.cpi = soup.select_one('#cpi').get('value')
        ut = soup.select_one('#heardUt').get('value')
        t = soup.select_one('#t').get('value')
        enc = soup.select_one('#enc').get('value')
        self.enc = soup.select_one('#oldenc').get('value')
        self.openc = soup.select_one('#openc').get('value')
        ee = soup.select_one('#examEnc')
        self.exam_enc = ee.get('value') if ee is not None else ''
        params = {
            'courseid': self.courseid,
            'clazzid': self.clazzid,
            'cpi': self.cpi,
            'ut': ut,
            't': t,
            'stuenc': enc,
        }
        self._stu_params = params  # 留给 refresh_chapter_status 重拉章节树用
        response = self.__risky_req('GET', course_url, params=params, headers=course_headers)
        return response

    def get_capter_list(self, response):
        soup = BeautifulSoup(response.content, 'html.parser')
        self.capter_list = []  # refresh 重拉时重填，避免重复累加
        # 章节树每项（div.chapter_item[id^=cur]）自带全部信息：
        # id=cur+章节id、.clicktitle=章节名、input.knowledgeJobCount=待完成任务点数
        # （已完成/无任务章没有此字段视为0）、.catalog_state.icon_yiwanc=已完成；
        # 头部 .chapter_head 带课程总进度「已完成任务点: x/y」
        self.chapter_status = {}
        self.chapter_done = set()
        for div in soup.select('div.chapter_item[id^=cur]'):
            cid = div.get('id')[len('cur'):]
            name = div.select_one('.clicktitle').get_text()
            self.capter_list.append((cid, re.sub(r'\s+', ' ', name).strip()))
            cnt = div.select_one('input.knowledgeJobCount')
            self.chapter_status[cid] = int(cnt.get('value')) if cnt else 0
            if div.select_one('.catalog_state.icon_yiwanc') is not None:
                self.chapter_done.add(cid)
        head = soup.select_one('.chapter_head')
        m = re.search(r'已完成任务点\D*(\d+)\s*/\s*(\d+)', head.get_text()) if head else None
        self.course_pts = (int(m.group(1)), int(m.group(2))) if m else None

    # 重拉章节树（studentcourse 页），刷新每章待完成数和课程总进度（刷完一章后调用）
    def refresh_chapter_status(self):
        if self._stu_params is None:
            return
        params = dict(self._stu_params, t=str(int(time.time() * 1000)))
        response = self.__risky_req('GET',
                                    'https://mooc2-ans.chaoxing.com/mooc2-ans/mycourse/studentcourse',
                                    params=params)
        self.get_capter_list(response)

    # 6、章节页获取源数据（studentstudy -> cards，cards 页面里带 mArg 任务数据）
    # num 是卡序号：一个章节可能多卡（如 num=0 视频卡、num=1 测验卡），每卡一份 mArg
    # 请求过快会触发 9010 图片验证码风控（202 + 提示页面），自动过码后重试
    def __get_task_meta(self, task_id, num=0):
        task_url = 'https://mooc1.chaoxing.com/mycourse/studentstudy'
        params = {
            'chapterId': task_id,
            'courseId': self.courseid,
            'clazzid': self.clazzid,
            'cpi': self.cpi,
            'enc': self.enc,
            'mooc2': '1',
            'hidetype': '0',
            'openc': self.openc,
        }
        cards_url = None
        for _ in range(3):
            self.session.get(task_url, params=params)
            time.sleep(0.5)  # 请求节流：studentstudy+cards 零间隔连发会触发9010风控
            cards_url = cards_url or self.get_cards_url(clazzid=self.clazzid, courseid=self.courseid,
                                                        knowledgeid=task_id, cpi=self.cpi, num=num)
            response = self.session.get(cards_url)
            if response.status_code != 202 and '9010' not in response.text:
                return response
            print('🛡️ 触发超星风控(9010)，自动过验证码...')
            if self.__verify_captcha():  # 内部等待窗口期直到真正过码/解除，🛡️只提示一次
                continue
            return response  # 过码失败，返回被拦的响应
        return response

    # 9010 拦截响应的验证码类型探测与过码分派：滑块强特征（captchaId/滑块关键词）
    # 走纯协议滑块过验（复用考试入口的 __slide_captcha_validate，拿到 captchaId 才能过），
    # 图形码/无特征走图形 OCR 过码；类型每次打印出来——以前只会「验证码未就绪」干等，
    # 看不出这次风控到底是滑块还是图形码
    def __resolve_9010(self, response):
        text = response.text or ''
        cm = re.search(r'captchaId["\'\s=:]+([a-f0-9]{16,})', text, re.I)
        if cm or re.search(r'滑块|slider|cxcaptcha|geetest', text, re.I):
            print('🛡️ 风控验证码类型：滑块')
            if cm:
                return bool(self.__slide_captcha_validate(
                    cm.group(1), 'https://mooc1.chaoxing.com/'))
            print('🛡️ 拦截页未带 captchaId，无法协议过滑块')
            return False
        if re.search(r'processVerify', text, re.I):
            print('🛡️ 风控验证码类型：图形验证码')
        else:
            print('🛡️ 风控验证码类型：未知，按图形验证码处理')
        return self.__verify_captcha()

    # 刷课中途的9010自愈请求：心跳/提交等关键请求被拦(202/9010)时过码后自动重发，
    # 避免「视频看到一半被拦、心跳悄悄丢、最后 isPassed=false」这种中途断流。
    # 与 __get_task_meta 的过码循环同构：过码后重发并复查，仍被拦则再过码（最多3轮）——
    # 「风控已解除」判定来自验证码图连续404，窗口期也 404 有误判可能，单次放行会把
    # 被拦响应静默交给调用方；复查不过就能再次进入过码，图就绪后走真过码（OCR）
    # 单次请求网络级重试（POST 也覆盖）：撞上被掐的连接/瞬断时退避后重发，
    # 3 次仍失败才抛给调用方；传输层 Retry 只管 GET，POST 的断连重发在这里兜底
    def __req_with_retry(self, method, url, **kw):
        for attempt in range(3):
            try:
                return self.session.request(method, url, **kw)
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise
                print('⚠ 网络异常 %s（第 %d/3 次），%d秒后重发' % (
                    type(e).__name__, attempt + 1, 2 * (attempt + 1)))
                time.sleep(2 * (attempt + 1))

    def __risky_req(self, method, url, **kw):
        response = self.__req_with_retry(method, url, **kw)
        for _ in range(3):
            if response.status_code != 202 and '9010' not in response.text:
                return response
            print('🛡️ 触发超星风控(9010)，自动过验证码...')
            if not self.__resolve_9010(response):
                return response  # 过码失败，返回被拦响应
            response = self.__req_with_retry(method, url, **kw)
        return response

    # 6.5、9010 风控自动过码：拉验证码图 -> 识别4位码 -> 提交 processVerify
    # 验证码图片: /processVerifyPng.ac?t=随机数；提交: /html/processVerify.ac (app=0, ucode=4位码)
    # 风控刚触发的窗口期里验证码图未就绪（坏图/404），这里内部循环等待直到真正过掉才返回：
    #   - 拉图404：连续4次判定风控已解除；否则等1秒继续拉（窗口期）
    #   - 识别不足4位：换图重拉
    def __verify_captcha(self):
        headers = {'Referer': 'https://mooc1.chaoxing.com/mooc-ans/knowledge/cards'}
        miss = 0
        for _ in range(20):
            png = self.session.get('https://mooc1.chaoxing.com/processVerifyPng.ac',
                                   params={'t': random.random() * 2147483647}, headers=headers)
            if png.status_code == 404:  # 非风控态该接口404；窗口期也404，连续多次才算解除
                miss += 1
                if miss >= 4:
                    print('✅ 连续4次无验证码图，风控已自行解除（未过码，无识别环节）')
                    return True
                print('⏳ 验证码图未就绪(404)，第%d/4次确认风控是否解除...' % miss)
                time.sleep(1)
                continue
            miss = 0
            try:
                code = self.__ocr_read(png.content)
            except Exception:
                code = ''
            if len(code) != 4:
                print('❌ 验证码识别失败，换图重试')
                time.sleep(0.5)
                continue
            print('🔎 OCR识别：%s' % code)
            self.session.post('https://mooc1.chaoxing.com/html/processVerify.ac',
                              data={'app': '0', 'ucode': code}, headers=headers)
            print('✅ 验证码已提交')
            time.sleep(0.5)
            # 复查：还能拉到验证码图说明仍在风控态，继续下一轮
            check = self.session.get('https://mooc1.chaoxing.com/processVerifyPng.ac',
                                     params={'t': random.random() * 2147483647}, headers=headers)
            if check.status_code == 404:
                print('✅ 验证码通过，风控已解除')
                return True
            print('⏳ 提交未生效，仍在风控，重试')
        print('❌ 验证码过码失败（重试20次未通过）')
        return False

    # 验证码识别：直接用 ddddocr 识别原图（其内部自带灰度预处理，外挂二值化反而毁图，实测 raw 最准）
    def __ocr_read(self, img_bytes):
        if ChaoXing.__ocr is None:
            import ddddocr
            ChaoXing.__ocr = ddddocr.DdddOcr(show_ad=False)
        prob = ChaoXing.__ocr.classification(img_bytes, probability=True)
        return (prob.get('text') or '').strip()

    # 7、从 cards 页面源码里提取 mArg = {...} 任务数据（括号平衡截取，防止字符串里带花括号）
    @staticmethod
    def __parse_marg(html_content):
        mark = 'mArg = {'
        if mark not in html_content:
            return None
        start = html_content.index(mark) + len('mArg = ') - 1
        depth = 0
        end = start
        in_str = False
        esc = False
        for i, c in enumerate(html_content[start:], start):
            if esc:
                esc = False
                continue
            if c == '\\':
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        return json.loads(html_content[start:end])

    # 8、拿章节的全部 mArg 任务数据：多卡章节循环 num 拉到没有 mArg 为止，聚合所有 attachments
    # （视频卡和测验卡各自带一份 mArg，只拉 num=0 会漏掉测验）
    def __get_marg(self, task_id):
        defaults, attachments, miss = None, [], 0
        for num in range(50):
            time.sleep(0.3)  # 拉卡稍作间隔，降低触发 9010 风控的概率
            marg = self.__parse_marg(self.__get_task_meta(task_id, num).text)
            if marg is None:
                miss += 1
                if miss >= 2:  # 连续两卡无 mArg，说明卡拉完了
                    break
                continue
            miss = 0
            if defaults is None:
                defaults = marg['defaults']
            attachments.extend(marg['attachments'])
        return {'defaults': defaults, 'attachments': attachments} if defaults else None

    # 9、视频状态接口，拿 dtoken（心跳上报路径要用）和视频时长
    def __get_video_status(self, objectid, fid):
        status_url = f'https://mooc1.chaoxing.com/ananas/status/{objectid}'
        params = {
            'k': fid,
            'flag': 'normal',
            'ro': '0',
            '_dc': str(int(time.time() * 1000)),
        }
        response = self.session.get(status_url, params=params, headers={
            'Referer': self.VIDEO_REFERER,
            'X-Requested-With': 'XMLHttpRequest',
        })
        return response.json()

    # 10、enc 签名：md5('[clazzId][userid][jobid][objectId][playingTime*1000][盐值][duration*1000][clipTime]')
    @staticmethod
    def __make_enc(clazz_id, userid, jobid, object_id, playing_time, duration, clip_time):
        s = '[%s][%s][%s][%s][%d][%s][%d][%s]' % (
            clazz_id, userid, jobid, object_id, playing_time * 1000,
            'd_yHJ!$pdA~5', duration * 1000, clip_time)
        return hashlib.md5(s.encode()).hexdigest()

    # 11、发一次心跳上报（reportUrl 是 mArg defaults 里返回的，末尾拼 dtoken 才是完整路径）
    def __send_log(self, attachment, defaults, dtoken, playing_time, isdrag):
        duration = int(attachment['attDuration'])
        clip_time = '0_%d' % duration
        enc = self.__make_enc(defaults['clazzId'], defaults['userid'], attachment['jobid'],
                              attachment['objectId'], playing_time, duration, clip_time)
        url = '%s/%s' % (defaults['reportUrl'], dtoken)
        # otherInfo 原生以裸 & 接在 URL 上，服务端按第一个 & 截断取值；
        # courseId 必须只出现一次（独立参数），URL 里参数重复会被直接 403
        params = {
            'clazzId': defaults['clazzId'],
            'playingTime': playing_time,
            'duration': duration,
            'clipTime': clip_time,
            'objectId': attachment['objectId'],
            'otherInfo': attachment['otherInfo'].split('&')[0],
            'courseId': defaults['courseid'],
            'jobid': attachment['jobid'],
            'userid': defaults['userid'],
            'isdrag': isdrag,
            'view': 'pc',
            'enc': enc,
            'rt': '0.9',
            'videoFaceCaptureEnc': attachment.get('videoFaceCaptureEnc', ''),
            'dtype': 'Video',
            '_t': str(int(time.time() * 1000)),
            'attDuration': duration,
            'attDurationEnc': attachment.get('attDurationEnc', ''),
            'courseEngineInfo': 'false',
        }
        # 手工拼 query：otherInfo 原样输出（与原生一致），其余参数照常转义
        query = '&'.join(
            '%s=%s' % (key, value if key == 'otherInfo' else quote(str(value), safe=''))
            for key, value in params.items()
        )
        try:
            response = self.__risky_req('GET', '%s?%s' % (url, query), headers={
                'Referer': self.VIDEO_REFERER,
                'X-Requested-With': 'XMLHttpRequest',
            })
        except requests.exceptions.RequestException as e:
            # 重发仍断连：丢这一拍不致命，进度按时钟累积，60s 后下一拍把完整进度补上
            print('   ⚠ 心跳上报网络异常: %s' % type(e).__name__)
            return {}
        if response.status_code != 200:
            print('心跳上报异常:', response.status_code)
            return {}
        try:
            return response.json()
        except ValueError:
            print('心跳上报响应非JSON:', response.text[:120])
            return {}

    # 12、刷完一个视频：isdrag=3 起始上报 -> isdrag=2 每60秒进度上报 -> isdrag=4 结束上报（isPassed 的开关在这里）
    # 服务端有异步复核，会按心跳的真实时间间隔核对观看时长，快进/秒刷的 isPassed=true 过几分钟会被回滚，
    # 所以这里必须按 1 倍速真实时间挂机，节奏与原生播放器 reportTimeInterval=60 一致
    def __finish_one_video(self, attachment, defaults):
        title = self.__att_title(attachment)
        duration = int(attachment['attDuration'])
        print('📺%s 开始观看 %s（真实时长等待）' % (title, self.__fmt_dur(duration)))
        status = self.__get_video_status(attachment['objectId'], defaults.get('fid') or self.fid)
        dtoken = status['dtoken']
        # 从服务端记录的播放位置续播（playTime 是毫秒）；已到片尾说明上次没判过，从头再来
        played_from = int(status.get('playTime') or 0) // 1000
        if played_from >= duration:
            played_from = 0
        if played_from:
            print('   📺%s 从 %s 处续播' % (title, self.__fmt_dur(played_from)))

        self.__send_log(attachment, defaults, dtoken, played_from, 3)

        # playingTime 严格跟随真实时钟：每真实流逝1秒进度走1秒，每60秒发一次 isdrag=2 心跳
        # 触发条件用【进度差】而不是真实时间差：进度由时钟线性推出，锚点记进度值本身，
        # 心跳请求耗时/ sleep 粒度都不会累积进间隔，显示严格 +60s
        report_interval = 60
        start_real = time.time()
        last_reported = played_from
        print('   📺%s 观看中 %s / %s' % (
            title, self.__fmt_dur(played_from), self.__fmt_dur(duration)))
        while True:
            time.sleep(1)
            playing = min(played_from + int(time.time() - start_real), duration)
            if playing >= duration:
                print('   📺%s 观看中 %s / %s' % (
                    title, self.__fmt_dur(duration), self.__fmt_dur(duration)))
                break
            if playing - last_reported >= report_interval:
                print('   📺%s 观看中 %s / %s' % (
                    title, self.__fmt_dur(playing), self.__fmt_dur(duration)))
                self.__send_log(attachment, defaults, dtoken, playing, 2)
                last_reported = playing
        resp = self.__send_log(attachment, defaults, dtoken, duration, 4)

        passed = resp.get('isPassed', False) if isinstance(resp, dict) else False
        print('✅📺%s 观看完成' % title if passed else '❌📺%s 观看失败: %s' % (title, resp))
        if passed:
            # 实时任务点：每刷完一个+1并即时输出总进度条（章节末服务端刷新会再校正）
            self.pts_done = min(self.pts_done + 1, self.pts_all)
            print(self.__progress_bar(self.pts_done, self.pts_all))
        return passed

    # 13、刷完一个章节内的所有视频任务
    def finish_video(self, task_id):
        marg = self.__get_marg(task_id)
        if marg is None:
            print('没有任务数据，跳过')
            return False

        defaults = marg['defaults']
        jobs = []
        for attachment in marg['attachments']:
            # 只要视频任务（有 jobid/objectId/attDuration 的才是）
            if 'jobid' not in attachment or 'objectId' not in attachment or 'attDuration' not in attachment:
                continue
            if attachment.get('isPassed'):
                continue
            jobs.append(attachment)
        passed_list = []
        for attachment in jobs:
            # 单个视频意外异常（断连重试仍败等）只牺牲这一个视频，不拖垮整章；
            # 没刷完的下次运行按服务端进度自动续播
            try:
                passed_list.append(self.__finish_one_video(attachment, defaults))
            except Exception as e:
                print('❌📺%s 异常中断: %s: %s' % (
                    self.__att_title(attachment), type(e).__name__, e))
                passed_list.append(False)
        return all(passed_list) if passed_list else False

    # 14、把章节附件分成视频/文档/测验三类
    # 靠 type 字段区分（video/document/workid）；视频旧形态顶层带 attDuration，也兜底识别
    @staticmethod
    def __split_tasks(marg):
        vids, docs, quizzes = [], [], []
        if marg is not None:
            for a in marg['attachments']:
                if 'jobid' not in a:
                    continue
                t = a.get('type')
                if t == 'video' or 'attDuration' in a:
                    vids.append(a)
                elif t == 'document':
                    docs.append(a)
                elif t == 'workid':
                    quizzes.append(a)
        return vids, docs, quizzes

    # 附件标题：property（mArg 解析后是 dict，兼容 JSON 字符串形态）里的 name/title（视频/文档/作业通用），
    # 去掉扩展名；无则空串，前导空格便于直接拼接
    @staticmethod
    def __att_title(a):
        prop = a.get('property')
        if isinstance(prop, str):
            try:
                prop = json.loads(prop)
            except ValueError:
                prop = {}
        name = ((prop or {}).get('name') or (prop or {}).get('title') or '').strip()
        if name:
            name = re.sub(r'\.(mp4|avi|mkv|flv|wmv|pdf|ppt|pptx|doc|docx|xls|xlsx)$', '', name, flags=re.I)
        return ' 《%s》' % name if name else ''

    # 任务清单：只列待完成的任务，每个一行（emoji+标题），标题取不到就只显示类型；
    # 过滤口径与 main 的 pending 判定一致：视频 isPassed 未过、文档/作业 job=True，mode 决定列哪些类型
    # （watch 含图文：视频+图文一起刷；course 三类都含）
    @classmethod
    def __task_list_line(cls, vids, docs, quizzes, mode):
        lines = []
        if mode in ('all', 'watch', 'course'):
            for a in vids:
                if not a.get('isPassed'):
                    lines.append('      📺%s' % cls.__att_title(a))
            for a in docs:
                if a.get('job') is True:
                    lines.append('      📄%s' % cls.__att_title(a))
        if mode in ('all', 'test', 'course'):
            for a in quizzes:
                if a.get('job') is True:
                    lines.append('      📝%s' % cls.__att_title(a))
        return '\n'.join(lines)

    # 秒 → m:ss
    @staticmethod
    def __fmt_dur(sec):
        return '%d:%02d' % (sec // 60, sec % 60)

    # 总进度条（UI 输出用）
    @staticmethod
    def __progress_bar(done, total, bar_len=30, label='🔴 任务点'):
        if total == 0:
            return '📊 %s: 0/0' % label
        filled = bar_len * done // total
        bar = '█' * filled + '░' * (bar_len - filled)
        return '📊 %s: %d/%d %s %d%%' % (label, done, total, bar, done * 100 // total)

    # 15、刷完一个章节内的所有文档任务：对每个未完成(job=true)的文档调一次完成接口
    # 网页端滚动到底部触发的 finishJob 就是这一个请求，服务端只认它，不校验真实阅读时长
    # （已实测：不滚动直接调，reload 后附件的 job 字段消失，完成持久化）
    # 响应：新完成 {"msg":"添加考核点成功","status":true}；本来就已完成 {"msg":"考核点已经完成","status":true}
    def finish_document(self, task_id):
        marg = self.__get_marg(task_id)
        if marg is None:
            print('没有任务数据，跳过')
            return False

        defaults = marg['defaults']
        ok_list = []
        for attachment in marg['attachments']:
            if attachment.get('type') != 'document' or 'jobid' not in attachment:
                continue
            if attachment.get('job') is not True:
                continue
            title = self.__att_title(attachment)
            print('📄%s 开始完成' % title)
            params = {
                'jobid': attachment['jobid'],
                'knowledgeid': defaults['knowledgeid'],
                'courseid': defaults['courseid'],
                'clazzid': defaults['clazzId'],
                'jtoken': attachment['jtoken'],
                'checkMicroTopic': 'true',
                'microTopicId': '',
                'courseEngineInfo': 'false',
            }
            response = self.__risky_req('GET', 'https://mooc1.chaoxing.com/mooc-ans/job/document', params=params,
                                        headers={
                                            'Referer': 'https://mooc1.chaoxing.com/ananas/modules/pdf/index.html?v=2026-0826-1905',
                                            'X-Requested-With': 'XMLHttpRequest',
                                        })
            try:
                resp = response.json()
            except ValueError:
                resp = {}
            done = response.status_code == 200 and resp.get('status') is True
            if done:
                print('✅📄%s %s' % (title, resp.get('msg', '')))
                # 实时任务点：每完成一个+1并即时输出总进度条
                self.pts_done = min(self.pts_done + 1, self.pts_all)
                print(self.__progress_bar(self.pts_done, self.pts_all))
            else:
                print('❌📄%s %s' % (title, response.text[:200]))
            ok_list.append(done)
        return all(ok_list) if ok_list else False

    # ---------- cxsecret 字体反爬解码 ----------

    # 16、从题面 HTML 抽内嵌 base64 TTF（@font-face 的 data:application/font-ttf;base64,...）
    @staticmethod
    def __extract_secret_font(html):
        m = re.search(r'base64,([A-Za-z0-9+/=]+)', html)
        return base64.b64decode(m.group(1)) if m else None

    # 17、基准字体（思源黑体 CN Normal）：本地没有就下载
    def __ensure_base_font(self):
        if not os.path.exists(self.CXSECRET_BASE_FONT):
            os.makedirs(os.path.dirname(self.CXSECRET_BASE_FONT), exist_ok=True)
            print('🔤 下载基准字体 思源黑体 CN Normal...')
            r = self.session.get(self.CXSECRET_BASE_FONT_URL, timeout=120)
            with open(self.CXSECRET_BASE_FONT, 'wb') as f:
                f.write(r.content)
        return self.CXSECRET_BASE_FONT

    # 18、自动字形比对：渲染密文字与思源黑体全部 CJK 逐字 IoU（归一化到 128 网格），取最高分为真字
    # 实测当前字体 18 个密文字全部自动识别正确（含 IoU 仅 0.57 的"对"，靠拉伸归一化区分结构）
    @staticmethod
    def __auto_match_secret(font_bytes, base_font_path, px=100, grid=128):
        def glyph_arr(font, code):
            img = Image.new('L', (px * 3, px * 3), 0)
            ImageDraw.Draw(img).text((px * 1.5, px * 1.5), chr(code), font=font, fill=255, anchor='mm')
            box = img.getbbox()
            if not box:
                return None
            # 拉伸填满网格：消除整体宽高比差异，只比笔画结构
            return np.array(img.crop(box).resize((grid, grid)), dtype=np.uint8) > 100

        def iou(a, b):
            union = (a | b).sum()
            return (a & b).sum() / union if union else 0.0

        secret_codes = [c for c in TTFont(io.BytesIO(font_bytes)).getBestCmap() if c >= 0x3400]
        sfont = ImageFont.truetype(io.BytesIO(font_bytes), px)
        secrets = {}
        for code in secret_codes:
            arr = glyph_arr(sfont, code)
            if arr is not None:
                secrets[code] = arr

        bfont = ImageFont.truetype(base_font_path, px)
        bases = []
        for code, _ in TTFont(base_font_path).getBestCmap().items():
            if not (0x3400 <= code <= 0x9FFF):
                continue
            arr = glyph_arr(bfont, code)
            if arr is not None:
                bases.append((code, arr))

        mapping = {}
        for code, sarr in secrets.items():
            best_code, best = None, 0.0
            for cand, arr in bases:
                v = iou(sarr, arr)
                if v > best:
                    best, best_code = v, cand
            if best_code is not None:
                mapping[chr(code)] = chr(best_code)
        return mapping

    # 19、建立密文映射：全局一张 {密字: 真字} 表 + 字体免检名单，
    # 缓存结构 {"table": {...}, "verified": ["字体md5摘要"...]}。字体池可能有几十个字体文件，
    # 但实测各字体映射一致（服务端=固定密文表+轮换字体）；verified 里的字体直接用表、零比对，
    # 新字体首次露面才全量校验一次：完全一致→进免检名单；发现新密字→并入表；冲突→警告走人工 CXSECRET_MAP 兜底
    def __build_secret_map(self, font_bytes):
        table, verified = {}, []
        if os.path.exists(self.CXSECRET_CACHE):
            try:
                with open(self.CXSECRET_CACHE, encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get('table'), dict):
                    table = data['table']
                    verified = [m for m in (data.get('verified') or []) if isinstance(m, str)]
                elif isinstance(data, dict) and data:
                    # 旧格式迁移：纯 {密字:真字} 单表、{"<md5>": {..}} 多字体、{"md5":..,"map":{..}} 单字体
                    merged, conflict, mds = {}, [], []
                    if data.get('md5') and isinstance(data.get('map'), dict):
                        mds.append(data['md5'])
                        merged = dict(data['map'])
                    else:
                        for md5, mp in data.items():
                            if isinstance(mp, dict):
                                mds.append(md5)
                                for k, v in mp.items():
                                    if k in merged and merged[k] != v:
                                        conflict.append(k)
                                    merged[k] = v
                        if not mds:
                            merged = {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
                    table = merged
                    verified = mds
                    self.__save_secret_cache(table, verified)
                    print('🔤 旧缓存已迁移为全局密文表（%d 字%s）' % (
                        len(table), '，冲突 %d 处以最后为准' % len(conflict) if conflict else ''))
            except (ValueError, OSError):
                pass

        fmd5 = hashlib.md5(font_bytes).hexdigest()[:8] if font_bytes else ''
        if not table:
            print('🔤 首次运行，全量字形比对建表（约 1 分钟）...')
            table = self.__auto_match_secret(font_bytes, self.__ensure_base_font())
            if fmd5:
                verified.append(fmd5)
            self.__save_secret_cache(table, verified)
            print('🔤 自动识别 %d 字，映射已缓存到 %s' % (len(table), self.CXSECRET_CACHE))
        elif fmd5 and fmd5 not in verified:
            print('🔤 新字体 %s 首次出现，校验与全局表一致性...' % fmd5)
            fm = self.__auto_match_secret(font_bytes, self.__ensure_base_font())
            conflict = {k: (table[k], v) for k, v in fm.items() if k in table and table[k] != v}
            new_keys = {k: v for k, v in fm.items() if k not in table}
            if conflict:
                print('🔤 ⚠ 字体 %s 有 %d 处映射与全局表不同:' % (fmd5, len(conflict)))
                for k, (old, new) in conflict.items():
                    print('   %s: 表里=%s 字体=%s' % (k, old, new))
                print('   请人工核对后把正确映射加入脚本顶部 CXSECRET_MAP 兜底')
            if new_keys:
                table.update(new_keys)
                print('🔤 字体 %s 发现 %d 个新密字，并入全局表:' % (fmd5, len(new_keys)))
                for k, v in new_keys.items():
                    print('   %s → %s' % (k, v))
            if not conflict and not new_keys:
                print('🔤 字体 %s 与全局表完全一致，加入免检名单' % fmd5)
            verified.append(fmd5)
            self.__save_secret_cache(table, verified)
        else:
            print('🔤 cxsecret 映射命中缓存（%d 字）' % len(table))

        conflict = [k for k, v in self.CXSECRET_MAP.items() if k in table and table[k] != v]
        if conflict:
            print('🔤 ⚠ 自动结果与人工核对冲突: %s，以人工核对为准' % conflict)
        mapping = dict(table)
        mapping.update(self.CXSECRET_MAP)
        return mapping

    # 19.1、保存密文映射缓存
    def __save_secret_cache(self, table, verified):
        with open(self.CXSECRET_CACHE, 'w', encoding='utf-8') as f:
            json.dump({'table': table, 'verified': sorted(verified)}, f, ensure_ascii=False, indent=2)

    # 20、密文文本解码：映射表内的码位替换成真字，表外原样保留
    @staticmethod
    def __decode_cxsecret(text, mapping):
        if not mapping:
            return text
        return ''.join(mapping.get(c, c) for c in text)

    # ---------- 章节测验（答题） ----------

    # 21、拉题面：api/work 会 302 到 doHomeWorkNew 题面页，返回 (HTML, 最终URL)
    def __fetch_quiz_html(self, att, defaults):
        params = {
            'api': '1',
            'workId': att['property']['workid'],
            'jobid': att['jobid'],
            'originJobId': att['jobid'],
            'needRedirect': 'true',
            'skipHeader': 'true',
            'knowledgeid': defaults['knowledgeid'],
            'ktoken': defaults['ktoken'],
            'cpi': defaults['cpi'],
            'ut': 's',
            'clazzId': defaults['clazzId'],
            'type': '',
            'enc': att['enc'],
            'utenc': 'undefined',
            'mooc2': '1',
            'courseid': defaults['courseid'],
        }
        r = self.__risky_req('GET', 'https://mooc1.chaoxing.com/mooc-ans/api/work', params=params)
        return r.text, r.url

    # 22、解析题面：每个 TiMu → {qid, qtype, answertype, stem, options:[(data, 文本)]}
    @classmethod
    def __parse_quiz(cls, html, mapping):
        soup = BeautifulSoup(html, 'html.parser')
        questions = []
        for timu in soup.select('div.TiMu'):
            title_el = timu.select_one('.Zy_TItle')
            if title_el is None:
                continue
            stem = cls.__decode_cxsecret(title_el.get_text(' ', strip=True), mapping)
            lis = timu.select('li[qid][qtype]')
            type_input = timu.select_one('input[name^="answertype"]')
            if not lis:
                # 简答/填空类无选项题（题型 4）：没有 li[qid][qtype] 选项结构，qid 从
                # answertype 输入框名字后缀拿，qtype 用其 value，options 留空走文本作答
                m = re.match(r'answertype(\d+)$', type_input.get('name', '')) if type_input is not None else None
                if m is None:
                    continue
                questions.append({
                    'qid': m.group(1),
                    'qtype': type_input.get('value', ''),
                    'answertype': type_input.get('value', ''),
                    'stem': stem,
                    'options': [],
                })
                continue
            options = []
            for li in lis:
                span = li.select_one('span.num_option_dx, span.num_option')
                data = span.get('data', '') if span is not None else ''
                if not data:
                    # 选项节点缺 data（题面结构变化）：无效选项丢弃；否则空 data 进 letters
                    # 会让 __ask_llm 里 [%s] 空集炸正则（unterminated character set）
                    continue
                text_el = li.select_one('a.after')
                text = text_el.get_text(' ', strip=True) if text_el is not None else li.get_text(' ', strip=True)
                options.append((data, cls.__decode_cxsecret(text, mapping)))
            questions.append({
                'qid': lis[0]['qid'],
                'qtype': lis[0]['qtype'],
                'answertype': type_input.get('value', '') if type_input is not None else '',
                'stem': stem,
                'options': options,
            })
        return questions

    # 23、大模型答题：题面 prompt（题目+选项+历次错误答案）→ 回复 → 解析出选项 data 列表
    # 判断题统一用 A=对 B=错 喂给模型（true/false 当字母模型回答不稳定），解析后再映射回去；
    # 多选时附加题型说明，避免模型只回一个选项；判定用双证据：qtype=='1'（章节 li[qtype] 与
    # 考试 fd type{qid} 编号体系未实测互证）或 type_name 含"多选"（typeName{qid} 中文题型名，无歧义）
    def __ask_llm(self, stem, options, wrong_answers=None, qtype='', type_name=''):
        # 简答题（无选项）不经这里：__do_quiz 里直接留空提交待批阅，本函数只处理客观题
        if not options:
            # 客观题但选项全空（题面结构异常）：无选项可组提示词，本题留空返回；
            # 测验/考试共用入口，必须防御（letters 空集曾炸 [%s] 正则）
            print('   ⚠ 选项未识别出有效 data，本题留空')
            return [], ''
        is_judge = any(d in ('true', 'false') for d, _ in options)
        lines = [stem]
        for d, t in options:
            if is_judge:
                d = 'A' if d == 'true' else 'B'
            lines.append('%s. %s' % (d, t))
        for w in wrong_answers or []:
            w_show = ' '.join({'true': 'A', 'false': 'B'}.get(x, x) for x in w.split())
            lines.append('答案 %s 不是正确的答案' % w_show)
        lines.append('本题目主题是%s，请基于这个主题给我这道题正确选项的答案，如果遇到有争议的题目先排查错误选项再仔细思考选择正确的答案，若涉及政治方面问题需要联网搜索寻找权威答案不要瞎猜（仅返回选项，不需要讲解）'
                     % getattr(self, 'course_name', ''))
        if str(qtype) == '1' or '多选' in (type_name or ''):  # 单选/判断不干扰模型
            lines.append('注：本题为多选题！！！')
        prompt = '\n'.join(lines)
        print('   🤖 当前模型: %s' % self.LLM_MODEL)
        if getattr(self, 'show_prompt', False):
            print('   🤖 提示词：')
            print(prompt)
        payload = {
            'model': self.LLM_MODEL,
            'temperature': 0.1,
            'messages': [{'role': 'user', 'content': '\n'.join(lines)}],
        }
        headers = {'Authorization': 'Bearer %s' % self.LLM_API_KEY}
        # 超时/网络异常/HTTP 错误/空回复统一重试，最多 3 次，都失败返回空答案（本题本轮不选，下轮再试）
        r = None
        for attempt in range(3):
            try:
                r = requests.post(self.LLM_API_URL, json=payload, headers=headers, timeout=120)
            except requests.RequestException as e:
                r = None
                print('   ⚠ 模型请求异常 %s（第 %d/3 次）' % (type(e).__name__, attempt + 1))
            else:
                if r.status_code == 200:
                    break
                print('   ⚠ 模型 HTTP %s（第 %d/3 次）' % (r.status_code, attempt + 1))
                r = None
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
        if r is None:
            print('   ❌ 模型 3 次均失败，本题本轮跳过，下一轮再试')
            return [], prompt
        try:
            reply = r.json()['choices'][0]['message']['content'].strip()
        except (ValueError, KeyError, IndexError):
            print('   ❌ 模型响应格式异常，本题本轮跳过')
            return [], prompt
        if not reply:
            print('   ❌ 模型返回空内容，本题本轮跳过')
            return [], prompt
        print('   💬 %s' % reply.replace('\n', ' '))
        if is_judge:
            # 先找回复里的 A/B，没有再按 对/错 关键字兜底；判断题只取一个
            groups = re.findall(r'[AB]+', reply.upper())
            best = max(groups, key=len) if groups else ''
            if not best:
                if re.search(r'错|不|false', reply, re.I):
                    best = 'B'
                elif re.search(r'对|正确|true', reply, re.I):
                    best = 'A'
            return (['true' if best[:1] == 'A' else 'false'] if best else []), prompt
        # 选择题：字母集按题目实际选项动态生成（多选可到 E/F，写死 A-D 会漏选）
        letters = ''.join(sorted(set(d for d, _ in options)))
        L = re.escape(letters)
        up = reply.upper()
        picked = []
        for m in re.finditer(r'(?:正确答案|答案是|答案为|答案|正确选项|answer)[^A-Z\n]{0,6}((?:[%s][、,，\s]*)+)' % L,
                             up):
            picked = re.findall(r'[%s]' % L, m.group(1))  # 取最后一个匹配（答案通常在结尾）
        if not picked:
            # 无关键词：找分隔字母列表（如 markdown 里的 "A、B、D"）
            m = re.search(r'(?<![A-Z\-.])([%s](?:[、,，\s/]+[%s])+)' % (L, L), up)
            if m:
                picked = re.findall(r'[%s]' % L, m.group(1))
        if not picked:
            # 连续串兜底（如 "ABCDE"）
            groups = re.findall(r'[%s]+' % L, up)
            picked = list(max(groups, key=len)) if groups else []
        return [d for d in [x for x, _ in options] if d in picked], prompt

    # 24、组表单提交：隐藏字段全部沿用题面现值，填 answer{qid} + answeraqid（全部 qid 逗号串）；
    # dry_run=True 只打印不提交；题面 JS 提交前会在 URL 追加 ua/formType/saveStatus/pos/version，enc/pyFlag 为空
    def __submit_quiz(self, html, page_url, questions, answers, dry_run=False):
        soup = BeautifulSoup(html, 'html.parser')
        form = soup.find('form', id='form1')
        if form is None:
            print('❌📝 找不到提交表单 form1')
            return False
        action = urljoin(page_url, form.get('action', ''))
        action += '&ua=pc&formType=post&saveStatus=1&pos=&version=1'
        data = {}
        for inp in form.find_all('input'):
            name = inp.get('name')
            if name:
                data[name] = inp.get('value', '')
        data['answerwqbid'] = ','.join(q['qid'] for q in questions) + ','
        for q in questions:
            picked = answers.get(q['qid'])
            if picked:
                data['answer%s' % q['qid']] = ''.join(picked)
        if dry_run:
            print('📝 [dry-run] 不真实提交，action=%s' % action)
            for q in questions:
                print(
                    '   answer%s=%s (answertype=%s)' % (q['qid'], data.get('answer%s' % q['qid'], ''), q['answertype']))
            return True
        r = self.__risky_req('POST', action, data=data, headers={
            'Referer': page_url,
            'Content-Type': 'application/x-www-form-urlencoded',
        })
        try:
            return r.json()
        except ValueError:
            return None

    # 25、刷一个章节的测验：拉题面 → 提取字体自动解码 → 客观题 LLM 多轮答题 → 提交判分 → 记录结果；
    # 简答题一律不作答留空待老师批阅；full_score=True 时客观题未全对自动重做争取满分，默认提交即收工
    def finish_quiz(self, task_id, dry_run=False, full_score=False):
        marg = self.__get_marg(task_id)
        quizzes = [a for a in marg['attachments']
                   if a.get('type') == 'workid' and 'jobid' in a] if marg is not None else []
        if not quizzes:
            print('没有测验任务')
            return False
        defaults = marg['defaults']
        # 章节名用于错题报告文件名（capter_list 里该 task_id 对应的名称，查不到兜底用 task_id）
        chapter_name = str(task_id)
        for cid, cname in self.capter_list:
            if str(cid) == str(task_id):
                chapter_name = cname
                break
        ok_list = []
        for att in quizzes:
            title = self.__att_title(att)
            if att.get('job') is not True:
                ok_list.append(True)
                continue
            print('📝%s 开始作答' % title)
            html, page_url = self.__fetch_quiz_html(att, defaults)
            if 'TiMu' not in html:
                print('❌📝%s 拉题面失败: %s' % (title, html[:150]))
                ok_list.append(False)
                continue
            # 已批阅页也渲染 TiMu 题面（带 marking_dui/marking_cuo 对错标记），直接提交会被拒，
            # 这里前置检测：调 retest 重置后重新拉作答页，不浪费一轮作答
            if 'marking_dui' in html or 'marking_cuo' in html or 'Finalresult' in html:
                print('📝%s 检测到已批阅，自动重做' % title)
                if not self.__redo_quiz(page_url):
                    ok_list.append(False)
                    continue
                html, page_url = self.__fetch_quiz_html(att, defaults)
                if 'TiMu' not in html:
                    print('❌📝%s 重做后拉题面失败: %s' % (title, html[:150]))
                    ok_list.append(False)
                    continue

            def fetch_page():
                # 重做后重新拉题面并解析（字体映射重走一遍）
                h, u2 = self.__fetch_quiz_html(att, defaults)
                f2 = self.__extract_secret_font(h)
                m2 = self.__build_secret_map(f2) if f2 else {}
                return h, u2, self.__parse_quiz(h, m2)

            font = self.__extract_secret_font(html)
            mapping = self.__build_secret_map(font) if font else {}
            questions = self.__parse_quiz(html, mapping)
            print('   📄 共 %d 题' % len(questions))
            ok_list.append(self.__do_quiz(att['jobid'], html, page_url, questions, dry_run,
                                          chapter_name, fetch_page, full_score))
        # 返回本轮新完成的份数（实时进度条已在完成时即时输出，这里仅作结果参考）
        return sum(1 for ok in ok_list if ok)

    # 25.1、单份作业的多轮作答闭环：
    #   每轮：输出题目 → 客观题 LLM 作答（判对的题沿用，判错的题带历次错误答案重问）→ 提交 → 判分；
    #   简答题一律留空待批阅；提交被拒「已批阅作业」时调重做接口重置再拉新题面；
    #   达标口径=客观题全对（简答题待老师批阅不参与判定，纯简答测验视同达标）：
    #   full_score=True 未达标自动重做（最多 QUIZ_MAX_ROUNDS 轮争取满分），默认提交一次即收工
    def __do_quiz(self, jobid, html, page_url, questions, dry_run, chapter_name,
                  fetch_page=None, full_score=False):
        if not questions:
            print('❌📝 题面解析为空，无法作答（题型超出支持范围？）')
            return False
        has_short = any(q['qtype'] == '4' for q in questions)
        wrong_hist = {q['qid']: [] for q in questions}  # qid → 历次错误答案
        last_prompt = {}  # qid → 最后一次发给 LLM 的提示词
        correct = {}  # qid → 已判对的答案
        all_right = False
        submitted_ok = False  # 默认模式：提交一次即收工，不追求满分
        for rnd in range(1, self.QUIZ_MAX_ROUNDS + 1):
            # 轮次标题只在满分模式下显示：默认提交一次即收工永远只有一轮，
            # 打「第 1/3 轮」会让用户误以为要跑 3 轮
            if full_score:
                print('━━━━━━ 第 %d/%d 轮作答 ━━━━━━' % (rnd, self.QUIZ_MAX_ROUNDS))
            answers = {}
            for q in questions:
                self.__print_question(q)
                if q['qid'] in correct:
                    picked = correct[q['qid']]
                    print('   答案：%s（上轮已判对，直接沿用）' % self.__fmt_answer(picked))
                elif q['qtype'] == '4':
                    # 简答题一律不作答：留空提交，等老师批阅（answerwqbid 仍含其 qid，否则 code-2）
                    picked = []
                    print('   答案：（简答题不作答，留空待批阅）')
                elif not q['options']:
                    # 客观题但选项解析为空（题面结构异常）：无法作答，留空提交防整场炸掉
                    picked = []
                    print('   答案：（选项未识别，留空提交）')
                else:
                    picked, prompt = self.__ask_llm(q['stem'], q['options'], wrong_hist[q['qid']],
                                                    qtype=q.get('qtype', ''))
                    last_prompt[q['qid']] = prompt
                    print('   答案：%s' % self.__fmt_answer(picked))
                answers[q['qid']] = picked
            if dry_run:
                self.__submit_quiz(html, page_url, questions, answers, dry_run=True)
                return False
            res = self.__submit_quiz(html, page_url, questions, answers)
            if res and res.get('status') and all(q['qtype'] == '4' for q in questions):
                # 纯简答测验：提交即达标（无客观题可判分，批阅页没有任何对错标记可解析，
                # __grade_quiz 会误判为提交失败）
                submitted_ok = True
                break
            result = self.__grade_quiz(page_url, questions, res)
            if result is None:
                # 作业提交一次即被批阅锁定，重复提交被拒：调重做接口重置后拉新题面继续
                if (not dry_run and res and not res.get('status')
                        and '已批阅' in (res.get('msg') or '') and fetch_page is not None):
                    if self.__redo_quiz(page_url):
                        html, page_url, questions = fetch_page()
                continue
            for q in questions:
                qid = q['qid']
                if result.get(qid):
                    correct[qid] = answers[qid]
                elif answers[qid] and q['qtype'] != '4':
                    # 简答题留空不参与判分，也没有自动对错标记，不记「错误答案」
                    wrong_hist[qid].append(' '.join(answers[qid]))
            # 达标口径 = 客观题全对（简答题待批阅不参与判定；纯简答测验无客观题，视同达标）
            all_right = all(result.get(q['qid']) for q in questions if q['qtype'] != '4')
            if all_right:
                print('🎉 客观题全部答对！%s' % ('（简答题待老师批阅）' if has_short else ''))
                # 实时任务点：每完成一份作业+1并即时输出总进度条
                self.pts_done = min(self.pts_done + 1, self.pts_all)
                print(self.__progress_bar(self.pts_done, self.pts_all))
                break
            if not full_score:
                # 默认不追求满分：提交一次即收工
                submitted_ok = True
                break
            print('⚠ 本轮客观题有 %d 题答错，%s' % (
                sum(1 for q in questions if q['qtype'] != '4' and not result.get(q['qid'])),
                '继续下一轮' if rnd < self.QUIZ_MAX_ROUNDS else '已达最大重试次数'))
            if rnd < self.QUIZ_MAX_ROUNDS and fetch_page is not None:
                # 刚才提交成功即被批阅锁定，下一轮提交前必须先调重做接口重置并拉新题面，
                # 否则会被「已批阅作业，不允许重复提交」拒绝
                if self.__redo_quiz(page_url):
                    html, page_url, questions = fetch_page()
        if submitted_ok:
            print('📝 已提交（默认不追求满分%s），本轮作答完成' %
                  ('，简答题留空待老师批阅' if has_short else ''))
            # 实时任务点：作业已提交，按完成计
            self.pts_done = min(self.pts_done + 1, self.pts_all)
            print(self.__progress_bar(self.pts_done, self.pts_all))
        elif not all_right:
            print('❌ 连续 %d 轮作答均未全部答对，本测验失败' % self.QUIZ_MAX_ROUNDS)
        if not submitted_ok:
            # 默认不追求满分的提交即走不落盘报告（这种场景错题详情无复盘价值；
            # 全对时报告内部也会自动跳过，实际只有 full_score 重做仍失败才写）
            self.__report_quiz_result(questions, correct, wrong_hist, chapter_name, last_prompt)
        return all_right or submitted_ok

    # 25.2、按用户格式输出题目（题干带题号，选项一行一个；判断题显示 A=对 B=错）
    @staticmethod
    def __question_lines(q):
        is_judge = any(d in ('true', 'false') for d, _ in q['options'])
        lines = [q['stem']]
        for d, text in q['options']:
            if is_judge:
                d = 'A' if d == 'true' else 'B'
            lines.append('%s. %s' % (d, text))
        if not q['options']:
            # 简答/填空类题型 与 选项解析异常的客观题 文案区分开，与答案行的提示一致
            lines.append('（简答题，不作答，留空待老师批阅）' if q['qtype'] == '4'
                         else '（选项未识别，不作答，留空待批阅）')
        return lines

    @staticmethod
    def __print_question(q):
        print('─' * 46)
        for ln in ChaoXing.__question_lines(q):
            print(ln)

    # 25.3、答案显示：判断题 true/false 转成 A/B（与选项显示一致，A=对 B=错）
    @staticmethod
    def __fmt_answer(picked):
        if not picked:
            return '（未识别出答案）'
        return ' '.join({'true': 'A', 'false': 'B'}.get(p, p) for p in picked)

    # 25.4、判分：提交成功后重拉批阅页，解析每题对错，返回 {qid: 是否正确}
    # 批阅页地址在提交响应的 url 字段里（带 submit=true 服务端才渲染成批阅版；原作答页拉不到对错标记）
    # 批阅渲染可能有延迟，没拉齐全部对错标记就等 1 秒重拉，最多 3 次
    def __grade_quiz(self, page_url, questions, submit_res):
        if not submit_res or not submit_res.get('status'):
            print('❌📝 提交未成功: %s' % (submit_res or {}).get('msg', '无响应'))
            return None
        grade_url = urljoin('https://mooc1.chaoxing.com/', submit_res.get('url') or page_url)

        def parse_marks(page):
            marks = {}
            for div in BeautifulSoup(page, 'html.parser').select('div.singleQuesId[data]'):
                qid = div.get('data')
                if div.select_one('.marking_dui') is not None:
                    marks[qid] = True
                elif div.select_one('.marking_cuo') is not None:
                    marks[qid] = False
            return marks

        result = {}
        soup = None
        for _ in range(3):
            r = self.session.get(grade_url)
            soup = BeautifulSoup(r.text, 'html.parser')
            result = parse_marks(r.text)
            if len(result) < len(questions):
                # submit=true 版会漏渲染部分对错标记（实测多选题常缺），补拉已批阅版合并
                extra = parse_marks(self.session.get(page_url).text)
                extra.update(result)  # 批阅版标记优先，已批阅版只补缺
                result = extra
            if len(result) == len(questions):
                break
            time.sleep(1)
        score_el = soup.select_one('.Finalresult i')
        n_true = sum(1 for v in result.values() if v)
        n_false = sum(1 for v in result.values() if v is False)
        # 批阅页可能漏渲染个别题的对错标记（实测多选题常缺）：未判定 = 总题数 - 已判数；
        # 未判定数恰好等于简答题数时，是等老师批阅而非漏渲染
        n_undet = len(questions) - len(result)
        n_short = sum(1 for q in questions if q['qtype'] == '4')
        if n_undet and n_undet == n_short:
            undet = '，含 %d 题简答待老师批阅' % n_short
        else:
            undet = '，未判定 %d 题按错题处理' % n_undet if n_undet else ''
        score_txt = score_el.get_text(strip=True) if score_el else ''
        print('📝 批阅: 最终成绩 %s（对 %d / 错 %d / 共 %d 题%s）' % (
            '%s 分' % score_txt if score_txt else '批阅中未出分',
            n_true, n_false, len(questions), undet))
        return result or None

    # 25.4.1、重做：作业被批阅锁定后重置为可重做状态，返回新作答页地址（题目/enc 不变）。
    # 新版作业页（isNewCeyan）没有 retest 按钮，重做入口是页面 reediter() 直接重开作答页
    # （调 /work/retest 会被拒「作答状态或作答次数已经发生变化」），优先抠它的跳转地址
    # （JS 字符串可能跨行拼接，按引号片段合并还原完整 URL）；
    # 旧版才走 /work/retest：所需字段优先从 page_url 的 query 参数取（api/work 302 后的
    # 作答页/已批阅页地址都自带全套）；URL 里没有的才从页面 hidden input 抠，且 id/name 都试、
    # id 与 value 之间允许隔其他属性（作答页写法是 <input id="classId" name="classId" value="...">，
    # 紧邻正则会漏配）
    def __redo_quiz(self, page_url):
        html = self.__risky_req('GET', page_url).text
        m = re.search(r'function\s+reediter\s*\(\s*\)\s*\{.*?location\.href\s*=\s*((?:"[^"]*"\s*\+?\s*)+);',
                      html, re.S)
        if m:
            url = ''.join(re.findall(r'"([^"]*)"', m.group(1))).replace('&amp;', '&')
            if url:
                print('📝 🔁 已通过 reEdit 入口重置为可重做状态')
                return urljoin('https://mooc1.chaoxing.com/', url)
        query = dict(parse_qsl(urlparse(page_url).query))
        fields = {}
        for fid in ('courseId', 'classId', 'workId', 'workAnswerId', 'knowledgeid',
                    'jobid', 'originJobId', 'enc', 'cpi'):
            if fid in query and query[fid]:
                fields[fid] = query[fid]
                continue
            m = re.search(r'(?:id|name)="%s"[^>]*?value="([^"]*)"' % fid, html)
            if not m:
                m = re.search(r'value="([^"]*)"[^>]*?(?:id|name)="%s"' % fid, html)
            if not m:
                print('📝 ⚠ 重做失败：页面缺少字段 %s' % fid)
                return None
            fields[fid] = m.group(1)
        r = self.__risky_req('GET', 'https://mooc1.chaoxing.com/work/retest',
                             params=dict(fields, mooc2=1, wMicroNodeId='0'))
        try:
            data = r.json()
        except ValueError:
            data = {}
        if data.get('url'):
            print('📝 🔁 已重置为可重做状态')
            return urljoin('https://mooc1.chaoxing.com/', data['url'])
        print('📝 ⚠ 重做失败：%s' % (data.get('msg') or r.text[:100]))
        return None

    # 25.5、最终报告：全部做对不输出（__do_quiz 已打 🎉）；错题详情只写文件不刷控制台，
    # 存进「错题报告」文件夹（文件名：用户名_章节名_测试.txt，重跑覆盖为最新）
    def __report_quiz_result(self, questions, correct, wrong_hist, chapter_name, last_prompt=None):
        wrong_qs = [q for q in questions if q['qid'] not in correct]
        if not wrong_qs:
            return
        lines = ['❌ 共 %d 题答错（历次错误答案如下）:' % len(wrong_qs)]
        for q in wrong_qs:
            chunk = ['─' * 46] + self.__question_lines(q)
            # 判断题历史答案 true/false 转成 A/B（与选项显示一致）
            hist = [self.__fmt_answer(h.split()) for h in wrong_hist[q['qid']]]
            chunk.append('   错误答案：%s' % ('、'.join(hist) if hist else '（无记录）'))
            lines.extend(chunk)
            if last_prompt and q['qid'] in last_prompt:
                # 最后一次提示词只写进报告（控制台不刷屏）
                lines.append('   最后一次提示词：')
                lines.extend('      ' + ln for ln in last_prompt[q['qid']].splitlines())
        os.makedirs(self.REPORT_DIR, exist_ok=True)
        safe_name = re.sub(r'[\\/:*?"<>|]', '_', chapter_name)  # 章节名可能含引号等 Windows 文件名非法字符
        fname = '%s_%s_测试.txt' % (self.realname or self.username, safe_name)
        path = os.path.join(self.REPORT_DIR, fname)
        with open(path, 'w', encoding='utf-8-sig') as f:  # utf-8-sig：记事本双击打开不乱码
            f.write('\n'.join(lines))
        print('💾 报告已保存：%s' % path)

    # ---------- 考试（exam-ans）----------

    # 27.0、滑块缺口识别引擎：utils/SliderCaptchaOcr/detector 进程内懒加载
    @classmethod
    def __get_gap_detector(cls):
        if cls.__gap_detector is None:
            det_dir = os.path.normpath(os.path.join(BASE_DIR, '..', '..', 'utils', 'SliderCaptchaOcr'))
            if det_dir not in sys.path:
                sys.path.insert(0, det_dir)
            import detector
            cls.__gap_detector = detector
        return cls.__gap_detector

    # 27.0.1、识别真缺口左缘 x（big 图坐标系）：
    # 页面渲染时在拼图当前位置画黄色高亮轮廓（假目标），真缺口是另一个暗洞：
    # 检出黄色簇时选与它距离>30px 且 conf 最高的 gap；无黄色且多 gap 时选 x 最大（实验样本验证）
    @staticmethod
    def __exam_gap_x(big_bytes, small_bytes):
        info = ChaoXing.__get_gap_detector().find_gap_info(big_bytes, small_bytes)
        gaps = (info or {}).get('gaps') or []
        gaps = [g for g in gaps if g.get('x') is not None]
        if not gaps:
            return None
        yellow_x = None
        try:
            rgb = np.asarray(Image.open(io.BytesIO(big_bytes)).convert('RGB'))
            ys, xs = np.where((rgb[:, :, 0] > 170) & (rgb[:, :, 1] > 120) & (rgb[:, :, 2] < 130))
            if len(xs) > 20:
                yellow_x = int(np.median(xs))
        except Exception:
            pass
        if yellow_x is not None:
            cands = [g for g in gaps if abs(int(g['x']) - yellow_x) > 30] or gaps
            best = max(cands, key=lambda g: g.get('conf', 0))
        elif len(gaps) > 1:
            best = max(gaps, key=lambda g: int(g['x']))
        else:
            best = gaps[0]
        return int(best['x'])

    # 27.0.2、cx_captcha JSONP 请求：响应形如 cx_captcha_function({...})，抠出括号内 JSON
    def __exam_jsonp(self, url, params, headers=None):
        r = self.session.get(url, params=params, headers=headers, timeout=15)
        txt = r.text.strip()
        a, b = txt.find('('), txt.rfind(')')
        return json.loads(txt[a + 1:b])

    # 27.1、考试滑块验证码纯协议过验（全部生成算法逆向自 load-i.min.js @124061-124354）：
    #   conf 接口拿服务器时间 t；captchaKey = md5(t+uuid4)；
    #   请求 token = md5(t+captchaId+'slide'+captchaKey)+':'+(t+300000)（5分钟有效期）；
    #   iv = md5(captchaId+'slide'+now+uuid4)，image 与 check 共用同一 iv；
    #   check 的 token 用 image 响应的大写 token；提交值 = 真缺口左缘 x（±3px 容差）；
    #   成功响应 extraData 里带 validate，拼成 validate_<captchaId>_<token>
    def __slide_captcha_validate(self, captcha_id, referer):
        headers = {'Referer': referer}
        cb = 'cx_captcha_function'
        base = 'https://captcha.chaoxing.com/captcha/v1'
        for attempt in range(1, 9):  # 单次识别非100%准，8轮叠加通过率
            try:
                conf = self.__exam_jsonp(
                    base + '/get/conf',
                    {'callback': cb, 'captchaId': captcha_id,
                     'version': self.EXAM_CAPTCHA_VERSION, '_': int(time.time() * 1000)},
                    headers)
                server_time = str(conf['t'])
                captcha_key = hashlib.md5((server_time + str(uuid.uuid4())).encode()).hexdigest()
                token_req = hashlib.md5(
                    (server_time + captcha_id + 'slide' + captcha_key).encode()).hexdigest()
                token_req += ':' + str(int(server_time) + 300000)
                iv = hashlib.md5(
                    (captcha_id + 'slide' + str(int(time.time() * 1000)) + str(uuid.uuid4())).encode()).hexdigest()
                img = self.__exam_jsonp(
                    base + '/get/verification/image',
                    {'callback': cb, 'captchaId': captcha_id, 'type': 'slide',
                     'version': self.EXAM_CAPTCHA_VERSION, 'captchaKey': captcha_key,
                     'token': token_req, 'referer': referer[:200], 'iv': iv,
                     '_': int(time.time() * 1000)}, headers)
                token = img['token']
                vo = img['imageVerificationVo']
                big = self.session.get(vo['shadeImage'], params={'t': 20703}, headers=headers, timeout=15).content
                small = self.session.get(vo['cutoutImage'], params={'t': 20703}, headers=headers, timeout=15).content
                x = self.__exam_gap_x(big, small)
                if x is None:
                    print('❌ 缺口OCR识别失败，换图重试')
                    continue
                print('🔎 缺口OCR识别：x=%s' % x)
                chk = self.__exam_jsonp(
                    base + '/check/verification/result',
                    {'callback': cb, 'captchaId': captcha_id, 'type': 'slide',
                     'token': token, 'coordinate': '[]', 'runEnv': 10,
                     'version': self.EXAM_CAPTCHA_VERSION, 't': 'a', 'iv': iv,
                     'textClickArr': json.dumps([{'x': x}], separators=(',', ':')),
                     '_': int(time.time() * 1000)}, headers)
                if chk.get('result') is True:
                    print('✅ 滑块校验已提交，验证通过')
                    return 'validate_%s_%s' % (captcha_id, token)
                print('⚠️ 校验未通过（x=%s），换图重试' % x)
            except Exception as e:
                print('❌ 滑块异常 %s，重试' % type(e).__name__)
            time.sleep(1.5)  # 轮间稍歇，连发太快不像人
        print('❌ 滑块验证失败（重试8轮未通过）')
        return None

    # 27.1.5、课程考试列表：考试 tab（exam-ans/mooc2/exam/exam-list），
    # 参数与章节页同款（courseid/clazzid/cpi/ut/t/stuenc）+ meta 页的 examEnc（实测缺它报「参数传递不正确」）；
    # 每个考试 li：onclick goTest(courseId,examId,relationId,'截止时间',paperId,...,'enc',..) +
    # p.overHidden2(考试名) + p.status(状态：待做/已完成/…)；解析失败返回空表
    def get_exam_list(self):
        params = dict(self._stu_params)
        params['examEnc'] = getattr(self, 'exam_enc', '')
        r = self.__risky_req('GET', 'https://mooc1.chaoxing.com/exam-ans/mooc2/exam/exam-list',
                             params=params,
                             headers={'Referer': 'https://mooc2-ans.chaoxing.com/'})
        soup = BeautifulSoup(r.text, 'html.parser')
        exams = []
        for li in soup.find_all('li'):
            name_el = li.select_one('p.overHidden2')
            if name_el is None:
                continue
            status_el = li.select_one('p.status')
            # goTest 参数可能在主 div（待做）或重考按钮 a（已完成，主 div 是 viewExamAnswer），逐个 onclick 找
            m = None
            for el in li.select('[onclick]'):
                m = re.search(r"goTest\('(\d+)',(\d+),(\d+),'([^']*)',(\d+),(true|false),'([^']*)'",
                              el.get('onclick', ''))
                if m:
                    break
            if m is None:
                continue
            ex = {'exam_id': m.group(2), 'relation_id': m.group(3),
                  'deadline': m.group(4),
                  'enc': m.group(7) if not m.group(7).startswith('$') else '',
                  'name': name_el.get_text(strip=True),
                  'status': status_el.get_text(strip=True) if status_el is not None else ''}
            num = li.select_one('.numspan')  # 已完成的 li 直接带最终分数（如「37分」）
            if num is not None:
                sm = re.search(r'([\d.]+)', num.get_text())
                if sm:
                    ex['score'] = float(sm.group(1))
            exams.append(ex)
        return exams

    # 27.2、进入考试：examnotes 页取参数 -> 滑块拿 validate -> examcheck 换 enc ->
    # 把 enc 填进跳转地址占位符 enc=? -> GET 答题页渲染第一题。
    # retake=True 走重考：examnotes?reset=true（服务端重置开启新一轮限时，
    # 与页面 jumpRetest() 一致）；重考要求本场已交卷且剩余重考次数>0。
    # 返回会话 dict（链三元组 lu/rt/enc + 全部入口参数），失败返回 None
    def start_exam(self, course_id, class_id, exam_id, cpi, retake=False):
        notes_url = ('https://mooc1.chaoxing.com/exam-ans/exam/test/examcode/examnotes'
                     '?courseId=%s&classId=%s&examId=%s&cpi=%s%s' % (
                         course_id, class_id, exam_id, cpi,
                         '&reset=true' if retake else ''))
        r = self.__risky_req('GET', notes_url)
        soup = BeautifulSoup(r.text, 'html.parser')

        def hid(iid):
            el = soup.find('input', id=iid)
            return el.get('value', '') if el else None

        answer_id = hid('answerId')
        start_btn = soup.find(id='startBtn')
        if not answer_id or start_btn is None:
            print('❌ examnotes 页解析失败（未登录/考试不存在/已用完次数）: %s' % r.url)
            return None
        # 跳转地址在 href 里（enc=? 是占位符，openc 服务端已渲染好）
        jump_tpl = urljoin(notes_url, start_btn.get('data') or start_btn.get('href', ''))
        openc = dict(parse_qsl(urlparse(jump_tpl).query)).get('openc', '')

        validate = ''
        if hid('captchaCheck') == '1':
            captcha_id = hid('captchaCaptchaId') or ''
            print('🧩 需要滑块验证码，开始协议过验...')
            validate = self.__slide_captcha_validate(captcha_id, notes_url)
            if not validate:
                return None
        # examcheck：captchavalidate 换 enc（enc 对同一考试是确定性值，重复进入不变）
        qs = {'view': 'json', 'answerId': answer_id, 'examId': exam_id, 'classId': class_id,
              'courseId': course_id, 'cpi': cpi, 'code': '', 'sdlkey': '', 'facekey': '',
              'captchavalidate': validate, '_signcode': ''}
        r2 = self.__risky_req('GET', 'https://mooc1.chaoxing.com/exam-ans/exam/test/examcheck',
                              params=qs, headers={'Referer': notes_url,
                                                  'X-Requested-With': 'XMLHttpRequest'})
        try:
            data = r2.json()
        except ValueError:
            data = {}
        if not data.get('status'):
            print('❌ examcheck 未通过: %s' % (data.get('msg') or r2.text[:150]))
            return None
        enc = data.get('enc', '')
        # 重考（reset=true）的真实重置动作：说明页只是进入重考模式，答卷状态要靠
        # reVersionReTest 重置（与页面 jumpExam -> reTestAction 一致），漏了这步
        # 答题页会渲染「考试已经提交」提示页
        if retake:
            rr = self.__risky_req('GET', 'https://mooc1.chaoxing.com/exam-ans/exam/test/reVersionReTest',
                                  params={'courseId': course_id, 'classId': class_id,
                                          'tId': exam_id, 'id': answer_id},
                                  headers={'Referer': notes_url, 'X-Requested-With': 'XMLHttpRequest'})
            try:
                rd = rr.json()
            except ValueError:
                rd = {}
            if not rd.get('status'):
                print('❌ 重考重置失败: %s' % (rd.get('msg') or rr.text[:120]))
                return None
            print('🔄 已重置答卷，开新一轮限时')
        jump_url = jump_tpl.replace('enc=?', 'enc=' + enc)
        # 服务端「10 分钟内禁止交卷」计时起点接近 startTest，进入时刻记下来供交卷等待用
        enter_time = time.time()
        # 进入答题页，渲染第一题（页面自带链三元组 enc/remainTime/encLastUpdateTime）
        # 走 __risky_req：考试请求同样会撞 9010 风控，命中后自动过码重发
        r3 = self.__risky_req('GET', jump_url, headers={'Referer': notes_url})
        page = self.__parse_exam_page(r3.text, jump_url)
        if not page or not page.get('qid'):
            print('❌ 答题页拉取失败: %s' % r3.url)
            return None
        sess = {'course_id': course_id, 'class_id': class_id, 'exam_id': exam_id,
                'cpi': cpi, 'openc': openc, 'lu': page['encLastUpdateTime'],
                'rt': page['remainTime'], 'enc': page['enc'], 'referer': jump_url,
                'total': page.get('total') or 0, 'answer_id': answer_id,
                'enter_time': enter_time}
        print('✍️ 考试开始作答')
        print('   📄 共 %s 题，剩余 %s 秒' % (sess['total'] or '?', page['remainTime']))
        sess['first_page'] = page
        return sess

    # 27.3、解析题面页：只取 #submitTest 表单内带 name 的 hidden（与浏览器 serialize 完全一致），
    # 题干/选项顺带解出；判断题选项 data 是 true/false，选择/多选是字母
    @staticmethod
    def __parse_exam_page(html, page_url):
        soup = BeautifulSoup(html, 'html.parser')
        form = soup.find('form', id='submitTest')
        if form is None:
            return None
        fields = []
        for inp in form.find_all('input'):
            nm = inp.get('name')
            if nm:
                fields.append((nm, inp.get('value', '')))
        fd = dict(fields)
        qid = fd.get('questionId')
        if not qid:
            return None
        h3 = soup.select_one('h3.mark_name')
        stem = h3.get_text(' ', strip=True) if h3 is not None else ''
        options = []
        for span in soup.select('span[data][qid]'):
            p = span.find_next('div', class_='answer_p')
            options.append((span.get('data', ''), p.get_text(' ', strip=True) if p is not None else ''))
        total_m = re.search(r'题量:\s*(\d+)', html)
        return {'page_url': page_url, 'fields': fields, 'fd': fd, 'qid': qid,
                'qtype': fd.get('type' + qid, ''),
                'type_name': fd.get('typeName' + qid, ''),
                'stem': stem, 'options': options,
                'total': int(total_m.group(1)) if total_m else 0,
                'remainTime': fd.get('remainTime', ''),
                'encLastUpdateTime': fd.get('encLastUpdateTime', ''),
                'enc': fd.get('enc', '')}

    # 27.4、翻题拉取：与页面 getTheNextQuestion 一致——
    # URL 带上一题保存响应的三元组（remainTimeParam/relationAnswerLastUpdateTime/enc），
    # 服务端按 start 渲染第 start+1 题；页面级校验失败会返回 1KB 错误页，此时置 None 断链
    def __fetch_exam_question(self, sess, start):
        url = ('https://mooc1.chaoxing.com/exam-ans/exam/test/reVersionTestStartNew'
               '?keyboardDisplayRequiresUserAction=1&getTheNextQuestion=1'
               '&courseId=%(course_id)s&classId=%(class_id)s&tId=%(exam_id)s&p=1&start=%(start)s'
               '&remainTimeParam=%(rt)s&relationAnswerLastUpdateTime=%(lu)s&enc=%(enc)s'
               '&monitorStatus=0&monitorOp=-1&examsystem=0&qbanksystem=0&qbankbackurl='
               '&cpi=%(cpi)s&openc=%(openc)s&newMooc=true&webSnapshotMonitor=0') % dict(
            sess, start=start)
        r = self.__risky_req('GET', url, headers={'Referer': sess['referer']})
        page = self.__parse_exam_page(r.text, url)
        if page:
            sess['referer'] = url
        return page

    # 27.5、保存/交卷：POST reVersionSubmitTestNew，URL 参数拼 tempSave + pos + version=1
    # （pos 行为签名字段，逆向确认 try-catch 包裹可为空，实测空 pos 保存成功）；
    # 表单沿用题面页 hidden 现值，仅覆盖 tempSave 与 answer<qid>；
    # 保存响应 data = lastUpdateTime|remainTime|enc，作为下一题拉取链；
    # tempSave=False 时响应 status 变为 submitted/forceSubmitted，url 字段是批阅页
    def __save_exam_question(self, sess, page, answer, temp_save=True):
        fd = page['fd']
        url = ('https://mooc1.chaoxing.com/exam-ans/exam/test/reVersionSubmitTestNew'
               '?classId=%s&courseId=%s&testPaperId=%s&testUserRelationId=%s'
               '&cpi=%s&tempSave=%s&pos=&version=1') % (
                  fd.get('classId', sess['class_id']), fd.get('courseId', sess['course_id']),
                  fd.get('testPaperId', sess['exam_id']), fd.get('testUserRelationId', ''),
                  fd.get('cpi', sess['cpi']), 'true' if temp_save else 'false')
        data = []
        for k, v in page['fields']:
            if k == 'tempSave':
                v = 'true' if temp_save else 'false'
            elif answer is not None and k in ('answer' + page['qid'], 'answers' + page['qid']):
                # 答案字段名随题型不同：单选/判断 name=answer<qid>（单数），
                # 多选页 hidden id=answer<qid> 但 name=answers<qid>（复数，
                # 与 JS addMultipleChoice 按 id 写入一致），两个都要匹配，
                # 否则多选答案覆盖不生效，提交原空值但接口仍返 success
                v = answer
            data.append((k, v))
        r = self.__risky_req('POST', url, data=data, headers={
            'Referer': page['page_url'],
            'X-Requested-With': 'XMLHttpRequest',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        })
        try:
            resp = r.json()
        except ValueError:
            resp = {'status': 'http%s' % r.status_code, 'msg': r.text[:150]}
        d = (resp.get('data') or '').split('|')
        if resp.get('status') == 'success' and len(d) == 3:
            sess['lu'], sess['rt'], sess['enc'] = d[0], d[1], d[2]
        return resp

    # 27.6、考试主流程：进入 -> 逐题 LLM 作答保存 -> 答完自动交卷 + 错题报告。
    # 限时从首次进入起算（如 90 分钟），耗尽后服务端仍放行进入但拒绝保存；
    # 此时自动走重考（examnotes?reset=true，服务端重置开新一轮限时），
    # 一次运行最多重考 MAX_RETAKE 次，重考的首次进入必须从重考入口开始
    MAX_RETAKE = 3

    # first_retake=True：首轮进入就走重考入口（reset=true）——
    # 已交卷但不满分重考刷分的场景，正常入口已不是答题页
    def finish_exam(self, course_id, class_id, exam_id, cpi, first_retake=False):
        for attempt in range(self.MAX_RETAKE + 1):
            retake = first_retake or attempt > 0
            sess = self.start_exam(course_id, class_id, exam_id, cpi, retake=retake)
            if sess is None:
                return False
            page = sess.pop('first_page')
            if int(page.get('remainTime') or 0) < 60:
                if attempt < self.MAX_RETAKE:
                    print('⚠ 限时已耗尽（剩 %s 秒），自动重考（第 %d/%d 次）' % (
                        page.get('remainTime'), attempt + 1, self.MAX_RETAKE))
                    time.sleep(1)
                    continue
                print('⚠ 限时耗尽，%d 次重考机会也已用完，本场放弃' % self.MAX_RETAKE)
                return False
            self.__do_exam(sess, page)
            if getattr(self, 'no_submit', False):
                # 只做题不交卷：成绩未出，不查批阅页；日志照常落盘（未交卷视为不满分）
                self.__exam_log_flush(sess, None)
                return True
            # 考完输出成绩进度条（批阅中可能未出）
            score, left = self.__exam_result_info(sess)
            if score is None:
                print('✍️ 成绩批阅中，稍后可在考试列表查看')
            else:
                print(self.__exam_score_line(score, left))
            # save_log=True：不满分（含批阅中）才落盘答题日志，满分丢弃
            self.__exam_log_flush(sess, score)
            return True
        return False

    # 27.6.0、单轮作答：逐题 LLM 作答保存 -> 答完自动交卷（交卷才有成绩，
    # 不交卷要白等限时耗尽才自动收卷）。
    # 保存失败自动重试（同题同答案最多发3次）；中途断链不交卷——已保存的答案
    # 仍在服务端，重新运行可从当前进度续作，避免剩下整卷空白
    def __do_exam(self, sess, page):
        total = sess['total'] or 0
        idx = 0
        last_resp = None
        completed = False
        while page is not None:
            idx += 1
            self.__print_question(page)
            picked, prompt = self.__ask_llm(page['stem'], page['options'],
                                            qtype=page.get('qtype', ''),
                                            type_name=page.get('type_name', ''))
            answer = ''.join(sorted(picked))  # 多选按字母序提交（与服务端 sortMultiAnswer 一致）
            print('   答案：%s' % self.__fmt_answer(picked))
            if getattr(self, 'save_log', False):
                self.__exam_log_collect(sess, page, picked, prompt)
            resp = None
            for retry in range(3):  # 保存重试：同题同答案最多发 3 次
                try:
                    resp = self.__save_exam_question(sess, page, answer, temp_save=True)
                except requests.RequestException as e:
                    resp = None
                    print('   ⚠ 保存请求异常 %s（第 %d/3 次）' % (type(e).__name__, retry + 1))
                if resp is not None and resp.get('status') == 'success':
                    break
                msg = str((resp or {}).get('msg', ''))
                if '时间已用完' in msg or '已提交' in msg:
                    break  # 确定性失败（限时耗尽/已交卷），重试无意义
                if retry < 2:
                    time.sleep(2)
            if resp is None or resp.get('status') != 'success':
                print('   ❌ 保存失败: %s' % ((resp or {}).get('msg') or '请求异常'))
                break
            print('   ✅ 已保存（剩余 %s 秒）' % sess['rt'])
            last_resp = resp
            if total and idx >= total:
                completed = True
                break
            time.sleep(0.8)  # 节流：翻题请求贴着保存发容易触发风控
            page = self.__fetch_exam_question(sess, idx)
            if page is None:
                print('⚠ 第 %d 题拉取失败，停止翻题' % (idx + 1))
                break
        print('✍️ 作答完成：%d 题（共 %s 题）' % (idx, total or '?'))
        if completed or not total:
            # 正常答满才交卷；题量未知时保守视为完成
            if getattr(self, 'no_submit', False):
                print('✈️ 已答完但未交卷：答案已保存在服务端，可自行检查后手动交卷')
            elif last_resp is not None:
                self.__submit_exam(sess, page, idx)
            else:
                print('⚠ 一题都没保存成功，跳过交卷')
        elif last_resp is not None:
            print('⚠ 中途断链未交卷：已保存的答案仍在，重新运行可从当前进度续作')
        return True

    # 27.6.1、交卷：对当前题再发一次 tempSave=false 即服务端交卷；交卷不可逆。
    # 服务端限制进入考试 10 分钟内禁止交卷（实测报「限时提交：考试10分钟内不允许
    # 提交考试」）：命中后等到进入后 605 秒再交（晚几秒防边界失败），等待后重新拉
    # 当前题刷新链三元组再提交；仍失败再等 60 秒重试，最多 3 次，不高频提交。
    # 批阅页不渲染逐题对错（实证 look 页只有成绩与重考入口），无考试错题报告；
    # 最终成绩/剩余重考次数由 finish_exam 尾部 __exam_result_info 输出
    def __submit_exam(self, sess, page, qidx):
        if page is None:
            print('❌ 无题面可交卷')
            return False
        print('📤 正在交卷...')
        msg = ''
        for attempt in range(3):
            resp = self.__save_exam_question(sess, page, None, temp_save=False)
            # 成功判定兼容两种响应：status=submitted/forceSubmitted，
            # 或 msg 直接是「提交成功」（实测同一服务端两种结构都出过，
            # 只认 status 会把成功的交卷误判成失败）
            if (str(resp.get('status')) in ('submitted', 'forceSubmitted')
                    or '提交成功' in str(resp.get('msg') or '')):
                print('✅ 交卷成功')
                return True
            msg = str(resp.get('msg') or resp.get('status') or '')
            if '10分钟' not in msg:
                break
            wait = 605 - (time.time() - sess.get('enter_time', time.time()))
            if wait <= 0:
                wait = 60  # 本地起点晚于服务端计时起点时兜底再等一轮
            print('⏳ 服务端限制进入 10 分钟内禁止交卷，等待 %d 秒后自动交卷' % int(wait))
            # 挂机等待期间每 60 秒报一次剩余（节奏与刷视频 60 秒心跳输出一致）；
            # 触发按进度差算并锚记已等时长，sleep 粒度/输出行耗时不会累积漂移
            start_wait = time.time()
            last_mark = 0.0
            while True:
                elapsed = time.time() - start_wait
                remain = wait - elapsed
                if remain <= 0:
                    break
                if elapsed - last_mark >= 60:
                    print('   ⏳ 剩余 %d 秒' % int(remain))
                    last_mark = elapsed
                time.sleep(1)
            fresh = self.__fetch_exam_question(sess, qidx - 1)  # 链三元组已过期，重新拉当前题刷新
            if fresh is not None:
                page = fresh
        print('❌ 交卷未成功: %s' % msg)
        return False

    # 统一章节行：✅/❌ + 章节名 + （已完成 / 差 N 个任务点）；progress 带 id，main 不带
    def __chapter_line(self, task_id, name, cnt, with_id=True):
        head = '[%s] ' % task_id if with_id else ''
        tail = '已完成' if cnt == 0 else '差 %d 个任务点' % cnt
        return '%s %s%s（%s）' % ('✅' if cnt == 0 else '❌', head, name, tail)

    # 26、查课程完成进度：show_detail=False 只列章节列表；True 带任务点状态和总进度。
    # 全部数据来自章节树（studentcourse 页一次请求全有，零逐章请求）：
    # 无待完成→✅，待完成→❌，括号里统一「差 N 个任务点」且各列上下对齐；
    # 总进度条用页面头部「已完成任务点: x/y」
    @classmethod
    def progress(cls, username, password, index, show_detail=False):
        user = cls(username, password)
        user.login()
        user.get_course_list()
        response = user.get_course(index - 1)
        user.get_capter_list(response)
        # 章节行/进度条/考试列表统一全量输出（快速模式已废弃，show_detail 仅保留兼容旧调用）
        print('📚 课程章节列表（共%d章）:' % len(user.capter_list))
        for task_id, name in user.capter_list:
            print(user.__chapter_line(task_id, name,
                                      user.chapter_status.get(str(task_id), 0)))
        pts_done, pts_total = user.course_pts or (0, 0)
        print(user.__progress_bar(pts_done, pts_total, label='🔴 任务点'))
        user.__print_exam_list()

    # courses：列出账号下全部课程（i.chaoxing.com/base 课程页数据链路），无需 index 参数；
    # 行首数字即 main/progress 的 index，指定刷哪门课时直接抄；同时返回 [(index, 课程名), ...]
    @classmethod
    def courses(cls, username, password):
        user = cls(username, password)
        user.login()
        user.get_course_list()
        print('📚 课程列表（共%d门）:' % len(user.course_list))
        courses = []
        for index, (title, _) in enumerate(user.course_list, 1):
            print('%d. %s' % (index, title))
            courses.append((index, title))
        return courses

    # 考试列表 UI：✍️ 标题 + 每场一行 ✅/❌ [examId] 名称（状态/上次成绩[，不满分待重考]，截止 …）；
    # progress 与 main 开头共用，保证两侧输出一致；无考试时不出考试段
    def __print_exam_list(self):
        try:
            exams = self.get_exam_list()
        except Exception as e:
            print('✍️ 考试列表获取失败: %s' % type(e).__name__)
            return
        if not exams:
            return
        print('✍️ 考试列表（共%d场）:' % len(exams))
        for ex in exams:
            done = '完成' in ex['status'] or '批阅' in ex['status']
            score = ex.get('score')
            # 不满分的已完成考试视同待做（❌ 重考刷分），与 __run_pending_exams 判定一致
            retake = done and score is not None and score < self.EXAM_FULL_SCORE
            seg = ['上次成绩 %s 分' % score if done and score is not None
                   else (ex['status'] or '未知')]
            if retake:
                seg.append('不满分待重考')
            seg.append('截止 %s' % (ex['deadline'] or '')[:16])
            print('%s [%s] %s（%s）' % (
                '✅' if done and not retake else '❌',
                ex['exam_id'], ex['name'], '，'.join(seg)))

    # 任务点计数：按 mode 统计 (已完成数, 总数)。1个视频/1个文档/1个测验各算1个任务点：
    # 视频 isPassed / 文档测验 job!=true 视为完成（与 finish_* 的跳过判定一致）；
    # 类型归属与 pending 判定同口径：视频+图文归 all/watch/course，测验归 all/test/course
    @staticmethod
    def __count_pts(vids, docs, quizzes, mode):
        n_done = n_all = 0
        if mode in ('all', 'watch', 'course'):
            n_all += len(vids)
            n_done += sum(1 for a in vids if a.get('isPassed'))
            n_all += len(docs)
            n_done += sum(1 for a in docs if a.get('job') is not True)
        if mode in ('all', 'test', 'course'):
            n_all += len(quizzes)
            n_done += sum(1 for a in quizzes if a.get('job') is not True)
        return n_done, n_all

    # 27.1.6、已交卷考试的最终成绩：look 批阅页「最终成绩 X 分」（= 历次最高分）。
    # 用于不满分重考判定；页面未渲染出成绩（刚交卷/批阅中）返回 None
    def __get_exam_score(self, ex):
        url = ('https://mooc1.chaoxing.com/exam-ans/exam/test/look'
               '?courseId=%s&classId=%s&examId=%s&examAnswerId=%s&cpi=%s' % (
                   self.courseid, self.clazzid, ex['exam_id'], ex['relation_id'], self.cpi))
        r = self.__risky_req('GET', url, headers={'Referer': 'https://mooc1.chaoxing.com/'})
        m = re.search(r'最终成绩<b[^>]*>([\d.]+)</b>分', r.text)
        return float(m.group(1)) if m else None

    # 27.1.7、考完收尾信息：look 批阅页一次拿最终成绩与剩余重考次数
    # （「最终成绩X分」「允许重考N次，已重考M次」，aria-label 与 span 分隔两种形态都匹配；
    # 未交卷时成绩未出为 None，剩余次数照常可取）
    def __exam_result_info(self, sess):
        url = ('https://mooc1.chaoxing.com/exam-ans/exam/test/look'
               '?courseId=%s&classId=%s&examId=%s&examAnswerId=%s&cpi=%s' % (
                   self.courseid, self.clazzid, sess['exam_id'], sess['answer_id'], self.cpi))
        r = self.__risky_req('GET', url, headers={'Referer': 'https://mooc1.chaoxing.com/'})
        sm = re.search(r'最终成绩<b[^>]*>([\d.]+)</b>分', r.text)
        rm = re.search(r'允许重考(?:<[^>]*>)?(\d+)(?:</[^>]*>)?次，已重考(?:<[^>]*>)?(\d+)', r.text)
        return (float(sm.group(1)) if sm else None,
                int(rm.group(1)) - int(rm.group(2)) if rm else None)

    # 成绩行：满分 🎉（与测验「🎉 本轮全部答对！」一致），不满分 🎯；
    # %s 保序 float 自带小数位（37.0），未出成绩不走这里
    @classmethod
    def __exam_score_line(cls, score, left):
        if score >= cls.EXAM_FULL_SCORE:
            return '🎉 考试成绩：%s 分（满分）' % score
        tail = '' if left is None else '（剩余重考 %d 次）' % left
        return '🎯 考试成绩：%s 分%s' % (score, tail)

    # 27.6.2、答题日志（save_log=True）：作答时逐题把题目+答案收进内存，考完拿到成绩
    # 后才决定落盘——不满分（含批阅中未出分）才写入 考试答题日志/<exam_id>_<时间戳>.txt，
    # 满分直接丢弃（日志用于复盘没答好的题，满分没有复盘价值）；文件名带时间戳，
    # 每轮不满分独立存档，历次重考互不覆盖，文件头带本轮成绩；只记题目相关内容
    # （题干/选项/答案），提示词是否一并写入跟随 show_prompt（与终端打印同一条件）
    def __exam_log_collect(self, sess, page, picked, prompt):
        lines = ['─' * 46]
        lines += self.__question_lines(page)
        lines.append('答案：%s' % self.__fmt_answer(picked))
        if getattr(self, 'show_prompt', False) and prompt:
            lines += ['【提示词】', prompt]
        sess.setdefault('log_lines', []).append('\n'.join(lines))

    # 落盘：文件名带时间戳（每轮一个新文件，历次重考/续作互不覆盖）；
    # 落盘后终端输出一行提示，让保存动作本身有输出反馈
    def __exam_log_flush(self, sess, score):
        if not getattr(self, 'save_log', False):
            return
        lines = sess.pop('log_lines', None)
        if not lines:
            return
        if score is not None and score >= self.EXAM_FULL_SCORE:
            return
        if not os.path.isdir(self.EXAM_LOG_DIR):
            os.makedirs(self.EXAM_LOG_DIR)
        path = os.path.join(self.EXAM_LOG_DIR, '%s_%s.txt' % (
            sess['exam_id'], time.strftime('%Y%m%d_%H%M%S')))
        with open(path, 'w', encoding='utf-8') as f:
            f.write('===== 成绩 %s =====\n' % (
                '批阅中' if score is None else '%s 分' % score))
            f.write('\n'.join(lines) + '\n')
        print('💾 答题日志已保存: %s' % path)

    # 27.7、跑全部待做考试：未做的直接做，已完成但不满分的重考刷分（取最高成绩规则）；
    # 自动作答保存并答完自动交卷；main 的 exam 不传 exam_id 与 all 尾部共用
    def __run_pending_exams(self):
        try:
            exams = self.get_exam_list()
        except Exception as e:
            print('✍️ 考试列表获取失败: %s' % type(e).__name__)
            return
        todo = []
        for ex in exams:
            st = ex['status']
            if '完成' not in st and '批阅' not in st:
                todo.append(ex)  # 未做：直接做
                continue
            # 已完成：不满分也算待做，重考刷分（取最高成绩规则，不会更差）；
            # 分数优先用列表自带，缺失才拉批阅页补
            score = ex.get('score')
            if score is None:
                score = self.__get_exam_score(ex)
            if score is None:
                continue  # 成绩未出（刚交卷/批阅中），不盲考
            if score < self.EXAM_FULL_SCORE:
                ex['score'] = score
                todo.append(ex)
        if not todo:
            print('✍️ 没有待做的考试')
            return
        for ex in todo:
            head = '🚀 进入考试 %s' % ex['name']
            if 'score' in ex:
                head += '（上次成绩 %s 分，不满分待重考）' % ex['score']
            else:
                head += '（截止 %s）' % (ex['deadline'] or '')[:16]
            print(head)
            try:
                self.finish_exam(self.courseid, self.clazzid, ex['exam_id'], self.cpi,
                                 first_retake='score' in ex)  # 带分数=已完成重考，首轮即走重考入口
            except Exception as e:
                # 单场考试意外炸了只放弃这一场，剩余考试继续
                print('❌ 考试 %s 异常中断: %s: %s' % (ex['name'], type(e).__name__, e))

    # main：index 是第几个课程；chapter_id 传了就只做这个章节，不传默认全部做；
    # mode='all' 全流程（默认，章节任务 + 待做考试）/ 'course' 只刷课程（视频+图文+测验，不碰考试）
    # / 'watch' 视频+图文（测验/考试不碰）/ 'test' 只刷测验 / 'exam' 只跑考试
    # （exam_id 传了只做这一场，不传则把所有待做的考试都做一遍）。
    # full_score=False（默认）测验客观题答完提交一次即收工（简答题一律不作答留空待批阅）；
    # True 时客观题未全对自动重做，最多 QUIZ_MAX_ROUNDS 轮争取满分
    # 答完自动交卷（交卷才有成绩，等超时收卷要白等限时）；限时耗尽自动重考（一次运行最多 3 次）
    # 扫描阶段零逐章请求：全部数据来自章节树（每章待完成任务点数+页面头部总进度），
    # 待完成章进入后才拉任务明细，输出清单并逐个执行
    @classmethod
    def main(cls, username, password, index, chapter_id=None, mode='all', exam_id=None,
             show_prompt=False, save_log=False, no_submit=False, full_score=False):
        user = cls(username, password)
        user.login()
        user.get_course_list()
        response = user.get_course(index - 1)
        user.get_capter_list(response)
        # show_prompt=True 时每题打印发给 LLM 的完整提示词（测验/考试共用 __ask_llm）
        user.show_prompt = show_prompt
        # save_log=True 时考试题目存日志（不满分才写入，提示词跟随 show_prompt）
        user.save_log = save_log
        # no_submit=True 时只作答保存不交卷（答案留在服务端，可自行检查后手动交）
        user.no_submit = no_submit
        if mode == 'exam':
            if exam_id:
                # 单场模式：先查列表带上次成绩，已交卷且不满分直接从重考入口进
                try:
                    ex = next((e for e in user.get_exam_list()
                               if e['exam_id'] == str(exam_id)), None)
                except Exception:
                    ex = None
                if ex:
                    head = '🚀 进入考试 %s' % ex['name']
                    if ex.get('score') is not None:
                        head += '（上次成绩 %s 分）' % ex['score']
                    print(head)
                user.finish_exam(
                    user.courseid, user.clazzid, exam_id, user.cpi,
                    first_retake=bool(ex and ex.get('score') is not None
                                      and ex['score'] < user.EXAM_FULL_SCORE))
            else:
                # 不传 exam_id：把所有待做的考试都做一遍
                user.__run_pending_exams()
            return

        # 实时任务点计数以章节树基线初始化，任务点完成时在 finish_* 内部+1并即时输出进度条
        user.pts_done, user.pts_all = user.course_pts or (0, 0)
        print('📚 课程章节列表（共%d章）:' % len(user.capter_list))
        targets = []
        for task_id, name in user.capter_list:
            if chapter_id is not None and str(task_id) != str(chapter_id):
                continue
            cnt = user.chapter_status.get(str(task_id), 0)
            print(user.__chapter_line(task_id, name, cnt))
            if cnt == 0:
                continue
            targets.append((task_id, name))
        print(user.__progress_bar(user.pts_done, user.pts_all))
        user.__print_exam_list()
        print('开始刷剩余%d章...' % len(targets))

        for task_id, name in targets:
            print('👤 当前用户：%s' % (user.realname or user.username))
            print('🚀 进入章节 %s' % name)
            try:
                vids, docs, quizzes = user.__split_tasks(user.__get_marg(task_id))
                if not vids and not docs and not quizzes:
                    print('   ➖ 无任务')
                    continue
                watch_pending = mode in ('all', 'watch', 'course') and any(not a.get('isPassed') for a in vids)
                doc_pending = mode in ('all', 'watch', 'course') and any(a.get('job') is True for a in docs)
                quiz_pending = mode in ('all', 'test', 'course') and any(a.get('job') is True for a in quizzes)
                if mode != 'all':
                    # 非 all 模式总进度只统计对应类型任务点，进章后把该章总数累计进基线
                    # （完成数由任务点完成时即时+1，这里不再累计）
                    _, n_all = user.__count_pts(vids, docs, quizzes, mode)
                    user.pts_all += n_all
                if not watch_pending and not doc_pending and not quiz_pending:
                    print('   （无待完成任务，跳过）')
                    continue
                print('   📋 任务清单：')
                print(user.__task_list_line(vids, docs, quizzes, mode))
                if watch_pending:
                    user.finish_video(task_id)
                if doc_pending:
                    user.finish_document(task_id)
                if quiz_pending:
                    user.finish_quiz(task_id, full_score=full_score)
            except Exception as e:
                # 长挂机一章上百个请求，单章意外炸了只放弃本章（下次运行自动续），
                # 剩余章节照常推进——以前这里一炸整个脚本就退出
                print('   ❌ 章节任务异常中断（跳到下一章）: %s: %s' % (type(e).__name__, e))
                continue
            # 刷完重拉服务端最新状态校正实时计数：all 模式重拉章节树头部基线（权威值）；
            # 非 all 模式任务点完成时已即时+1，无需再拉
            pts_before = (user.pts_done, user.pts_all)
            try:
                if mode == 'all':
                    time.sleep(1)  # 服务端落库延迟，立即重拉可能还是旧值
                    user.refresh_chapter_status()
                    if user.course_pts:
                        srv_done, user.pts_all = user.course_pts
                        # done 只增不减：含简答题的测验提交后要等老师批阅，批阅前服务端
                        # 不计完成，权威值会小于本地实时计数，直接采纳会让进度条倒退
                        user.pts_done = max(user.pts_done, srv_done)
            except Exception as e:
                print('   ⚠ 进度刷新失败: %s: %s' % (type(e).__name__, e))
            if (user.pts_done, user.pts_all) != pts_before:
                # 权威值与本地实时计数有偏差（如服务端异步补录）才补一条进度条，一致则不重复输出
                print(user.__progress_bar(user.pts_done, user.pts_all))

        # 章节完处理考试（all 模式全流程含考试；course/watch/test 只管章节）
        if mode == 'all':
            user.__run_pending_exams()


if __name__ == '__main__':
    # 用法1（刷课）: python chaoxing.py <用户名> <密码> <index>
    # 用法2（考试）: python chaoxing.py <用户名> <密码> <index> <exam_id>
    #   exam_id 是考试地址里 examId= 后那串数字；答完自动交卷
    # 用法3（课程列表）: python chaoxing.py <用户名> <密码> ls
    #   列出账号下全部课程，行首序号即 main/progress 的 index
    # 想看发给 LLM 的提示词加 show_prompt=True（测验/考试都生效）
    # 考试题目存日志加 save_log=True（不满分才写入 考试答题日志/<exam_id>_<时间戳>.txt）
    # 只做题不交卷加 no_submit=True（答案保存在服务端，可自行检查后手动交卷）
    if len(sys.argv) == 4 and sys.argv[3] == 'ls':
        ChaoXing.courses(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        ChaoXing.main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        pass
