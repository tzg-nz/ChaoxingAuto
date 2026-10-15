# -*- coding: utf-8 -*-
"""cxsecret 字体解密准确性自检。

两种模式：

1) 离线自检（不需要账号、不联网，用仓库自带的真实密文字体样本）：

       python test_decode.py --offline

   样本来自课程「10月」预防校园暴力、心理健康 1.1 章测验的真实加密题面，
   tests/cxsecret_expected.json 里的期望映射已用批阅页明文逐题核对过（全对）。
   这一项验证字形匹配算法本身没被改坏。

2) 在线端到端对照（真实账号、真实题面）：

       python test_decode.py <账号> <密码>
       python test_decode.py <账号> <密码> <courseid>
       python test_decode.py <账号> <密码> <courseid> <chapterId>

   原理：**已批阅**的测验页是明文（不带加密字体），正好当标准答案；调脚本自带的
   重做接口把它重置后，服务端返回的就是加密题面，于是同一份卷子可以做
   「加密题面解码结果」vs「批阅页明文原文」的逐题比对——这是最硬的验证。

   ⚠️ 会重置被测的那一份测验（答案清空、变回未做状态），但不会提交任何答案。
   测完想恢复，正常跑一次刷课/测验流程重新作答即可。
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from bs4 import BeautifulSoup  # noqa: E402

from chaoxing import ChaoXing  # noqa: E402


def offline():
    """离线：用样本字体跑字形匹配，与已核对过的期望映射逐条比对。"""
    font_path = os.path.join(BASE_DIR, 'tests', 'cxsecret_font.bin')
    exp_path = os.path.join(BASE_DIR, 'tests', 'cxsecret_expected.json')
    if not (os.path.exists(font_path) and os.path.exists(exp_path)):
        print('❌ 缺少样本文件（tests/cxsecret_font.bin 或 cxsecret_expected.json）')
        return 1
    import json
    expected = json.load(open(exp_path, encoding='utf-8'))
    font = open(font_path, 'rb').read()
    base = os.path.join(BASE_DIR, 'fonts', 'SourceHanSansCN-Normal.otf')
    if not os.path.exists(base):
        print('❌ 缺少基准字体 %s（首次运行脚本会自动下载）' % base)
        return 1

    print('🔤 离线自检：样本字体 %d 字节，期望映射 %d 条' % (len(font), len(expected)))
    got = ChaoXing._ChaoXing__auto_match_secret(font, base)
    print('🔤 实际匹配 %d 条' % len(got))
    bad = {k: (expected[k], got.get(k)) for k in expected if got.get(k) != expected[k]}
    extra = {k: v for k, v in got.items() if k not in expected}
    if not bad and not extra:
        print('✅ 全部 %d 个密字识别正确（与批阅页明文核对过的期望值完全一致）' % len(expected))
        return 0
    for k, (e, g) in bad.items():
        print('   ❌ %s(U+%04X): 期望=%s 实际=%s' % (k, ord(k), e, g))
    for k, v in extra.items():
        print('   ⚠️ 多出 %s(U+%04X) -> %s' % (k, ord(k), v))
    print('❌ 离线自检未通过：错 %d 条，多 %d 条' % (len(bad), len(extra)))
    return 1


def _stems(html):
    out = []
    for t in BeautifulSoup(html, 'html.parser').select('div.TiMu'):
        el = t.select_one('.Zy_TItle')
        if el is not None:
            out.append(el.get_text(' ', strip=True))
    return out


def online(username, password, courseid=None, chapter_id=None):
    """在线：批阅页明文当真值 -> 重做拿加密题面 -> 解码 -> 逐题对照。"""
    user = ChaoXing(username, password)
    user.login()
    user.course_list = []
    user.get_course_list()
    if not user.course_list:
        print('❌ 课程列表为空（风控或页面结构变化），稍后重试')
        return 1

    idx = None
    for i in range(1, len(user.course_list) + 1):
        user.get_course(i - 1)
        if courseid is None or str(user.courseid) == str(courseid):
            idx = i
            break
    if idx is None:
        print('❌ 没找到课程 %s' % courseid)
        return 1
    print('📚 课程 idx=%s [%s] courseid=%s' % (idx, user.course_name, user.courseid))
    user.get_capter_list(user.get_course(idx - 1))

    target = None
    for task_id, name in user.capter_list:
        if chapter_id is not None and str(task_id) != str(chapter_id):
            continue
        marg = user._ChaoXing__get_marg(task_id)
        if not marg:
            continue
        for a in marg.get('attachments') or []:
            if a.get('type') == 'workid' and 'jobid' in a:
                target = (task_id, name, a, marg['defaults'])
                break
        if target:
            break
    if target is None:
        print('❌ 没找到测验附件（可指定 courseid / chapterId 再试）')
        return 1

    task_id, name, att, defaults = target
    print('📝 章节 %s [%s]' % (task_id, name))
    html, page_url = user._ChaoXing__fetch_quiz_html(att, defaults)
    graded = 'marking_dui' in html or 'marking_cuo' in html or 'Finalresult' in html
    if not graded:
        print('⚠️ 这份测验是未作答状态，批阅页明文真值拿不到；'
              '只打印解码结果供人工核对（不提交）')
        truth = []
    else:
        truth = _stems(html)
        print('✅ 已批阅页取到明文真值 %d 条，重置以获取加密题面...' % len(truth))
        user._ChaoXing__redo_quiz(page_url)
        html, page_url = user._ChaoXing__fetch_quiz_html(att, defaults)

    font = user._ChaoXing__extract_secret_font(html)
    if not font:
        print('❌ 题面里没有内嵌字体 —— 这卷未加密，无法验证解密（换一份试试）')
        return 1
    mapping = user._ChaoXing__build_secret_map(font)
    print('🔤 字体 %d 字节，映射 %d 条' % (len(font), len(mapping)))

    questions = user._ChaoXing__parse_quiz(html, mapping)
    print('🔤 解析出 %d 题' % len(questions))
    if not truth:
        for i, q in enumerate(questions[:5]):
            print('   题%d: %s' % (i + 1, q['stem'][:70]))
        print('ℹ️ 无真值可对照，请肉眼确认上面句子是否通顺')
        return 0

    n = min(len(truth), len(questions))
    ok = 0
    for i in range(n):
        same = questions[i]['stem'] == truth[i]
        ok += same
        print('%s 题%d' % ('✅' if same else '❌', i + 1))
        if not same:
            print('   真值: %s' % truth[i][:90])
            print('   解码: %s' % questions[i]['stem'][:90])
            diff = [(a, b) for a, b in zip(truth[i], questions[i]['stem']) if a != b]
            print('   差异字: %s' % ' '.join('%s≠%s' % (a, b) for a, b in diff[:15]))
    print('\n结果：%d/%d 题解码与明文完全一致' % (ok, n))
    if ok == n:
        print('✅ 解密准确')
        return 0
    print('❌ 存在解码错误，请把上面输出反馈给作者')
    return 1


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '--offline':
        sys.exit(offline())
    if len(args) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(online(args[0], args[1],
                    args[2] if len(args) > 2 else None,
                    args[3] if len(args) > 3 else None))
