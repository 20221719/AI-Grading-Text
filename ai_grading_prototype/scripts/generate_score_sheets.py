from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from decimal import Decimal, ROUND_HALF_UP

from fpdf import FPDF


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "output" / "pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")


STUDENTS: List[Dict] = [
    {
        "name": "李五",
        "slug": "liwu",
        "unit": "吉林大学",
        "exam_no": "20231312",
        "source": RAW_DIR / "6e42db39c7e2318b6f548c8a4e860732.jpg",
        "scores": [
            {
                "q": "17",
                "full": 10,
                "passes": [1, 2, 1, 2, 1, 2, 2, 1, 2, 1],
                "comment": "只写出了 sec x = 1/cos x 的起步，后续变形与最终结果不成立。",
            },
            {
                "q": "18",
                "full": 12,
                "passes": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                "comment": "积分变形与换元思路不正确，最终结果未到标准答案。",
            },
            {
                "q": "19",
                "full": 12,
                "passes": [4, 5, 4, 5, 4, 5, 4, 5, 4, 5],
                "comment": "能想到辅助函数思路，但未完整完成罗尔定理证明。",
            },
        ],
    },
    {
        "name": "胡四",
        "slug": "husi",
        "unit": "吉林大学",
        "exam_no": "14769264",
        "source": RAW_DIR / "a327319400a3140638287a976e5eb55f.png",
        "scores": [
            {
                "q": "17",
                "full": 10,
                "passes": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                "comment": "思路正确，等价变形与对数结果到位，属于满分解法。",
            },
            {
                "q": "18",
                "full": 12,
                "passes": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "comment": "代换与积分处理不正确，最终答案与标准答案不符。",
            },
            {
                "q": "19",
                "full": 12,
                "passes": [5, 5, 4, 5, 4, 5, 4, 5, 4, 5],
                "comment": "有辅助函数与导数方向，但证明链条没有完整闭合。",
            },
        ],
    },
    {
        "name": "张三",
        "slug": "zhangsan",
        "unit": "吉林大学",
        "exam_no": "20221719",
        "source": RAW_DIR / "d76228c26ee247d9225f70250b6d7ccf.png",
        "scores": [
            {
                "q": "17",
                "full": 10,
                "passes": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                "comment": "标准换元思路清楚，最终结果正确。",
            },
            {
                "q": "18",
                "full": 12,
                "passes": [12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                "comment": "换元、化简、回代和最终结果都正确。",
            },
            {
                "q": "19",
                "full": 12,
                "passes": [12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                "comment": "辅助函数、罗尔定理与结论都完整，证明成立。",
            },
        ],
    },
]


def round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def summarize_passes(passes: List[int]) -> Dict[str, object]:
    avg = sum(passes) / len(passes)
    final = round_half_up(avg)
    spread = max(passes) - min(passes)
    review = spread > 5
    return {"avg": avg, "final": final, "spread": spread, "review": review}


def build_pdf(student: Dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_font("SimHei", "", str(FONT_PATH), uni=True)
    pdf.set_font("SimHei", size=16)

    total = sum(summarize_passes(item["passes"])["final"] for item in student["scores"])
    total_full = sum(item["full"] for item in student["scores"])

    pdf.add_page()
    pdf.set_font("SimHei", size=18)
    pdf.cell(0, 10, "数学答题卡成绩单", ln=1, align="C")
    pdf.ln(2)

    pdf.set_font("SimHei", size=12)
    pdf.cell(0, 8, f"报考单位：{student['unit']}", ln=1)
    pdf.cell(0, 8, f"考生姓名：{student['name']}", ln=1)
    pdf.cell(0, 8, f"准考证号：{student['exam_no']}", ln=1)
    pdf.cell(0, 8, f"评分说明：基于当前图像中可辨识内容进行人工估分", ln=1)
    pdf.ln(2)

    pdf.set_fill_color(240, 240, 240)
    pdf.cell(18, 9, "题号", border=1, fill=True, align="C")
    pdf.cell(24, 9, "得分", border=1, fill=True, align="C")
    pdf.cell(24, 9, "满分", border=1, fill=True, align="C")
    pdf.cell(124, 9, "评语", border=1, fill=True, align="C", ln=1)

    pdf.set_font("SimHei", size=11)
    for item in student["scores"]:
        row_y = pdf.get_y()
        summary = summarize_passes(item["passes"])
        item["score"] = summary["final"]
        pdf.cell(18, 14, item["q"], border=1, align="C")
        pdf.cell(24, 14, str(item["score"]), border=1, align="C")
        pdf.cell(24, 14, str(item["full"]), border=1, align="C")
        pdf.set_xy(10 + 18 + 24 + 24, row_y)
        pdf.multi_cell(124, 7, item["comment"], border=1)
        pdf.set_xy(10, row_y + 14)

    pdf.ln(4)
    pdf.set_font("SimHei", size=13)
    pdf.cell(0, 8, f"总分：{total}/{total_full}", ln=1)
    pdf.set_font("SimHei", size=10)
    pdf.set_x(10)
    pdf.multi_cell(190, 6, "十次评分明细：")
    for item in student["scores"]:
        summary = summarize_passes(item["passes"])
        pass_text = ",".join(str(x) for x in item["passes"])
        flag_text = "【人工复核】" if summary["review"] else ""
        pdf.set_x(10)
        pdf.multi_cell(
            190,
            6,
            f"{item['q']}题：[{pass_text}]，平均 {summary['avg']:.2f} -> {summary['final']}，极差 {summary['spread']} {flag_text}",
        )
    pdf.ln(1)
    pdf.set_x(10)
    pdf.multi_cell(
        190,
        6,
        "备注：本成绩单按当前可辨识内容十次独立评分，并取平均后四舍五入为整数。原图更清晰时可重新调整。",
    )

    if student["source"].exists():
        pdf.add_page()
        pdf.set_font("SimHei", size=14)
        pdf.cell(0, 10, "原始答题卡参考图", ln=1, align="C")
        page_w = 210 - 20
        img_w = page_w
        pdf.image(str(student["source"]), x=10, y=22, w=img_w)

    out_path = OUT_DIR / f"{student['slug']}_score_sheet.pdf"
    pdf.output(str(out_path))
    return out_path


def main() -> None:
    paths = []
    for student in STUDENTS:
        paths.append(build_pdf(student))
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
