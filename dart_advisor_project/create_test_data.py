import pandas as pd
from pathlib import Path

# 간단한 재무제표 데이터 생성
data = {
    '계정과목': ['매출액', '영업이익', '당기순이익', '자산총계', '부채총계', '자본총계', '영업현금흐름'],
    '2019년': [1000, 150, 100, 2000, 800, 1200, 120],
    '2020년': [1200, 180, 120, 2400, 900, 1500, 150],
    '2021년': [1500, 240, 180, 2800, 950, 1850, 200],
    '2022년': [1800, 300, 220, 3200, 1000, 2200, 250],
    '2023년': [2200, 380, 280, 3800, 1100, 2700, 300]
}

df = pd.DataFrame(data)

# Excel 파일로 저장
output_path = Path('examples/sample_financials.xlsx')
output_path.parent.mkdir(exist_ok=True)
df.to_excel(output_path, index=False, sheet_name='재무제표')

print(f"✓ Created: {output_path}")
print("\n재무제표 미리보기:")
print(df.to_string())
