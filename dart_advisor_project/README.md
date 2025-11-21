# DART Advisor

> AI-Powered Investment Analysis Platform

## 🚀 Quick Start

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 설정
cp .env.example .env
# .env 파일을 열어 ANTHROPIC_API_KEY 입력

# 4. 초기화
python -m dart_advisor.main init

# 5. 테스트
python -m dart_advisor.main --help
```

## 📁 Project Structure

```
dart_advisor_project/
├── dart_advisor/           # Main package
│   ├── config/            # Configuration
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── ingestion/         # Document parsing
│   │   └── __init__.py
│   ├── analysis/          # Data analysis
│   │   └── __init__.py
│   ├── llm/               # LLM integration
│   │   └── __init__.py
│   ├── report/            # Report generation
│   │   └── __init__.py
│   ├── utils/             # Utilities
│   │   ├── __init__.py
│   │   └── logger.py
│   ├── __init__.py
│   └── main.py            # Entry point
├── tests/                 # Test files
├── examples/              # Example scripts
├── output/                # Generated reports
├── logs/                  # Log files
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🔧 Configuration

Edit `.env` file:

```bash
ANTHROPIC_API_KEY=your_api_key_here
LOG_LEVEL=INFO
LOG_FILE=logs/dart_advisor.log
OUTPUT_DIR=output
CACHE_DIR=.cache
CLAUDE_MODEL=claude-sonnet-4-20250514
MAX_TOKENS=8192
TEMPERATURE=0.3
```

## 📚 Features

- 📄 Multi-format document parsing (PDF, DOCX, XLSX)
- 🤖 AI-powered financial analysis using Claude
- 📊 Automated report generation
- 🔍 Intelligent data extraction
- 📈 Investment insights and recommendations

## 🛠️ Development

### Adding New Modules

Follow the structure for additional modules:

1. `dart_advisor/config/prompts.py` - Prompt templates
2. `dart_advisor/ingestion/document_parser.py` - Document parser
3. `dart_advisor/llm/claude_client.py` - Claude client
4. `dart_advisor/ingestion/financial_extractor.py` - Financial extraction

### Running Tests

```bash
pytest tests/
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
