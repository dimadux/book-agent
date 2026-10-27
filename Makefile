.PHONY: setup run verify

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@[ -f .env ] || cp .env.example .env
	@echo ""
	@echo "✅ Setup complete. Now edit .env with your API keys, then run: make verify"

run:
	.venv/bin/python scripts/run_dev.py

verify:
	.venv/bin/python scripts/verify_setup.py
