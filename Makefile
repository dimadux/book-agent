.PHONY: setup run verify

setup:
	uv sync
	@[ -f .env ] || cp .env.example .env
	@echo ""
	@echo "✅ Setup complete. Now edit .env with your API keys, then run: make verify"

run:
	uv run scripts/run_dev.py

verify:
	uv run scripts/verify_setup.py
