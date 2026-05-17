# Prefer virtualenv python at .venv/bin/python when available
ifneq ($(wildcard .venv/bin/python),)
PY := .venv/bin/python
else
PY ?= python
endif

IMAGE_CHATGPT = debug_chatgpt_images

.PHONY: help upload_to_notion account-status

help:
	@echo "Available targets:"
	@echo "  upload_to_notion                   Run upload-to-notion (module)"
	@echo "  account-status                     Run account-status (module)"

upload_to_notion:
	$(PY) -m chatgpt_to_notion.cli.app upload-to-notion --image-folder $(IMAGE_CHATGPT) --no-remove

account_status:
	$(PY) -m chatgpt_to_notion.cli.app account-status
