# Prefer virtualenv python at .venv/bin/python when available
ifneq ($(wildcard .venv/bin/python),)
PY := .venv/bin/python
else
PY ?= python
endif

IMAGE_CHATGPT = debug_chatgpt_images

.PHONY: help upload_to_notion upload_to_notion_remove clean-output-path

help:
	@echo "Available targets:"
	@echo "  upload_to_notion                   Run upload-to-notion (no-remove)"
	@echo "  upload_to_notion_remove            Run upload-to-notion (remove)"
	@echo "  clean-output-path                  Run clean-output-path (module)"

upload_to_notion:
	$(PY) main.py upload-to-notion --image-folder $(IMAGE_CHATGPT) --no-remove

upload_to_notion_remove:
	$(PY) main.py upload-to-notion --image-folder $(IMAGE_CHATGPT) --remove

clean-output-path:
	$(PY) -m main clean-output-path
