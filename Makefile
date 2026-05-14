# Prefer virtualenv python at .venv/bin/python when available
ifneq ($(wildcard .venv/bin/python),)
PY := .venv/bin/python
else
PY ?= python
endif

IMAGE_CHATGPT = debug_chatgpt_images

.PHONY: help chatgpt_upload_to_notion chatgpt_upload_to_notion_remove clean-output-path

help:
	@echo "Available targets:"
	@echo "  chatgpt_upload_to_notion           Run chatgpt-upload-to-notion (no-remove)"
	@echo "  chatgpt_upload_to_notion_remove    Run chatgpt-upload-to-notion (remove)"
	@echo "  clean-output-path                  Run clean-output-path (module)"

chatgpt_upload_to_notion:
	$(PY) main.py chatgpt-upload-to-notion --image-folder $(IMAGE_CHATGPT) --no-remove-in-chatgpt

chatgpt_upload_to_notion_remove:
	$(PY) main.py chatgpt-upload-to-notion --image-folder $(IMAGE_CHATGPT) --remove-in-chatgpt

clean-output-path:
	$(PY) -m main clean-output-path
