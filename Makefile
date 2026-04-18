.PHONY: setup set verify-cuda-kernels preprocess train train-resume eval eval-benchmarks eval-benchmarks-base eval-benchmarks-both show-config push-to-hub inspect-step

PYTHON ?= python3
config ?= config
eval_config ?= config
limit ?= 400
ovr ?=
HF_REPO ?=
CKPT ?= final
HF_PRIVATE ?= false
SKIP_CAUSAL_CONV1D ?= 0

define WITH_TORCH_LIB
TORCH_LIB_DIR="$$( $(PYTHON) -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' )"; \
export LD_LIBRARY_PATH="$$TORCH_LIB_DIR:$${LD_LIBRARY_PATH:-}"; \
$(1)
endef

setup:
	$(PYTHON) -m pip install -e . --no-deps -q
	$(PYTHON) -m pip install -U huggingface_hub -q
	$(PYTHON) -m pip install "transformers==5.5.4" "trl>=0.15.0" --no-deps -q
	$(PYTHON) -c "import transformers; print('  transformers_version:', transformers.__version__)"
	$(PYTHON) -m pip install "hydra-core>=1.3.2" "omegaconf>=2.3.0" -q
	@if $(PYTHON) -m pip install xformers -q; then \
		echo "xformers_install_ok=true"; \
	else \
		echo "xformers_install_ok=false"; \
	fi
	@if $(PYTHON) -m pip install weave -q; then \
		echo "weave_install_ok=true"; \
	else \
		echo "weave_install_ok=false"; \
	fi
	$(PYTHON) -m pip install --upgrade unsloth unsloth-zoo --no-deps -q
	@if [ "$(SKIP_CAUSAL_CONV1D)" = "1" ]; then \
		echo "  skip causal_conv1d setup (SKIP_CAUSAL_CONV1D=1)"; \
	else \
		PYTHON=$(PYTHON) bash scripts/ensure_causal_conv1d.sh; \
	fi
	$(PYTHON) -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" 2>/dev/null || $(PYTHON) -m pip install flash-linear-attention -q
	@$(PYTHON) -c "import torch; print('  flash_sdp:', torch.backends.cuda.flash_sdp_enabled())"
	$(PYTHON) -m pip install lm-eval -q 2>/dev/null || true

set: setup verify-cuda-kernels

verify-cuda-kernels:
	@if [ "$(SKIP_CAUSAL_CONV1D)" = "1" ]; then \
		echo "  skip CUDA kernel verification (SKIP_CAUSAL_CONV1D=1)"; \
	else \
		PYTHON=$(PYTHON) bash scripts/verify_cuda_kernels.sh; \
	fi

preprocess:
	@$(call WITH_TORCH_LIB,$(PYTHON) -m src.preprocess --config-path configs --config-name $(config) $(ovr))

train:
	@$(call WITH_TORCH_LIB,$(PYTHON) -m src.train --config-path configs --config-name $(config) $(ovr))

train-resume:
	@$(call WITH_TORCH_LIB,$(PYTHON) -m src.train --config-path configs --config-name $(config) training.resume_from_checkpoint=auto $(ovr))

eval:
	@$(call WITH_TORCH_LIB,$(PYTHON) -m src.evaluate --config-path configs --config-name $(eval_config) --limit $(limit))

eval-benchmarks:
	@$(call WITH_TORCH_LIB,$(PYTHON) -m src.evaluate --config-path configs --config-name $(eval_config) --benchmarks_only --bench_target cpt --limit $(limit))

eval-benchmarks-base:
	@$(call WITH_TORCH_LIB,$(PYTHON) -m src.evaluate --config-path configs --config-name $(eval_config) --benchmarks_only --bench_target base --limit $(limit))

eval-benchmarks-both:
	@$(call WITH_TORCH_LIB,$(PYTHON) -m src.evaluate --config-path configs --config-name $(eval_config) --benchmarks_only --bench_target both --limit $(limit))

show-config:
	@$(call WITH_TORCH_LIB,$(PYTHON) -m src.train --config-path configs --config-name $(config) $(ovr) --cfg job)

push-to-hub:
	@if [ -z "$(HF_REPO)" ]; then echo "HF_REPO is required. Example: make push-to-hub HF_REPO=your-name/your-model"; exit 1; fi
	@$(call WITH_TORCH_LIB,$(PYTHON) -m src.push_to_hub --config-path configs --config-name $(config) --repo $(HF_REPO) --checkpoint $(CKPT) $(if $(filter true,$(HF_PRIVATE)),--private,))
