TOPDIR = $(shell pwd)

build_SVMs: build_VSharp build_usvm

build_maps: build_dotnet_maps build_manually_collected_java_maps build_LogicNG build_jflex

build_VSharp:
	cd ./GameServers/VSharp/; \
	dotnet build VSharp.ML.GameServer.Runner -c Release

build_dotnet_maps:
	cd ./maps/DotNet/Maps; \
	dotnet build Root -c Release

build_usvm:
	cd ./GameServers/usvm/; \
	./gradlew assemble --no-daemon

build_manually_collected_java_maps:
	cd ./maps/Java/Maps; \
	./gradlew build

build_LogicNG:
	cd ./maps/Java/Maps/LogicNG; \
	mvn package -Dmaven.test.skip -Dmaven.javadoc.skip=true

build_jflex:
	cd ./maps/Java/Maps/jflex; \
	mvn package -Dmaven.test.skip -Dmaven.javadoc.skip=true -Dfmt.skip

check_submodules:
	@fail=0; \
	for path in $$(git config -f .gitmodules --get-regexp '^submodule\..*\.path$$' | awk '{print $$2}'); do \
		if [ ! -e "$$path/.git" ]; then \
			echo "UNINITIALIZED: $$path"; \
			fail=1; \
			continue; \
		fi; \
		deleted=$$(git -C "$$path" status --porcelain | grep -c '^D\|^ D' || true); \
		if [ "$$deleted" -gt 0 ]; then \
			echo "CORRUPTED: $$path ($$deleted deleted file(s); repair with: git -C $$path reset --hard HEAD)"; \
			fail=1; \
		else \
			echo "OK: $$path"; \
		fi; \
	done; \
	exit $$fail

# Pin stability: every submodule pointer must name a commit reachable from a
# stable ref (a tag, or the default branch) of the submodule's remote, so a
# force-push or branch deletion upstream cannot break `git submodule update`.
# Policy: docs/developer.rst, "Submodules".
check_submodule_pins:
	@fail=0; \
	for name in $$(git config -f .gitmodules --get-regexp '^submodule\..*\.path$$' | awk '{print $$1}' | sed 's/^submodule\.//;s/\.path$$//'); do \
		path=$$(git config -f .gitmodules --get "submodule.$${name}.path"); \
		url=$$(git config -f .gitmodules --get "submodule.$${name}.url"); \
		pin=$$(git ls-tree HEAD "$$path" | awk '{print $$3}'); \
		if [ -z "$$pin" ]; then \
			echo "MISSING: $$path (no gitlink in HEAD)"; \
			fail=1; continue; \
		fi; \
		default=$$(git ls-remote --symref "$$url" HEAD 2>/dev/null | awk '/^ref:/{print $$2}' | sed 's@^refs/heads/@@'); \
		if [ -z "$$default" ]; then \
			echo "UNRESOLVED: $$path (no default branch advertised by $$url)"; \
			fail=1; continue; \
		fi; \
		tmp=$$(mktemp -d); \
		git init --bare -q "$$tmp"; \
		if ! git -C "$$tmp" fetch -q --filter=blob:none "$$url" "refs/heads/$$default:refs/stable/default" "+refs/tags/*:refs/tags/*" 2>/dev/null; then \
			echo "UNREACHABLE: $$path (cannot fetch $$url)"; \
			rm -rf "$$tmp"; fail=1; continue; \
		fi; \
		ok=0; \
		for ref in $$(git -C "$$tmp" for-each-ref --format='%(objectname)' refs/stable/default 'refs/tags/*'); do \
			commit=$$(git -C "$$tmp" rev-parse -q --verify "$$ref^{commit}" 2>/dev/null); \
			if [ -n "$$commit" ] && git -C "$$tmp" merge-base --is-ancestor "$$pin" "$$commit" 2>/dev/null; then \
				ok=1; break; \
			fi; \
		done; \
		rm -rf "$$tmp"; \
		if [ "$$ok" -eq 1 ]; then \
			echo "OK: $$path ($$(echo $$pin | cut -c1-8) reachable from a stable ref of $$url)"; \
		else \
			echo "UNSTABLE: $$path ($$(echo $$pin | cut -c1-8) not reachable from any tag or from $$default of $$url)"; \
			fail=1; \
		fi; \
	done; \
	exit $$fail

init_data: generate_init_data move_init_data

STEPS_TO_SERIALIZE = 200

generate_init_data:
	cd ./GameServers/VSharp/VSharp.ML.GameServer.Runner/bin/Release/net8.0; \
	dotnet VSharp.ML.GameServer.Runner.dll --mode generator --datasetbasepath $(TOPDIR)/maps/DotNet/Maps/Root/bin/Release/net8.0 --datasetdescription $(TOPDIR)/maps/DotNet/Maps/dataset.json --stepstoserialize $(STEPS_TO_SERIALIZE)

move_init_data:
	cd ./AIAgent; \
	mkdir -p report; \
	mv ../GameServers/VSharp/VSharp.ML.GameServer.Runner/bin/Release/net8.0/8100/SerializedEpisodes/ report/

# Test system (see docs/testing.rst)

POETRY ?= poetry
PYTEST ?= $(POETRY) run pytest
COV_ARGS = --cov=AIAgent --cov=tools --cov-report=term-missing --cov-report=xml --cov-report=json

.PHONY: test-unit test-integration test-all test-cov

# Fast unit tier: no network, GPU, .NET or binary fixtures.
test-unit:
	$(PYTEST) -m "not integration and not e2e"

# In-process integration tier (fakes + golden fixtures).
test-integration:
	$(PYTEST) -m integration

# Everything, including the e2e pipelines (needs the built toolchain).
test-all:
	$(PYTEST)

# Unit + integration tier with the coverage ratchet (the gated tier).
test-cov:
	$(PYTEST) -m "not e2e" $(COV_ARGS)
