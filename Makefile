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

init_data: generate_init_data move_init_data

STEPS_TO_SERIALIZE = 200

generate_init_data:
	cd ./GameServers/VSharp/VSharp.ML.GameServer.Runner/bin/Release/net7.0; \
	dotnet VSharp.ML.GameServer.Runner.dll --mode generator --datasetbasepath $(TOPDIR)/maps/DotNet/Maps/Root/bin/Release/net7.0 --datasetdescription $(TOPDIR)/maps/DotNet/Maps/dataset.json --stepstoserialize $(STEPS_TO_SERIALIZE)

move_init_data:
	cd ./AIAgent; \
	mkdir -p report; \
	mv ../GameServers/VSharp/VSharp.ML.GameServer.Runner/bin/Release/net7.0/8100/SerializedEpisodes/ report/
