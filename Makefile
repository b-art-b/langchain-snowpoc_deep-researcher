PKGDIR=mylangchain
STAGE=packages
CONNECTION=langchain_deep_researcher

all: clean install package upload     # clean, install, package and upload. All you need

help:           # Show this help.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

install:        # Install packages locally
	pip install --no-binary=true --no-deps --target=$(PKGDIR) \
	. \
    langchain==0.3.14 \
    langchain-community==0.3.14 \
    langchain-core==0.3.30 \
    langchain-openai==0.3.1 \
    langchain-text-splitters==0.3.5 \
    langgraph==0.2.64 \
    langgraph-checkpoint==2.0.10 \
    langgraph-sdk==0.1.51 \
    langsmith==0.2.10 \
    tavily-python==0.5.0

clean:          # Clean local resources
	test -d $(PKGDIR) && rm -fR $(PKGDIR) || true
	test -e $(PKGDIR).zip && rm -fR $(PKGDIR).zip || true
	test -d build && rm -fR build || true
	test -d langchain_snowpoc.egg-info && rm -fR langchain_snowpoc.egg-info || true

package:        # Build a zip package locally
	cd $(PKGDIR) && zip -r ../$(PKGDIR).zip .

upload:         # Upload a package to a stage
	snow snowpark package upload -o -f $(PKGDIR).zip -s $(STAGE) -c $(CONNECTION)
