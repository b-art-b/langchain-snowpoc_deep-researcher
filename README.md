# LangGraph in Snowflake for a Deep Researcher

> **NOTE**: This is just a PoC, not production-ready and covers just a few use cases.

This is a PoC of how can Cortex be used with LangGraph
to implement a Deep Researcher.

It shows how easy it is to integrate LangGraph, Snowflake
Cortex, Streamlit and external data providers.

## Setting Up the Research Assistant

To deploy and start using the Research Assistant, follow these steps:

1. Create a [Tavily](https://tavily.com/) Account:
   * Sign up at Tavily and obtain an API key to access their services.

2. Configure Your Key:
   * Insert the API key into `setup_01.sql` by replacing `PUT YOUR KEY HERE`.

3. Run Initial Setup:
   * Execute `setup_01.sql` using an `ACCOUNTADMIN` role or a dedicated role of your choice.

4. Prepare Local Packages:
   * Add new connection to `~/.snowflake/connections.toml`. I have called mine `langchain_deep_researcher`.
   * Build necessary packages locally and upload `mylangchain.zip` to Snowflake using provided `Makefile`:

    ```bash
    > make all
    ```

    For help:

    ```bash
    ❯ make help
    all: clean, install, package and upload. All you need
    clean: Clean local resources
    help: Show this help.
    install: Install packages locally
    package: Build a zip package locally
    upload: Upload a package to a stage
    ```

5. Create a Streamlit Application in Snowsight:
   * Name the app `Deep_Researcher` and link it to `DEEP_RESEARCHER_WH` and `DEEP_RESEARCHER_DB.PUBLIC`.

6. Add Dependencies:
   * Include `@packages/mylangchain.zip` in the `Stage Packages` section.
   * Add the required packages listed in `packages.txt` to the `Anaconda Packages` section of the Streamlit GUI.

7. Finalize Configuration:
   * Run `SHOW STREAMLITS IN SCHEMA DEEP_RESEARCHER_DB.PUBLIC` to retrieve the name of your application.
   * Use this name in `setup_02.sql` and execute it to complete the setup.

8. Deploy Streamlit Code:
   * Copy and paste the contents of `streamlit_app.py` into the application’s code section within Streamlit.

> **Note**: This repo is for Medium blog post: URL
